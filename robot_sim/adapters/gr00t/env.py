import json
from functools import partial
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from loguru import logger

from robot_sim.backends.types import ArrayType
from robot_sim.configs import CameraConfig, ObjectConfig, ObjectType, RobotModel, SensorType, SimulatorConfig
from robot_sim.controllers import CompositeController, PIDController
from robot_sim.envs import MapCache, MapEnv
from robot_sim.utils.config import configclass

from .policy import (
    DecoupledWBCPolicy,
    LowerBodyPolicy,
    UpperBodyPolicy,
)
from .utils import act_joint_assign, obs_joint_extract


@configclass
class Gr00tTaskConfig:
    """Configuration for Map pick-and-place task."""

    task: str
    """Task name for Gr00t environment."""
    params: dict
    """Parameters for the specific task."""
    maps: dict[str, Any]
    """Maps configuration for observation, action, reward, termination, truncation, and info maps."""
    simulator: SimulatorConfig
    """Simulator configuration for MapEnv."""


class Gr00tEnv(MapEnv):
    """Gr00t Whole Body Control Environment.

    This environment uses a composite controller to manage different control strategies for the Gr00t robot.
    The composite controller routes commands to sub-controllers based on the current state and targets.
    Here, we define the environment setup and controller initialization specific to the Gr00t robot.
    Especially the maps for the definition of modality config in Gr00t.

    """

    def __init__(
        self,
        config: SimulatorConfig,
        maps: dict[str, Any],
        **kwargs,
    ):
        robot_config = [cfg for cfg in config.scene.objects.values() if cfg.type == ObjectType.ROBOT]
        assert len(robot_config) == 1, "Gr00tEnv only supports single robot."
        controllers = self.create_controllers(robot_config[0])

        super().__init__(config=config, controllers=controllers, **kwargs)
        self.robot_name = self.robot_names[0]
        self.robot = self.robots[self.robot_name]
        self._map_config = maps
        self._map_cache: MapCache = self._init_spaces_maps(**maps)

    def create_controllers(self, robot_cfg: ObjectConfig) -> dict[str, CompositeController]:
        # Initialize PD controller for low-level control
        controllers = {}
        coeff: float = 0.9
        robot = RobotModel(robot_cfg)
        for name, robot in self.robots.items():
            kp = robot.stiffness
            kd = robot.damping
            tor_limits = robot.get_joint_limits("torque", coeff=coeff)
            pd_controller = PIDController(kp=kp, kd=kd, dt=self.step_dt / self.decimation)
            controllers[name] = CompositeController(
                controllers={"pd_controller": pd_controller}, output_clips={"pd_controller": tor_limits}
            )
        return controllers

    def _init_spaces_maps(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        policy: dict[str, Any],
    ) -> MapCache:
        obs_maps = self._init_observation_spaces_map(**observation)
        logger.info(f"Observation maps order: {list(obs_maps.keys())}")

        act_maps = self._init_action_spaces_map(**action)
        act_maps[self.robot_name] = self._init_policy(**policy)
        logger.info(f"Action maps order: {list(act_maps.keys())}")

        return MapCache(
            observation=obs_maps,
            action=act_maps,
        )

    def _init_policy(
        self,
        upper_policy: dict[str, Any],
        lower_policy: dict[str, Any],
    ) -> Callable:
        upper_body_policy = UpperBodyPolicy(**upper_policy)
        lower_body_policy = self._init_lower_policy(**lower_policy)

        pos_clips = self.robot.get_joint_limits("position", coeff=0.9)
        wbc_policy = DecoupledWBCPolicy(
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
            output_clips=pos_clips,
            output_indices=None,
        )

        return lambda name, states, action: wbc_policy(name, states, action)

    ############################################################################
    ########## Helper Functions for Maps Initialization and Callbacks ##########
    ############################################################################

    def _init_observation_spaces_map(self, **kwargs) -> dict[str, Callable]:
        """
        Initialize the observation spaces map for the Gr00t environment.
        Returns:
            A dictionary mapping observation group names to their configurations.
        """
        joint_position_limit = self.robot.get_joint_limits("position")
        obs_map: dict[str, Callable] = {}
        _spaces: gym.spaces.Space

        for group_name, group_cfg in kwargs.items():
            if group_cfg["type"] == "joint":
                group_indices = group_cfg["indices"]
                epsilon: float = group_cfg.get("epsilon", 1e-3)
                obs_map[group_name] = partial(obs_joint_extract, indices=group_indices)
                _spaces = gym.spaces.Box(
                    low=joint_position_limit[0][group_indices] - epsilon,
                    high=joint_position_limit[1][group_indices] + epsilon,
                    shape=(len(group_indices),),
                    dtype=np.float32,
                )

            elif group_cfg["type"] == "sensor":
                sensor_name = group_cfg["sensor_name"]
                sensor_data_type = group_cfg.get("data_type", None)
                sensor_cfg = self.robot.sensors[sensor_name]
                # map
                if sensor_data_type:
                    obs_map[group_name] = lambda name, states, sn=sensor_name, dt=sensor_data_type: states[
                        name
                    ].sensors[sn][dt]
                else:
                    obs_map[group_name] = lambda name, states, sn=sensor_name: states[name].sensors[sn]
                # _spaces
                if sensor_cfg.type in [SensorType.CAMERA]:
                    assert isinstance(sensor_cfg, CameraConfig), "Sensor config must be of type CameraConfig."
                    _spaces = gym.spaces.Box(
                        low=0,
                        high=255,
                        # Assuming fixed camera resolution; adjust as needed
                        shape=(sensor_cfg.height, sensor_cfg.width, 3),
                        dtype=np.uint8,
                    )
                else:
                    raise NotImplementedError

            elif group_cfg["type"] == "constant":
                value = group_cfg["value"]
                obs_map[group_name] = lambda *args, **kwargs: value
                if isinstance(value, str):
                    allowed_language_charset = (
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
                    )
                    _spaces = gym.spaces.Text(
                        max_length=group_cfg.get("max_length", 1024),
                        charset=group_cfg.get("charset", allowed_language_charset),
                    )
                else:
                    raise NotImplementedError

            else:
                raise ValueError(f"Unsupported observation map type '{group_cfg['type']}' for group '{group_name}'.")

            self.observation_space[group_name] = _spaces

        return obs_map

    def _init_action_spaces_map(self, **kwargs) -> dict[str, Callable]:
        joint_position_limit = self.robot.get_joint_limits("position")
        action_map: dict[str, Callable] = {}
        _spaces: gym.spaces.Space

        for group_name, group_cfg in kwargs.items():
            if group_cfg["type"] == "joint":
                epsilon: float = group_cfg.get("epsilon", 1e-3)
                group_indices = group_cfg["indices"]
                action_map[group_name] = partial(act_joint_assign, group_name=group_name, indices=group_indices)
                _spaces = gym.spaces.Box(
                    low=joint_position_limit[0][group_indices] - epsilon,
                    high=joint_position_limit[1][group_indices] + epsilon,
                    shape=(len(group_indices),),
                    dtype=np.float32,
                )

            elif group_cfg["type"] == "command":
                action_map[group_name] = lambda *args, **kwargs: None  # No processing needed; direct assignment
                bound = group_cfg["bound"]
                command_dim = len(bound["min"])
                _spaces = gym.spaces.Box(
                    low=np.array(bound["min"], dtype=np.float32),
                    high=np.array(bound["max"], dtype=np.float32),
                    shape=(command_dim,),
                    dtype=np.float32,
                )

            else:
                raise ValueError(f"Unsupported action map type '{group_cfg['type']}' for group '{group_name}'.")

            self.action_space[group_name] = _spaces

        return action_map

    ####################################################################
    ########## Helper Functions for Controller Initialization ##########
    ####################################################################
    def _init_lower_policy(
        self,
        actuator_indices: list[int],
        hand_indices: list[int],
        observation_params: dict[str, Any],
        use_rpy_cmd_from_waist: bool = True,
        **kwargs,
    ) -> LowerBodyPolicy:
        num_dofs = self.robot.num_dofs
        used_joint_indices_in_env = np.array([i for i in range(num_dofs) if i not in hand_indices], dtype=np.int32)

        observation_params["used_joint_indices"] = used_joint_indices_in_env
        observation_params["default_joint_position"] = self.robot.default_joint_positions
        torso_index = self.get_body_names(self.robot_name).index("torso_link")
        pelvis_index = self.get_body_names(self.robot_name).index("pelvis")

        lower_body_policy = LowerBodyPolicy(
            actuator_indices=actuator_indices,
            observation_params=observation_params,
            use_rpy_cmd_from_waist=use_rpy_cmd_from_waist,
            torso_index=torso_index,
            pelvis_index=pelvis_index,
            **kwargs,
        )
        return lower_body_policy

    ####################################################################


####################################################
### Teleoperation Wrapper #########################
#####################################################
class Gr00tTeleopWrapper(gym.Wrapper):
    """Gr00t Teleoperation task for Gr00t adapter using Pico Ultra 4 and Motion Tracker."""

    def __init__(
        self,
        env: gym.Env,
        robot_name: str,
        body2part_indices: dict[str, list[int] | ArrayType],
        redis_config: dict[str, Any],
        command_index: list[int] = [0, 1, 2, 3, 4, 5],
        command_scale: float | list[float] = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(env)
        self.robot_name: str = robot_name
        assert self.robot_name, "TeleoperationTask requires the env to have a 'robot_name' attribute."
        logger.info(f"TeleoperationTask with robot: {self.robot_name}")
        for k, v in body2part_indices.items():
            body2part_indices[k] = np.array(v, dtype=np.int32)
        self.body2part_indices = body2part_indices
        self.command_index = command_index
        self.command_scale = np.array(command_scale)
        assert self.command_scale.shape == (6,) or self.command_scale.shape == (), (
            "command_scale must be a float or a list of 6 floats or a scalar."
        )
        self._init_redis(**redis_config)

    def get_action(self) -> dict[str, ArrayType]:
        # Get mimic obs from Redis
        for k in self.redis_recv_keys:
            self.redis_pipeline.get(k)
        redis_results = self.redis_pipeline.execute()
        for i, k in enumerate(self.redis_recv_keys):
            self.__recv_buffer[k] = json.loads(redis_results[i])
        # mimic: xy velocity, z position, roll/pitch, yaw angular velocity, 29 dof
        # left hand: 7 dof
        # right hand: 7 dof
        # nav + rpy cmd: 6 dims
        dof_pos, nav_cmd, height_cmd, rpy_cmd = self.unpack_action()
        action = {
            "action.navigate_command": nav_cmd,
            "action.base_height_command": height_cmd,
            # "action.rpy_command": rpy_cmd,
        }
        for k, v in self.body2part_indices.items():
            action[k] = dof_pos[..., v]

        return action

    def unpack_action(self):
        action_mimic = np.array(self.__recv_buffer[f"action_body_{self.robot_name}"])  # 35 dims
        # _ = action_mimic[0:2]
        height_cmd = action_mimic[2:3]
        # _ = action_mimic[3:6]
        body_dof = action_mimic[6:]

        lin_ang_vel = np.array(self.__recv_buffer[f"action_cmd_{self.robot_name}"])[self.command_index]  # 6 dims
        cmd = np.concatenate([lin_ang_vel[:2], lin_ang_vel[-1:], lin_ang_vel[3:6]]) * self.command_scale
        nav_cmd = cmd[:3]
        rpy_cmd = cmd[3:6]

        action_left_hand = np.array(self.__recv_buffer[f"action_hand_left_{self.robot_name}"])
        action_right_hand = np.array(self.__recv_buffer[f"action_hand_right_{self.robot_name}"])

        dof_pos = np.concatenate([body_dof, action_left_hand, action_right_hand], axis=-1)[np.newaxis, :]

        return dof_pos, nav_cmd, height_cmd, rpy_cmd

    def _init_redis(
        self,
        # send_keys: list[str],
        # send_size: list[int],
        recv_keys: list[str],
        host: str = "localhost",
        port: int = 8888,
        db: int = 0,
    ):
        import redis

        # redis settings
        redis_client = redis.Redis(host=host, port=port, db=db)
        self.redis_pipeline = redis_client.pipeline()

        # self.redis_send_keys = [f"{k}_{self.robot_name}" for k in send_keys]
        # self.__send_buffer = {k: json.dumps([0] * send_size[i]) for i, k in enumerate(self.redis_send_keys)}
        # for k in self.redis_send_keys:
        #     self.redis_pipeline.set(k, self.__send_buffer.get(k, None))
        # self.redis_pipeline.execute()

        self.redis_recv_keys = [f"{k}_{self.robot_name}" for k in recv_keys]
        self.__recv_buffer = {k: None for k in self.redis_recv_keys}
