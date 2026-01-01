from functools import partial
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from loguru import logger

from robot_sim.configs import CameraConfig, SensorType, SimulatorConfig
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
        super().__init__(config=config, **kwargs)

        self._map_cache: MapCache = self._init_spaces_maps(**maps)
        self._controller: CompositeController = self._init_controller()

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

    def _init_controller(self) -> CompositeController:
        ##### Initialize PD controller
        kp = self.robot.stiffness
        kd = self.robot.damping
        tor_limits = self.robot.get_joint_limits("torque", coeff=0.9)
        pd_controller = PIDController(kp=kp, kd=kd, dt=self.step_dt / self.decimation)
        return CompositeController(
            controllers={"pd_controller": pd_controller}, output_clips={"pd_controller": tor_limits}
        )

    def _init_policy(
        self,
        upper_policy: dict[str, Any],
        lower_policy: dict[str, Any],
    ) -> Callable:
        """Initialize the composite controller for the Gr00t robot.

        Returns:
            An instance of CompositeController.
        """

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

            self._observation_space[group_name] = _spaces

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

            self._action_space[group_name] = _spaces

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

    def _init_pd_controller(self) -> PIDController:
        kp = np.array(self.robot.stiffness, dtype=np.float32)
        kd = np.array(self.robot.damping, dtype=np.float32)
        pd_controller = PIDController(kp=kp, kd=kd, dt=self.step_dt / self.decimation)

        return pd_controller

    ####################################################################

    @property
    def robot_name(self) -> str:
        return self.robot_names[0]

    @property
    def robot(self):
        return self.robots[self.robot_name]
