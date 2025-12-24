from dataclasses import MISSING
from functools import partial
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from robot_sim.configs import ControlType, ObjectType, SensorType, SimulatorConfig
from robot_sim.controllers import CompositeController, PIDController
from robot_sim.envs import MapCache, MapEnv
from robot_sim.utils.config import configclass

from .controller import DecoupledWBCPolicy, Gr00tWBCController, LowerBodyPolicy, UpperBodyPolicy
from .utils import act_joint_assign, obs_joint_extract, rpy_cmd_from_waist


@configclass
class Gr00tEnvConfig:
    """Configuration for MapEnv environments."""

    simulator: SimulatorConfig = MISSING
    """Simulator configuration for MapEnv."""
    controller: dict[str, Any] = MISSING
    """Controller configuration for MapEnv."""
    maps: dict[str, Any] = MISSING
    """Maps configuration for observation, action, reward, termination, truncation, and info maps."""


@configclass
class Gr00tTaskConfig:
    """Configuration for Map pick-and-place task."""

    task: str = MISSING
    """Task name for Gr00t environment."""
    params: dict = MISSING
    """Parameters for the specific task."""
    environment: Gr00tEnvConfig = MISSING
    """Environment configuration for the task."""


class Gr00tEnv(MapEnv):
    """Gr00t Whole Body Control Environment.

    This environment uses a composite controller to manage different control strategies for the Gr00t robot.
    The composite controller routes commands to sub-controllers based on the current state and targets.
    Here, we define the environment setup and controller initialization specific to the Gr00t robot.
    Especially the maps for the definition of modality config in Gr00t.

    """

    def __init__(
        self,
        config: Gr00tEnvConfig,
        **kwargs,
    ):
        self.__maps_config = config.maps
        self.__controller_config = config.controller
        _robot_names = [name for name, obj in config.simulator.scene.objects.items() if obj.type == ObjectType.ROBOT]
        assert len(_robot_names) == 1, "Only single robot supported in MapEnv currently."
        self._robot_name = _robot_names[0]

        super().__init__(config=config.simulator, **kwargs)

    def _create_controller_and_maps(self) -> tuple[Gr00tWBCController, MapCache]:
        maps: MapCache = self._init_spaces_maps(**self.__maps_config)
        controller: CompositeController = self._init_controller(**self.__controller_config)
        return controller, maps

    def _init_spaces_maps(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
    ) -> MapCache:
        self._observation_space_dict: dict[str, gym.spaces.Space] = None
        self._action_space_dict: dict[str, gym.spaces.Space] = None
        obs_maps = self._init_observation_spaces_map(**observation)
        act_maps = self._init_action_spaces_map(**action)

        return MapCache(
            observation=obs_maps,
            action=act_maps,
        )

    def _init_controller(
        self,
        upper_policy: dict[str, any] = None,
        lower_policy: dict[str, any] = None,
    ) -> Gr00tWBCController:
        """Initialize the composite controller for the Gr00t robot.

        Returns:
            An instance of Gr00tWBCController.
        """

        upper_body_policy = UpperBodyPolicy(**upper_policy)
        lower_body_policy = self._init_lower_policy(**lower_policy)
        wbc_policy = DecoupledWBCPolicy(
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
            output_indices=np.arange(len(self.get_actuator_names(self.robot_name))),
        )

        pd_controller = self._init_pd_controller()

        controller = Gr00tWBCController(
            wbc_policy=wbc_policy,
            pd_controller=pd_controller,
        )
        return controller

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(self._observation_space_dict)

    @property
    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(self._action_space_dict)

    ############################################################################
    ########## Helper Functions for Maps Initialization and Callbacks ##########
    ############################################################################

    def _init_observation_spaces_map(self, **kwargs) -> dict[str, Callable]:
        """
        Initialize the observation spaces map for the Gr00t environment.
        Returns:
            A dictionary mapping observation group names to their configurations.
        """
        robot_cfg = self.get_object_config(self.robot_name)
        joint_position_limit = np.array([joint.position_limit for joint in robot_cfg.joints.values()], dtype=np.float32)
        self._observation_space_dict = {}
        obs_map = {}
        for group_name, group_cfg in kwargs.items():
            if group_cfg["type"] == "joint":
                epsilon: float = group_cfg.get("epsilon", 1e-3)
                obs_map[group_name] = partial(obs_joint_extract, indices=group_cfg["indices"])
                _spaces = gym.spaces.Box(
                    low=joint_position_limit[:, 0] - epsilon,
                    high=joint_position_limit[:, 1] + epsilon,
                    shape=(len(joint_position_limit),),
                    dtype=np.float32,
                )
            elif group_cfg["type"] == "sensor":
                sensor_name = group_cfg["sensor_name"]
                sensor_data_type = group_cfg.get("data_type", None)
                # map
                if sensor_data_type:
                    obs_map[group_name] = partial(
                        lambda name, states: states[name].sensors[sensor_name][sensor_data_type],
                    )
                else:
                    obs_map[group_name] = partial(
                        lambda name, states: states[name].sensors[sensor_name],
                    )
                # _spaces
                if robot_cfg.sensors[sensor_name].type in [SensorType.CAMERA]:
                    _spaces = gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(
                            robot_cfg.sensors[sensor_name].height,
                            robot_cfg.sensors[sensor_name].width,
                            3,
                        ),  # Assuming fixed camera resolution; adjust as needed
                        dtype=np.uint8,
                    )
                else:
                    raise NotImplementedError
            elif group_cfg["type"] == "constant":
                value = group_cfg["value"]
                obs_map[group_name] = partial(
                    lambda name, states: value,
                )
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
            self._observation_space_dict[group_name] = _spaces

        return obs_map

    def _init_action_spaces_map(self, **kwargs) -> dict[str, Callable]:
        robot_cfg = self.get_object_config(self.robot_name)
        joint_position_limit = np.array([joint.position_limit for joint in robot_cfg.joints.values()], dtype=np.float32)
        self._action_space_dict = {}
        action_map = {}
        for group_name, group_cfg in kwargs.items():
            if group_cfg["type"] == "joint":
                epsilon: float = group_cfg.get("epsilon", 1e-3)
                action_map[group_name] = partial(act_joint_assign, group_name=group_name)
                _spaces = gym.spaces.Box(
                    low=joint_position_limit[:, 0] - epsilon,
                    high=joint_position_limit[:, 1] + epsilon,
                    shape=(len(joint_position_limit),),
                    dtype=np.float32,
                )
            elif group_cfg["type"] == "command":
                if group_name in ["action.rpy_command"]:
                    # Get torso and waist indices
                    torso_index = self.get_body_names(self.robot_name).index("torso_link")
                    pelvis_index = self.get_body_names(self.robot_name).index("pelvis")
                    action_map[group_name] = partial(
                        rpy_cmd_from_waist,
                        torso_index=torso_index,
                        pelvis_index=pelvis_index,
                    )
                elif group_name in ["action.base_height_command", "action.navigate_command"]:
                    action_map[group_name] = lambda **kwargs: None  # No processing needed; direct assignment
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
            self._action_space_dict[group_name] = _spaces

        return action_map

    ####################################################################
    ########## Helper Functions for Controller Initialization ##########
    ####################################################################
    def _init_lower_policy(
        self,
        actuator_indices: list[int],
        hand_indices: list[int],
        observation_params: dict[str, Any],
        **kwargs,
    ) -> LowerBodyPolicy:
        num_dofs = self.num_dofs[self.robot_name]
        used_joint_indices_in_env = np.array([i for i in range(num_dofs) if i not in hand_indices], dtype=np.int32)
        robot_cfg = self.get_object_config(self.robot_name)
        default_joint_position = np.array(
            [joint.default_position for joint in robot_cfg.joints.values() if joint.actuated], dtype=np.float32
        )

        observation_params["used_joint_indices"] = used_joint_indices_in_env
        observation_params["default_joint_position"] = default_joint_position

        lower_body_policy = LowerBodyPolicy(
            actuator_indices=actuator_indices,
            observation_params=observation_params,
            **kwargs,
        )
        return lower_body_policy

    def _init_pd_controller(self) -> PIDController:
        robot_cfg = self.get_object_config(self.robot_name)
        kp = np.array([joint.stiffness for joint in robot_cfg.joints.values() if joint.actuated], dtype=np.float32)
        kd = np.array([joint.damping for joint in robot_cfg.joints.values() if joint.actuated], dtype=np.float32)
        used_pd_indices = [
            i
            for i, joint in enumerate(robot_cfg.joints.values())
            if ControlType(joint.control_type) == ControlType.TORQUE
        ]
        pd_controller = PIDController(kp=kp, kd=kd, dt=self.step_dt / self.decimation, enabled_indices=used_pd_indices)

        return pd_controller

    ####################################################################
