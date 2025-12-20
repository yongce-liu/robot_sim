"""Env wrapper for gr00t experiments.

This module provides a wrapper environment that adapts gr00t_wbc's control
interface to work with robot_sim's backend architecture.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import regex as re

from robot_sim.adapters.gr00t.config import Gr00tConfig
from robot_sim.backends.types import ActionType, ArrayState
from robot_sim.configs import ObjectType, SensorType
from robot_sim.controllers import PIDController
from robot_sim.envs.base import BaseEnv


class Gr00tEnv(BaseEnv):
    """Environment wrapper for gr00t experiments.

    This environment wraps the backend simulator and provides an interface
    compatible with gr00t_wbc's whole body control framework.

    Args:
        config: Speicialized configuration for Gr00tEnv
        **kwargs: Additional configuration options, e.g, render_mode
    """

    def __init__(
        self,
        config: Gr00tConfig | None = None,
        **kwargs,
    ) -> None:
        _robot_names = [
            obj_name
            for obj_name, obj_cfg in config.simulator_config.scene.objects.item
            if obj_cfg.type == ObjectType.ROBOT
        ]

        assert len(_robot_names) == 1, "Only single robot supported in Gr00tEnv"
        assert config.simulator_config.sim.num_envs == 1, "Only single environment supported in Gr00tEnv"
        super().__init__(config=config.simulator_config, **kwargs)
        self.config = config
        self.robot_name = _robot_names[0]
        self.robot_cfg = config.simulator_config.scene.objects[self.robot_name]

        self._observation_mapping = self._init_observation_mapping()
        self._action_mapping = self._init_action_mapping()

        self._observation_space = self._init_observation_space()
        self._action_space = self._init_action_space()

        self.controller = self._init_controller()

    def _init_observation_mapping(self) -> dict[str, list[int] | str]:
        # initialize observation group mapping
        joint_names = [joint.split("/")[-1].split(".")[-1] for joint in self.backend.get_joint_names(self.robot_name)]
        observation_mapping = {}
        for group_name, name_patterns in self.config.observation_mapping.items():
            if group_name.startswith("state."):
                if isinstance(name_patterns, str):
                    name_patterns = [name_patterns]
                observation_mapping[group_name] = []
                for pattern in name_patterns:
                    rx = re.compile(pattern)
                    matched_joint_indices = [joint_names.index(name) for name in joint_names if rx.fullmatch(name)]
                    observation_mapping[group_name].extend(matched_joint_indices)
            elif group_name.startswith("video."):
                if isinstance(name_patterns, str):
                    name_patterns = [name_patterns]
                observation_mapping[group_name] = []
                for pattern in name_patterns:
                    rx = re.compile(pattern)
                    available_sensors = [
                        name for name, cfg in self.robot_cfg.sensors.items() if cfg.type == SensorType.CAMERA
                    ]
                    matched_camera_names = [name for name in available_sensors if rx.fullmatch(name)]
                    observation_mapping[group_name].extend(matched_camera_names)
            elif group_name.startswith("annotation."):
                assert isinstance(name_patterns, str), "Annotation pattern must be a string"
                observation_mapping[group_name] = name_patterns
            else:
                raise ValueError(f"Unsupported observation group name: {group_name}")
        return observation_mapping

    def _init_action_mapping(self) -> dict[str, list[int]]:
        pass

    def _init_observation_space(self) -> gym.spaces.Dict:
        observation_space_dict = {}
        joint_names = [joint.split("/")[-1].split(".")[-1] for joint in self.backend.get_joint_names(self.robot_name)]
        for group_name, group_val in self._observation_mapping.items():
            if group_name.startswith("state."):
                low_val = [self.robot_cfg.joints[name].position_limit[0] for name in joint_names]
                high_val = [self.robot_cfg.joints[name].position_limit[1] for name in joint_names]
                observation_space_dict[group_name] = gym.spaces.Box(
                    low=np.array(low_val, dtype=np.float32),
                    high=np.array(high_val, dtype=np.float32),
                    shape=(len(group_val),),
                    dtype=np.float32,
                )
            elif group_name.startswith("video."):
                observation_space_dict[group_name] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.robot_cfg.sensors[group_val].get("height"),
                        self.robot_cfg.sensors[group_val].get("width"),
                        3,
                    ),  # Assuming fixed camera resolution; adjust as needed
                    dtype=np.uint8,
                )
            elif group_name.startswith("annotation."):
                observation_space_dict[group_name] = gym.spaces.Text(
                    max_length=1024, charset=self.config.allowed_language_charset
                )
            else:
                raise ValueError(f"Unsupported observation group name: {group_name}")
        return gym.spaces.Dict(observation_space_dict)

    def _init_action_space(self) -> gym.spaces.Dict:
        pass

    def _init_controller(self) -> PIDController:
        return PIDController(
            kp=np.array([100.0] * self.num_dofs, dtype=np.float32),
            ki=None,
            kd=np.array([1.0] * self.num_dofs, dtype=np.float32),
            dt=1.0 / self.config.simulator_config.sim.control_frequency,
        )

    def stateArray2observation(self, states: ArrayState) -> gym.spaces.Dict:
        observation_dict = {}
        robot_state = states.objects[self.robot_name]
        for group_name, group_val in self._observation_mapping.items():
            if group_name.startswith("state."):
                observation_dict[group_name] = robot_state.joint_pos[..., group_val]
            elif group_name.startswith("video."):
                observation_dict[group_name] = robot_state.sensors[group_val]["rgb"]
            elif group_name.startswith("annotation."):
                # For simplicity, return a placeholder annotation
                observation_dict[group_name] = group_val
            else:
                raise ValueError(f"Unsupported observation group name: {group_name}")

    def action2actionArray(self, action: dict[str, Any]) -> ActionType:
        """Convert action to backend format.

        Args:
            action: Action dictionary with 'q' (target joint positions) and optionally 'tau'

        Returns:
            Action dictionary in backend format
        """
        action_array = {}

        # Convert joint position targets
        if "q" in action:
            q = np.asarray(action["q"], dtype=np.float32)
            action_array["q"] = q

        # Add torque if provided (for gravity compensation or additional control)
        if "tau" in action:
            tau = np.asarray(action["tau"], dtype=np.float32)
            action_array["tau"] = tau
        else:
            action_array["tau"] = np.zeros(self.num_dofs, dtype=np.float32)

        return action_array

    #     self.robot_name = robot_name
    #     self.enable_gravity_compensation = enable_gravity_compensation
    #     self.gravity_compensation_joints = gravity_compensation_joints or ["arms"]
    #     self.kwargs = kwargs

    #     # Get robot model from backend buffer
    #     self._init_robot_info()

    #     # Initialize cache for observations
    #     self.cache: Dict[str, Any] = {
    #         "obs": None,
    #         "reward": 0.0,
    #         "terminated": False,
    #         "truncated": False,
    #         "info": {},
    #     }

    #     # Episode step counter for truncation
    #     self._episode_step = 0
    #     self._max_episode_steps = kwargs.get("max_episode_steps", 1000)

    #     # Define observation and action spaces
    #     self._setup_spaces()

    # def _init_robot_info(self) -> None:
    #     """Initialize robot information from backend."""
    #     # Get robot configuration from backend
    #     robot_configs = [obj for obj in self.backend.objects.values() if obj.type == "robot"]
    #     if not robot_configs:
    #         raise ValueError("No robot found in backend configuration")

    #     self.robot_config = robot_configs[0]  # Use first robot
    #     self.robot_obj_name = list(self.backend.objects.keys())[0]

    #     # Get joint information from backend buffer
    #     self.joint_names = self.backend.get_joint_names(self.robot_obj_name)
    #     self.num_dofs = len(self.joint_names)

    #     # Get body information
    #     self.body_names = self.backend.get_body_names(self.robot_obj_name)

    # def _setup_spaces(self) -> None:
    #     """Setup observation and action gym.spaces."""
    #     # Action space: joint positions
    #     self.action_space = gym.spaces.Dict(
    #         {
    #             "q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs,), dtype=np.float32),
    #             "tau": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs,), dtype=np.float32),
    #         }
    #     )

    #     # Observation space: robot state
    #     self.observation_space = gym.spaces.Dict(
    #         {
    #             "q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs,), dtype=np.float32),
    #             "dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs,), dtype=np.float32),
    #             "ddq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs,), dtype=np.float32),
    #             "tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs,), dtype=np.float32),
    #             "floating_base_pose": gym.spaces.Box(
    #                 low=-np.inf,
    #                 high=np.inf,
    #                 shape=(7,),  # [x, y, z, qw, qx, qy, qz]
    #                 dtype=np.float32,
    #             ),
    #             "floating_base_vel": gym.spaces.Box(
    #                 low=-np.inf,
    #                 high=np.inf,
    #                 shape=(6,),  # [vx, vy, vz, wx, wy, wz]
    #                 dtype=np.float32,
    #             ),
    #             "floating_base_acc": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
    #         }
    #     )

    # def get_initial_states(self) -> ArrayState:
    #     """Get initial states for environment reset.

    #     Returns:
    #         Initial state dictionary for backend
    #     """
    #     # Create default initial state
    #     # This should match your backend's state structure
    #     initial_state = {
    #         "q": np.zeros(self.num_dofs, dtype=np.float32),
    #         "dq": np.zeros(self.num_dofs, dtype=np.float32),
    #         "root_pos": np.array([0.0, 0.0, 1.0], dtype=np.float32),  # Default standing height
    #         "root_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity quaternion
    #         "root_vel": np.zeros(6, dtype=np.float32),
    #     }

    #     # You can customize initial joint positions here
    #     # For example, set to a default standing pose
    #     if hasattr(self, "default_joint_positions"):
    #         initial_state["q"] = self.default_joint_positions.copy()

    #     return initial_state

    # def compute_reward(self, states: ArrayState, action: Any) -> float:
    #     """Compute reward for the current step.

    #     Args:
    #         states: Current backend states
    #         action: Action taken

    #     Returns:
    #         Reward value (default: 0.0, to be customized for specific tasks)
    #     """
    #     # Default reward is 0 - override in subclass for specific tasks
    #     reward = 0.0

    #     # Example: reward for maintaining upright posture
    #     if hasattr(states, "root_quat"):
    #         root_quat = self._to_numpy(states.root_quat)
    #         # Check if robot is upright (qw should be close to 1)
    #         upright_reward = root_quat[0]  # qw component
    #         reward += upright_reward

    #     return reward

    # def compute_terminated(self, states: ArrayState) -> bool:
    #     """Compute whether episode has terminated.

    #     Args:
    #         states: Current backend states

    #     Returns:
    #         True if episode should terminate (e.g., robot fell)
    #     """
    #     # Default: check if robot fell
    #     if hasattr(states, "root_pos"):
    #         root_pos = self._to_numpy(states.root_pos)
    #         # Terminate if robot falls below threshold
    #         if root_pos[2] < 0.3:  # Height threshold
    #             return True

    #     return False

    # def compute_truncated(self, states: ArrayState) -> bool:
    #     """Compute whether episode should be truncated.

    #     Args:
    #         states: Current backend states

    #     Returns:
    #         True if episode should be truncated (e.g., time limit)
    #     """
    #     # Truncate if max episode steps reached
    #     return self._episode_step >= self._max_episode_steps

    # def reset(
    #     self,
    #     *,
    #     seed: int | None = None,
    #     options: dict[str, Any] | None = None,
    # ) -> tuple[Dict[str, np.ndarray], dict[str, Any]]:
    #     """Reset the environment.

    #     Args:
    #         seed: Random seed
    #         options: Additional reset options

    #     Returns:
    #         Observation and info dictionary
    #     """
    #     # Reset episode counter
    #     self._episode_step = 0

    #     # Call parent reset
    #     observation, info = super().reset(seed=seed, options=options)

    #     # Update cache
    #     self.cache["obs"] = observation
    #     self.cache["reward"] = 0.0
    #     self.cache["terminated"] = False
    #     self.cache["truncated"] = False
    #     self.cache["info"] = info

    #     return observation, info

    # def step(self, action: Dict[str, Any]) -> tuple[Dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
    #     """Execute one step in the environment.

    #     Args:
    #         action: Action dictionary with 'q' and optionally 'tau'

    #     Returns:
    #         Observation, reward, terminated, truncated, and info
    #     """
    #     # Increment step counter
    #     self._episode_step += 1

    #     # Execute step
    #     observation, reward, terminated, truncated, info = super().step(action)

    #     # Update cache
    #     self.cache["obs"] = observation
    #     self.cache["reward"] = reward
    #     self.cache["terminated"] = terminated
    #     self.cache["truncated"] = truncated
    #     self.cache["info"] = info

    #     return observation, reward, terminated, truncated, info

    # def observe(self) -> Dict[str, np.ndarray]:
    #     """Get current observation from cache.

    #     Returns:
    #         Current observation dictionary
    #     """
    #     if self.cache["obs"] is None:
    #         raise RuntimeError("Environment not initialized. Call reset() first.")
    #     return self.cache["obs"]

    # @staticmethod
    # def _to_numpy(value: Any) -> np.ndarray:
    #     """Convert value to numpy array."""
    #     if isinstance(value, np.ndarray):
    #         return value
    #     elif hasattr(value, "cpu"):  # PyTorch tensor
    #         return value.cpu().numpy()
    #     else:
    #         return np.asarray(value)
