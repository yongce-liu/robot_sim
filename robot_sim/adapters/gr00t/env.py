"""Env wrapper for gr00t experiments.

This module provides a wrapper environment that adapts gr00t_wbc's control
interface to work with robot_sim's backend architecture.
"""

from typing import Any, Callable

import gymnasium as gym
from loguru import logger

from robot_sim.adapters.gr00t.config import Gr00tConfig
from robot_sim.adapters.gr00t.controller import Gr00tController
from robot_sim.backends.types import ActionType, ArrayState
from robot_sim.configs import ObjectType
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
            for obj_name, obj_cfg in config.simulator_config.scene.objects.items()
            if obj_cfg.type == ObjectType.ROBOT
        ]

        assert len(_robot_names) == 1, "Only single robot supported in Gr00tEnv"
        assert config.simulator_config.sim.num_envs == 1, "Only single environment supported in Gr00tEnv"
        super().__init__(config=config.simulator_config, decimation=config.decimation, **kwargs)
        self.config = config
        self.robot_name = _robot_names[0]
        self.robot_cfg = config.simulator_config.scene.objects[self.robot_name]

        self._observation_mapping: dict[str, Callable[[str, "Gr00tEnv"]]] = {}
        self._action_mapping: dict[str, Callable[[str, "Gr00tEnv"]]] = {}
        self._observation_space_dict: dict[str, gym.spaces.Space] = {}
        self._action_space_dict: dict[str, gym.spaces.Space] = {}
        self.num_dofs = len(self.backend.get_actuator_names(self.robot_name))

        logger.info(f"{'=' * 20} Initializing Gr00tEnv {'=' * 20}")

        logger.info("Initializing observation and action mapping...")
        self._init_observation_mapping_space()
        self._init_action_mapping_space()

        logger.info(f"Observation Space: {self.observation_space}\nAction Space: {self.action_space}")

        self.controller: Gr00tController = self._init_controller()

        logger.info(f"{'=' * 20} Gr00tEnv Initialized {'=' * 20}")

    def _init_observation_mapping_space(self) -> None:
        for group_name, (callable_fn, params) in self.config.observation_mapping.items():
            callable_fn(group_name, self, **params)
            self._observation_mapping[group_name] = (callable_fn, params)

    def _init_action_mapping_space(self) -> None:
        for group_name, (callable_fn, params) in self.config.action_mapping.items():
            callable_fn(group_name, self, **params)
            self._action_mapping[group_name] = (callable_fn, params)

    def _init_controller(self) -> Gr00tController:
        # controller = Gr00tController(self)
        # return controller
        pass

    def stateArray2observation(self, states: ArrayState) -> gym.spaces.Dict:
        observation_dict = {}
        for group_name, (callable_fn, params) in self._observation_mapping.items():
            observation_dict[group_name] = callable_fn(
                group_name,
                self,
                **params,
                states=states,
            )
        return observation_dict

    def action2actionArray(self, action: dict[str, Any]) -> ActionType:
        """Convert action to backend format.

        Args:
            action: Action dictionary with key group names and values

        Returns:
            Action dictionary in backend format with key as robot name and value as action array (torque control currently, position control may be added later)
        """
        action_array = self.controller.compute(action)
        return action_array

    def compute_info(self, observation, action=None):
        return {}

    def compute_reward(self, observation, action=None):
        return 0.0

    def compute_terminated(self, observation, action=None):
        return True

    def compute_truncated(self, observation, action=None):
        return True

    @property
    def observation_mapping(self) -> dict[str, Callable]:
        return self._observation_mapping

    @property
    def action_mapping(self) -> dict[str, Callable]:
        return self._action_mapping

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(self._observation_space_dict)

    @property
    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(self._action_space_dict)

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
