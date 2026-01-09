import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from robot_sim.backends import BackendFactory
from robot_sim.configs import ObjectConfig, RobotModel, SimulatorConfig
from robot_sim.configs.types import ActionsType, ArrayType, ObjectState, StatesType


@dataclass
class MDPCache:
    """Cache for MDP related information."""

    observation: Any | None = None
    action: Any | None = None
    reward: float | ArrayType | None = None
    terminated: bool | ArrayType | None = None
    truncated: bool | ArrayType | None = None
    info: dict[str, Any] | None = None


class BaseEnv(ABC):
    """Base environment class for robot simulation environments.

    This class serves as a foundation for all robot simulation environments,
    providing common functionality and structure. It integrates the backend
    simulator and provides the Env interface.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": None}

    def __init__(self, config: SimulatorConfig, render_mode: str | None = None, **kwargs) -> None:
        # Create backend
        self._backend = BackendFactory.create(config=config, **kwargs)
        # Launch the backend if not already launched
        if not self._backend.is_launched:
            self._backend.launch()

        """Initialize the base environment with a backend.

        Args:
            config: Simulator configuration. If provided, a backend will be created.
            render_mode: The mode for rendering the environment.
        """
        assert render_mode in self.metadata["render_modes"] or render_mode is None, (
            f"Invalid render_mode: {render_mode}. Supported modes are: {self.metadata['render_modes']}"
        )
        if render_mode == "human":
            assert config.sim.headless is False, "Headless mode must be False for human rendering."
        self.metadata["render_fps"] = int(1.0 / (config.sim.dt * config.extras.get("decimation", 1)))
        self.render_mode = render_mode

        # constant
        self._decimation: int = config.extras.get("decimation", 1)
        logger.info(f"Decimation factor set to: {self._decimation}")
        self._episode_step: ArrayType = np.zeros((config.sim.num_envs,), dtype=np.int32)
        self._max_episode_steps: int = config.extras.get("max_episode_steps", sys.maxsize)
        assert isinstance(self._max_episode_steps, int) and self._max_episode_steps > 0, (
            "max_episode_steps must be a positive integer."
        )
        logger.info(f"Max episode steps set to: {self._max_episode_steps}")

        # cache
        self._mdp_cache: MDPCache = MDPCache()
        self.__cache: dict[str, Any] = {}

    def reset(self, states: StatesType) -> tuple[Any, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.

        Returns:
            observation: The initial observation.
            info: Additional information dictionary.
        """
        self._episode_step *= 0
        self._backend.reset(states)

        # FIXME: Whether need to resimulate?
        # self._backend.simulate()

        # Get observation from backend state
        states = self._backend.get_states()

        # Get extra info

        # build mdp cache
        self._mdp_cache = MDPCache(
            observation=self.statesType2observation(states),
            info=self.compute_info(self.observation, None),
        )
        return self.observation, self.info

    def step(self, action: Any):
        """Run one timestep of the environment's dynamics.

        Args:
            action: An action provided by the agent.

        Returns:
            observation: The observation after taking the action.
            reward: The reward for this step.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated.
            info: Additional information dictionary.
        """
        # Convert action to backend format
        self._episode_step += 1

        action_dict = dict(action)  # ensure action is a dict, avoid modifying the original action
        _set_action = self.action2actionsType(action_dict)
        for _ in range(self.decimation):
            # Set action and simulate
            self._backend.set_actions(_set_action)
            self._backend.simulate()

        # Get new state and observation
        states = self._backend.get_states()

        ######### assign mdp cache to avoid time-cost, only rebuild cache when reset #########
        self._mdp_cache.action = action
        self._mdp_cache.observation = self.statesType2observation(states)
        self._mdp_cache.reward = self.compute_reward(self.observation, self.action)
        self._mdp_cache.terminated = self.compute_terminated(self.observation, self.action)
        self._mdp_cache.truncated = self.compute_truncated(self.observation, self.action)
        self._mdp_cache.info = self.compute_info(self.observation, self.action)

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def render(self) -> Any | None:
        """Render the environment.

        Returns:
            Rendered output if render_mode requires it, otherwise None.
        """
        if self.render_mode == "rgb_array":
            return self._backend.get_rgb_image()
        elif self.render_mode == "human":
            self._backend._render()
        return None

    def close(self) -> None:
        """Close the environment and cleanup resources."""
        self._backend.close()

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def statesType2observation(self, states: StatesType) -> Any:
        """Convert backend state array to observation.

        Args:
            states: The state dictionary from backend.

        Returns:
            observation: The observation in the format defined by observation_space.
        """
        raise NotImplementedError

    @abstractmethod
    def action2actionsType(self, action: Any) -> ActionsType:
        """Convert action to backend action array format.

        Args:
            action: The action in the format defined by action_space.

        Returns:
            action_array: The action in backend format.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, observation: Any, action: Any | None = None) -> float | ArrayType:
        """Compute the reward for the current step.

        Args:
            observation: The current observation.
            action: The action taken.

        Returns:
            reward: The reward value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_terminated(self, observation: Any, action: Any | None = None) -> bool | ArrayType:
        """Compute whether the episode has terminated.

        Args:
            observation: The current observation.
            action: The action taken.
        Returns:
            terminated: Whether the episode has reached a terminal state.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_truncated(self, observation: Any, action: Any | None = None) -> bool | ArrayType:
        """Compute whether the episode should be truncated.

        Args:
            observation: The current observation.
            action: The action taken.
        Returns:
            truncated: Whether the episode should be truncated (e.g., time limit).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_info(self, observation: Any, action: Any | None = None) -> dict[str, Any]:
        """Compute additional info for the current step.

        Args:
            observation: The current observation.
            action: The action taken.
        Returns:
            info: Additional information dictionary.
        """
        raise NotImplementedError

    ######################### Backend ########################
    def get_object_state(self, name: str) -> ObjectState:
        """Get the current states from the backend.

        Returns:
            states: The current state dictionary from backend.
        """
        return self._backend.get_states()[name]

    def get_states(self) -> StatesType:
        """Get the current states from the backend.

        Returns:
            states: The current state dictionary from backend.
        """
        return self._backend.get_states()

    @property
    def initial_states(self) -> StatesType:
        """Get the initial states for environment reset.

        Returns:
            initial_states: The initial state dictionary for backend.
        """
        return self._backend.initial_states

    def get_object_config(self, name: str) -> ObjectConfig:
        """Get the configuration of the specified robot/object.

        Args:
            name: The name of the robot/object.
        Returns:
            config: The configuration object of the robot/object.
        """
        return self._backend.cfg_objects[name]

    @property
    def step_dt(self) -> float:
        """Get the time duration of a single environment step.

        Returns:
            step_dt: The time duration of a single environment step.
        """
        return self._backend.cfg_sim.dt * self.decimation

    @property
    def num_envs(self) -> int:
        """Get the number of parallel environments."""
        return self._backend.num_envs

    @property
    def robot_names(self) -> list[str]:
        """Get the robot configurations in the environment.

        Returns:
            robots: A dictionary of robot configurations keyed by robot name.
        """
        return self._backend.robot_names

    @property
    def robots(self) -> dict[str, RobotModel]:
        """Get the RobotModel instance for the Gr00t robot.

        Returns:
            An instance of RobotModel representing the Gr00t robot.
        """
        return self._backend.robots

    ######################### Env ########################

    @property
    def episode_step(self) -> ArrayType:
        return self._episode_step

    @episode_step.setter
    def episode_step(self, value: ArrayType) -> None:
        self._episode_step = value

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def decimation(self) -> int:
        """Get the decimation factor for simulation steps.

        Returns:
            decimation: The number of simulation steps per environment step.
        """
        return self._decimation

    ######################## MDP Cache Properties ########################
    @property
    def mdp_cache(self) -> MDPCache:
        """Get the MDP cache instance.

        Returns:
            mdp_cache: The MDP cache storing observation, action, reward, done flags, and info.
        """
        return self._mdp_cache

    @property
    def observation(self) -> Any:
        """Get the latest observation.

        Returns:
            observation: The latest observation from the MDP cache.
        """
        return self._mdp_cache.observation

    @property
    def action(self) -> Any:
        """Get the latest action.

        Returns:
            action: The latest action from the MDP cache.
        """
        return self._mdp_cache.action

    @property
    def reward(self) -> float | ArrayType | None:
        """Get the latest reward.

        Returns:
            reward: The latest reward from the MDP cache.
        """
        return self._mdp_cache.reward

    @property
    def terminated(self) -> bool | ArrayType | None:
        """Get the latest terminated flag.

        Returns:
            terminated: The latest terminated flag from the MDP cache.
        """
        return self._mdp_cache.terminated

    @property
    def truncated(self) -> bool | ArrayType | None:
        """Get the latest truncated flag.

        Returns:
            truncated: The latest truncated flag from the MDP cache.
        """
        return self._mdp_cache.truncated

    @property
    def info(self) -> dict[str, Any]:
        """Get the latest info dictionary.

        Returns:
            info: The latest info dictionary from the MDP cache.
        """
        if self._mdp_cache.info is None:
            return {}
        return self._mdp_cache.info
