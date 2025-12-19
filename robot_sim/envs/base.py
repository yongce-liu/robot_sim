from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from loguru import logger

from robot_sim.backends import BackendFactory, BaseBackend
from robot_sim.backends.types import ActionType, ArrayState
from robot_sim.configs import SimulatorConfig


class BaseEnv(ABC, gym.Env):
    """Base environment class for robot simulation environments.

    This class serves as a foundation for all robot simulation environments,
    providing common functionality and structure. It integrates the backend
    simulator and provides the gym.Env interface.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the base environment with a backend.

        Args:
            config: Simulator configuration. If provided, a backend will be created.
            render_mode: The mode for rendering the environment.
        Note:
            You should define the properties
            observation_space: gym.Space
            action_space: gym.Space
        """
        super().__init__()

        assert (render_mode == "human" and not config.sim.headless) or (
            render_mode == "rgb_array" or render_mode is None and not config.sim.headless
        ), f"Incompatible render_mode: {render_mode} and headless: {config.sim.headless} setting."
        self._backend = BackendFactory.createbackend(config)
        self.render_mode = render_mode

        # Launch the backend if not already launched
        if not self.backend.is_launched:
            self.backend.launch()
        self._initial_states: ArrayState = self.backend.get_states()

        self._observation_space: gym.Space = None  # to be defined in subclass
        self._action_space: gym.Space = None  # to be defined in subclass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.

        Returns:
            observation: The initial observation.
            info: Additional information dictionary.
        """
        super().reset(seed=seed)

        # Reset the backend state
        states = options.get("initial_states", self._initial_states)
        self.backend.set_states(states)

        # FIXME: Whether need to resimulate?
        # self.backend.simulate()

        # Get observation from backend state
        states = self.backend.get_states()
        observation = self.stateArray2observation(states)

        # Get extra info
        info = self.compute_info(observation, None)

        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
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
        action_array = self.action2actionArray(action)

        # Set action and simulate
        self.backend.set_actions(action_array)
        self.backend.simulate()

        # Get new state and observation
        states = self.backend.get_states()
        observation = self.stateArray2observation(states)

        # Calculate reward and done flags
        reward = self.compute_reward(observation, action)
        terminated = self.compute_terminated(observation, action)
        truncated = self.compute_truncated(observation, action)

        # Get extra info
        info = self.compute_info(observation, action)

        return observation, reward, terminated, truncated, info

    def render(self) -> Any | None:
        """Render the environment.

        Returns:
            Rendered output if render_mode requires it, otherwise None.
        """
        if self.render_mode == "rgb_array":
            if hasattr(self.backend, "get_world_image"):
                return self.backend.get_world_image()
            else:
                logger.warning("Backend does not support image rendering.")
        elif self.render_mode == "human":
            self.backend.render()
        return None

    def close(self) -> None:
        """Close the environment and cleanup resources."""
        self.backend.close()

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def stateArray2observation(self, states: ArrayState) -> Any:
        """Convert backend state array to observation.

        Args:
            states: The state dictionary from backend.

        Returns:
            observation: The observation in the format defined by observation_space.
        """
        raise NotImplementedError

    @abstractmethod
    def action2actionArray(self, action: Any) -> ActionType:
        """Convert action to backend action array format.

        Args:
            action: The action in the format defined by action_space.

        Returns:
            action_array: The action in backend format.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, observation: Any, action: Any | None = None) -> float | np.ndarray:
        """Compute the reward for the current step.

        Args:
            observation: The current observation.
            action: The action taken.

        Returns:
            reward: The reward value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_terminated(self, observation: Any, action: Any | None = None) -> bool | np.ndarray:
        """Compute whether the episode has terminated.

        Args:
            observation: The current observation.
            action: The action taken.
        Returns:
            terminated: Whether the episode has reached a terminal state.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_truncated(self, observation: Any, action: Any | None = None) -> bool | np.ndarray:
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

    @property
    def observation_space(self) -> gym.Space:
        """Required by gymnasium."""
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Required by gymnasium."""
        return self._action_space

    @property
    def backend(self) -> BaseBackend:
        """Get the backend instance."""
        return self._backend

    @property
    def initial_states(self) -> ArrayState:
        """Get the initial states for environment reset.

        Returns:
            initial_states: The initial state dictionary for backend.
        """
        return self._initial_states
