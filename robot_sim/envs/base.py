from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from loguru import logger

from robot_sim.backends import BackendFactory, BaseBackend
from robot_sim.backends.types import ActionsType, ArrayType, ObjectState, StatesType
from robot_sim.configs import SimulatorConfig


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
        """
        super().__init__()

        assert (
            (render_mode == "human" and not config.sim.headless)
            or (render_mode == "rgb_array" and not config.sim.headless)
            or render_mode is None
        ), f"Incompatible render_mode: {render_mode} and headless: {config.sim.headless} setting."

        self._backend = BackendFactory.create_backend(config)
        self.render_mode = render_mode

        # Launch the backend if not already launched
        if not self.backend.is_launched:
            self.backend.launch()
        self._initial_states: StatesType = self.backend.get_states()

        self._observation_space: Any = None  # to be defined in subclass
        self._action_space: Any = None  # to be defined in subclass
        self._mdp_cache: MDPCache = MDPCache()

        # constant
        self._decimation: int = kwargs.get("decimation", 1)
        logger.info(f"Decimation factor set to: {self._decimation}")

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
        if options is not None:
            states = options.get("initial_states", self._initial_states)
        else:
            states = self._initial_states
        self.backend.set_states(states)

        # FIXME: Whether need to resimulate?
        # self.backend.simulate()

        # Get observation from backend state
        states = self.backend.get_states()
        self._mdp_cache.observation = self.statesType2observation(states)

        # Get extra info
        self._mdp_cache.info = self.compute_info(self.observation, None)
        return self.observation, self.info

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
        for _ in range(self.decimation):
            action_array = self.action2actionsType(action)

            # Set action and simulate
            self.backend.set_actions(action_array)
            self.backend.simulate()

        # Get new state and observation
        states = self.backend.get_states()
        ######### assign to mdp cache #########
        self._mdp_cache.action = action
        self._mdp_cache.observation = self.statesType2observation(states)
        # Calculate reward and done flags
        self._mdp_cache.reward = self.compute_reward(self.observation, self.action)
        self._mdp_cache.terminated = self.compute_terminated(self.observation, self.action)
        self._mdp_cache.truncated = self.compute_truncated(self.observation, self.action)
        # Get extra info
        self._mdp_cache.info = self.compute_info(self.observation, self.action)

        return self.observation, self.reward, self.terminated, self.truncated, self.info

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

    def get_states(self, name: str) -> ObjectState:
        """Get the current states from the backend.

        Returns:
            states: The current state dictionary from backend.
        """
        return self.backend.get_states()[name]

    @property
    def observation_space(self) -> Any:
        """It can be any type that constrains the observation."""
        return self._observation_space

    @property
    def action_space(self) -> Any:
        """It can be any type that constrains the action."""
        return self._action_space

    @property
    def backend(self) -> BaseBackend:
        """Get the backend instance."""
        return self._backend

    @property
    def initial_states(self) -> StatesType:
        """Get the initial states for environment reset.

        Returns:
            initial_states: The initial state dictionary for backend.
        """
        return self._initial_states

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
    def info(self) -> dict[str, Any] | None:
        """Get the latest info dictionary.

        Returns:
            info: The latest info dictionary from the MDP cache.
        """
        return self._mdp_cache.info
