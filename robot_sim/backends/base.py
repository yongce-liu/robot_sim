"""Base backend interface for all simulators."""

from abc import ABC, abstractmethod

any

import numpy as np
from omegaconf import DictConfig


class BaseBackend(ABC):
    """Abstract base class for all simulation backends.

    This provides a unified interface for different simulators (Isaac Lab, MuJoCo, etc.)
    allowing easy switching between backends and supporting multi-simulator scenarios.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the backend.

        Args:
            config: Configuration for the backend
        """
        self.config = config
        self._is_initialized = False
        self._step_count = 0

    @classmethod
    def name(cls) -> str:
        """Get the name of the backend."""
        return cls.__name__

    @abstractmethod
    def setup(self) -> None:
        """Setup the simulation environment.

        This should initialize all necessary components for the simulation,
        including loading models, creating scenes, etc.
        """
        pass

    @abstractmethod
    def step(self) -> dict[str, any]:
        """Step the simulation forward by one timestep.

        Returns:
            dict containing simulation state information (observations, rewards, etc.)
        """
        pass

    @abstractmethod
    def reset(self) -> dict[str, any]:
        """Reset the simulation to initial state.

        Returns:
            dict containing initial state information
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the simulation and cleanup resources."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, any]:
        """Get current simulation state.

        Returns:
            dict containing current state (joint positions, velocities, etc.)
        """
        pass

    @abstractmethod
    def set_state(self, state: dict[str, any]) -> None:
        """Set simulation state.

        Args:
            state: dict containing state to set
        """
        pass

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Apply control action to the robot.

        Args:
            action: Control action (joint positions, velocities, or torques)
        """
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Get observation from the simulation.

        Returns:
            Observation array (proprioceptive and/or exteroceptive data)
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._is_initialized

    @property
    def step_count(self) -> int:
        """Get number of simulation steps executed."""
        return self._step_count

    @property
    def timestep(self) -> float:
        """Get simulation timestep."""
        if self.config is not None:
            return self.config.simulation.timestep
        return 0.001  # Default 1ms

    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return self.__class__.__name__
