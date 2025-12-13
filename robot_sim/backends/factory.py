from robot_sim.configs import BackendType, SimulatorConfig

from .base import BaseBackend
from .mujoco import MujocoBackend

# Registry to map sensor type to concrete class
_BACKEND_TYPE_REGISTRY: dict[BackendType, type] = {
    BackendType.MUJOCO: MujocoBackend,
}


class BackendFactory:
    _backend_instance: BaseBackend | None = None
    """Optional pre-created backend instance. If provided, this will be used instead of creating a new one."""

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self._backend_instance = self.create_backend(config)

    def reset(self) -> None:
        """Reset the backend instance."""
        self._backend_instance = None

    @staticmethod
    def create_backend(config: SimulatorConfig) -> BaseBackend:
        """Create a simulation backend based on the specified type.

        Args:
            config: Simulator configuration.
        Returns:
            An instance of the specified backend.
        """
        backend_type = config.backend
        if backend_type not in _BACKEND_TYPE_REGISTRY:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        backend_class = _BACKEND_TYPE_REGISTRY[backend_type]
        return backend_class(config=config)

    @property
    def backend(self) -> BaseBackend:
        """Get the simulation backend instance, creating it if necessary.

        Returns:
            The simulation backend instance.
        """
        if self._backend_instance is None:
            self._backend_instance = self.create_backend(self.config)
        return self._backend_instance
