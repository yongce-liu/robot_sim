from typing import Callable

from robot_sim.configs import BackendType, SimulatorConfig

from .base import BaseBackend
from .mujoco import MujocoBackend
from .unitree import UnitreeFactory

# Registry to map sensor type to concrete class
_BACKEND_TYPE_REGISTRY: dict[BackendType, Callable[..., BaseBackend]] = {
    BackendType.MUJOCO: MujocoBackend,
    BackendType.UNITREE: UnitreeFactory.create,
}


class BackendFactory:
    _backend_instance: BaseBackend | None = None
    """Optional pre-created backend instance. If provided, this will be used instead of creating a new one."""

    def __init__(self, **kwargs) -> None:
        self._backend_instance = self.create(**kwargs)

    @staticmethod
    def create(config: SimulatorConfig, **kwargs) -> BaseBackend:
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
        return backend_class(config=config, **kwargs)
