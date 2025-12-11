"""Factory function to create backends from configuration."""

from omegaconf import DictConfig

from robot_sim.backend.base import BaseBackend
from robot_sim.backend.isaac import IsaacBackend
from robot_sim.backend.mujoco import MuJoCoBackend


def create_backend(backend_type: str, config: DictConfig | None = None) -> BaseBackend:
    """Factory function to create a backend instance.

    Args:
        backend_type: Type of backend ("isaac", "mujoco", etc.)
        config: Optional configuration for the backend

    Returns:
        Backend instance

    Raises:
        ValueError: If backend_type is not recognized
    """
    backend_type = backend_type.lower()

    backend_classes = {
        "isaac": IsaacBackend,
        "mujoco": MuJoCoBackend,
        # Add more backends here as needed
    }

    if backend_type not in backend_classes:
        available = ", ".join(backend_classes.keys())
        raise ValueError(f"Unknown backend type: {backend_type}. Available backends: {available}")

    return backend_classes[backend_type](config)
