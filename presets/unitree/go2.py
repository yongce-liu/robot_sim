"""Unitree Go2 robot configuration."""

from typing import Any, Dict


class Go2Config:
    """Configuration for Unitree Go2 quadruped robot."""

    def __init__(self) -> None:
        """Initialize Go2 configuration."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get robot configuration dictionary."""
        raise NotImplementedError
