"""Unitree Go2 robot configuration."""

any, dict


class Go2Config:
    """Configuration for Unitree Go2 quadruped robot."""

    def __init__(self) -> None:
        """Initialize Go2 configuration."""
        pass

    def get_config(self) -> dict[str, any]:
        """Get robot configuration dictionary."""
        raise NotImplementedError
