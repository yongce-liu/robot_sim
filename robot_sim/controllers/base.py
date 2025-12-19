class BaseController:
    """Base class for robot controllers."""

    def __init__(self) -> None:
        """Initialize controller."""
        pass

    def compute(self, state: any, target: any) -> any:
        """Compute control output."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset controller state."""
        raise NotImplementedError
