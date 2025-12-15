"""Robot controller implementations."""


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


class PDController(BaseController):
    """PD controller implementation."""

    def __init__(self, kp: float = 1.0, kd: float = 0.1) -> None:
        """Initialize PD controller.

        Args:
            kp: Proportional gain
            kd: Derivative gain
        """
        super().__init__()
        self.kp = kp
        self.kd = kd

    def compute(self, state: any, target: any) -> any:
        """Compute PD control output."""
        raise NotImplementedError
