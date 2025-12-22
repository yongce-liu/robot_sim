from abc import ABC, abstractmethod
from typing import Any

from robot_sim.backends.types import ActionType


class BaseController(ABC):
    """Base class for controllers.

    A *controller* maps an input (observation/state) to an output action.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (integrators, hidden states, caches)."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """Compute control output.

        Signature is intentionally flexible; concrete controllers should
        document their expected inputs.
        """

        raise NotImplementedError


class CompositeController:
    """A controller that routes/combines multiple low-level controllers.

    Typical usage:
      - high-level controller produces targets (e.g., joint positions, base cmd)
      - low-level controller (e.g., PID) produces actuator command
      - optional additional controller (e.g., WBC) produces whole-body command

    This class is backend-agnostic; adapters can wrap env-specific I/O.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.controllers: dict[str, BaseController] | None = None

    def reset(self) -> None:
        for c in self.controllers.values():
            c.reset()

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> ActionType:
        """Override in subclasses to implement routing logic."""

        raise NotImplementedError
