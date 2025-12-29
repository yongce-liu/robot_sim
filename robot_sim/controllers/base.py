from abc import ABC, abstractmethod

from robot_sim.backends.types import ActionsType, StatesType


class BaseController(ABC):
    """Base class for controllers.

    A *controller* maps an input (observation/state) to an output action.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (integrators, hidden states, caches)."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, name: str, states: StatesType, targets: ActionsType, *args, **kwargs) -> ActionsType:
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

    def __init__(self, controllers: dict[str, BaseController]) -> None:
        self.controllers = controllers

    def reset(self) -> None:
        for c in self.controllers.values():
            c.reset()

    def compute(self, name: str, states: StatesType, targets: ActionsType) -> ActionsType:
        for controller in self.controllers.values():
            targets = controller.compute(name, states, targets)
        return targets
