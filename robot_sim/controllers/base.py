from abc import ABC, abstractmethod

from robot_sim.configs.types import ArrayType, ObjectState


class BaseController(ABC):
    """Base class for controllers.

    A *controller* maps an input (observation/state) to an output action.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (integrators, hidden states, caches)."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, state: ObjectState, target: ArrayType, *args, **kwargs) -> ArrayType:
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

    def __init__(
        self,
        controllers: dict[str, BaseController],
        output_clips: dict[str, tuple[ArrayType | float, ArrayType | float]] | None = None,
    ) -> None:
        self.controllers = controllers
        self.output_clips = output_clips

    def reset(self) -> None:
        for c in self.controllers.values():
            c.reset()

    def compute(self, state: ObjectState, target: ArrayType) -> ArrayType:
        for ctrl_name, controller in self.controllers.items():
            output = controller.compute(state, target)
            if self.output_clips is not None and ctrl_name in self.output_clips:
                clip_min, clip_max = self.output_clips[ctrl_name]
                output = output.clip(clip_min, clip_max)

        return output
