"""Robot controller implementations."""

import numpy as np
import torch

from robot_sim.backends.types import ArrayType, ObjectState
from robot_sim.controllers import BaseController


class PIDController(BaseController):
    """PID controller implementation."""

    def __init__(
        self,
        kp: ArrayType,
        ki: ArrayType | None = None,
        kd: ArrayType | None = None,
        dt: float = 0.01,
        enabled_indices: list[int] | ArrayType | None = None,
    ) -> None:
        """Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            dt: Time step for integration
        """
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_error = None  # Will be initialized on first compute
        self.disabled_indices = (
            np.array([i for i in range(len(kp)) if i not in enabled_indices], dtype=np.int32)
            if enabled_indices is not None
            else None
        )

    def compute(self, state: ObjectState, target: ArrayType) -> ArrayType:
        """Compute control action for the given robot.

        Args:
            name: Robot name
            states: Current states from the backend
            targets: Target positions (and optionally velocities)
        Returns:
            Control output (torque or force)
        """
        return self.apply_pid(target=target, position=state.joint_pos, velocity=state.joint_vel)

    def apply_pid(
        self,
        target: ArrayType,
        position: ArrayType,
        velocity: ArrayType | None = None,
    ) -> ArrayType:
        """Compute PID control output.

        Args:
            target: Target position
            position: Current position
            velocity: Current velocity (required for D term)

        Returns:
            Control output (torque or force)
        """
        # Compute position error
        error = target - position

        if self.ki is None and self.kd is not None:
            # PD controller
            # becasue for motor control, we often do not have integral term
            if velocity is None:
                raise ValueError("velocity is required for PD controller")
            control_output = self.kp * error - self.kd * velocity

        elif self.ki is None and self.kd is None:
            # P controller
            control_output = self.kp * error

        elif self.ki is not None and self.kd is None:
            # PI controller
            # Initialize integral error if needed
            if self.integral_error is None:
                if isinstance(error, torch.Tensor):
                    self.integral_error = torch.zeros_like(error)
                else:
                    self.integral_error = np.zeros_like(error)

            # Update integral error
            self.integral_error += error * self.dt
            control_output = self.kp * error + self.ki * self.integral_error

        else:
            # PID controller
            if velocity is None:
                raise ValueError("velocity is required for PID controller")

            # Initialize integral error if needed
            if self.integral_error is None:
                if isinstance(error, torch.Tensor):
                    self.integral_error = torch.zeros_like(error)
                else:
                    self.integral_error = np.zeros_like(error)

            # Update integral error
            self.integral_error += error * self.dt
            control_output = self.kp * error + self.ki * self.integral_error - self.kd * velocity

        if self.disabled_indices is not None:
            control_output[..., self.disabled_indices] = target[..., self.disabled_indices]

        return control_output

    def reset(self) -> None:
        """Reset integral error."""
        self.integral_error = None

    def to_numpy(self, dtype: np.dtype = np.float32):
        self.kp = np.asarray(self.kp, dtype=dtype)
        if self.ki is not None:
            self.ki = np.asarray(self.ki, dtype=dtype)
        if self.kd is not None:
            self.kd = np.asarray(self.kd, dtype=dtype)

    def to_tensor(self, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        self.kp = torch.tensor(self.kp, dtype=dtype, device=device)
        if self.ki is not None:
            self.ki = torch.tensor(self.ki, dtype=dtype, device=device)
        if self.kd is not None:
            self.kd = torch.tensor(self.kd, dtype=dtype, device=device)
