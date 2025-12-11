from dataclasses import MISSING, field
from enum import Enum

from robot_sim.configs.base import configclass


class ControlType(Enum):
    """Enumeration of available control types for robot actuators."""

    POSITION = "position"
    TORQUE = "torque"


@configclass
class JointConfig:
    name: str = MISSING
    """Name of the actuator."""
    # type: str = MISSING
    # """Type of the actuator (e.g., 'servo', 'stepper')."""
    torque_limit: float = MISSING
    """Maximum torque of the actuator."""
    velocity_limit: float = MISSING
    """Maximum velocity of the actuator."""
    position_limit: list[float] = MISSING
    """Position limits of the actuator [min, max]."""
    control_type: ControlType | None = MISSING
    """Control type of the actuator."""
    stiffness: float | None = None
    """Stiffness of the actuator."""
    damping: float | None = None
    """Damping of the actuator."""

    def __post_init__(self):
        if ControlType(self.control_type) in [ControlType.POSITION]:
            assert self.stiffness is not None, "Stiffness must be defined for position control."
            assert self.damping is not None, "Damping must be defined for position control."


@configclass
class RobotConfig:
    name: str = MISSING
    """Name of the robot."""
    model_path: str = MISSING
    """Path to the robot's model file."""
    initial_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial position of the robot in the simulation."""
    initial_orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    """Initial orientation of the robot as a quaternion, [x,y,z,w]"""
    joints: list[JointConfig] = field(default_factory=list)
    """List of actuators (joints) in the robot."""
