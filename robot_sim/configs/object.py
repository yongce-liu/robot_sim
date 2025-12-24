from dataclasses import MISSING, dataclass, field
from enum import Enum
from typing import Any

from .sensor import SensorConfig


class ObjectType(Enum):
    """Enumeration of available object types in the simulation."""

    ROBOT = "robot"
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CUSTOM = "custom"


class ControlType(Enum):
    """Enumeration of available control types for robot actuators."""

    POSITION = "position"
    TORQUE = "torque"


@dataclass
class JointConfig:
    # name: str = MISSING
    # """Name of the actuator."""
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
    default_position: float = 0.0
    """Default position of the actuator."""
    stiffness: float | None = None
    """Stiffness of the actuator."""
    damping: float | None = None
    """Damping of the actuator."""
    actuated: bool = True
    """Whether the actuator is can be actuated."""
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional properties specific to the joint."""

    def __post_init__(self):
        if ControlType(self.control_type) in [ControlType.POSITION]:
            assert self.stiffness is not None, "Stiffness must be defined for position control."
            assert self.damping is not None, "Damping must be defined for position control."


@dataclass
class ObjectConfig:
    # name: str = MISSING
    # """Name of the robot."""
    type: ObjectType = ObjectType.CUSTOM
    """Type of the object. Can be a ObjectType or 'custom'."""
    path: str | None = None
    """Path to the robot's model file."""
    initial_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial position of the robot in the simulation."""
    initial_orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    """Initial orientation of the robot as a quaternion, [w,x,y,z]"""
    joints: dict[str, JointConfig] | None = None
    """List of actuators (joints) in the robot/object. If it is None, no actuators are defined."""
    bodies: dict[str, Any] | None = None
    """List of bodies in the robot/object. If it is None, no bodies are defined."""
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    """Sensor configurations for the robot."""
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional properties specific to the robot."""

    def __post_init__(self):
        if self.path is None and self.type in [ObjectType.CUSTOM, ObjectType.ROBOT]:
            raise ValueError("For custom object type, a valid path to the model file must be provided.")
