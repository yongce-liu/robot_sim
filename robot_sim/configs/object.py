from dataclasses import MISSING, field
from enum import Enum
from typing import Any

from robot_sim.configs import configclass


class BuiltinObjectType(Enum):
    """Enumeration of available object types in the simulation."""

    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"


@configclass
class ObjectConfig:
    name: str = MISSING
    """Name of the object."""
    model_path: str | None = None
    """Path to the object's model file. If None, we can create it using the provided api from the simulator."""
    initial_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial position of the object in the simulation."""
    initial_orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    """Initial orientation of the object as a quaternion, [x,y,z,w]"""
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional properties specific to the object."""

    def __post_init__(self) -> None:
        if not isinstance(self.initial_position, list) or len(self.initial_position) != 3:
            raise ValueError("initial_position must be a list of three floats.")
        if not isinstance(self.initial_orientation, list) or len(self.initial_orientation) != 4:
            raise ValueError("initial_orientation must be a list of four floats representing a quaternion.")
        if self.model_path is None:
            assert BuiltinObjectType(self.name)