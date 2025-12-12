from dataclasses import field
from enum import Enum
from typing import Any

from .base import configclass


class ObjectType(Enum):
    """Enumeration of available object types in the simulation."""

    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CUSTOM = "custom"


@configclass
class ObjectConfig:
    path: str | None = None
    """Path to the object's model file. If None, we can create it using the provided api from the simulator."""
    initial_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial position of the object in the simulation."""
    initial_orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    """Initial orientation of the object as a quaternion, [w,x,y,z]"""
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional properties specific to the object."""
    type: ObjectType = ObjectType.CUSTOM
    """Type of the object. Can be a ObjectType or 'custom'."""
