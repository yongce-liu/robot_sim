"""Configuration system."""

from .base import configclass
from .object import BuiltinObjectType, ObjectConfig
from .robot import RobotConfig
from .scene import SceneConfig
from .simulator import PhysicsConfig, SimulatorConfig

__all__ = [
    "configclass",
    "ObjectConfig",
    "BuiltinObjectType",
    "SceneConfig",
    "RobotConfig",
    "PhysicsConfig",
    "SimulatorConfig",
]
