from dataclasses import field

from robot_sim.configs.base import configclass
from robot_sim.configs.object import ObjectConfig
from robot_sim.configs.robot import RobotConfig


@configclass
class SceneConfig:
    objects: list[ObjectConfig] = field(default_factory=list)
    """List of objects to include in the simulation."""
    robots: list[RobotConfig] = field(default_factory=list)
    """List of robots to include in the simulation."""


# Backwards compatibility for older name
SceneConfig = SceneConfig
