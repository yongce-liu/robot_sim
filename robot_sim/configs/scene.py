from dataclasses import field

from robot_sim.configs import ObjectConfig, RobotConfig, configclass


@configclass
class SceneConfig:
    objects: list[ObjectConfig] = field(default_factory=list)
    """List of objects to include in the simulation."""
    robots: list[RobotConfig] = field(default_factory=list)
    """List of robots to include in the simulation."""


# Backwards compatibility for older name
SceneConfig = SceneConfig
