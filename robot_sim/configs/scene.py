from dataclasses import field

from robot_sim.configs import ObjectConfig, RobotConfig, configclass


@configclass
class SceneConfig:
    objects: dict[str, ObjectConfig] = field(default_factory=dict)
    """List of objects to include in the simulation."""
    robots: dict[str, RobotConfig] = field(default_factory=dict)
    """List of robots to include in the simulation."""

    def __post_init__(self) -> None:
        pass
