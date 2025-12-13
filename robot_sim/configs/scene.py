from dataclasses import MISSING, field

from robot_sim.utils import configclass

from .object import ObjectConfig
from .robot import RobotConfig
from .terrain import TerrainConfig


@configclass
class SceneConfig:
    terrain: TerrainConfig = MISSING
    """Terrain configuration."""
    objects: dict[str, ObjectConfig] = field(default_factory=dict)
    """List of objects to include in the simulation."""
    robots: dict[str, RobotConfig] = field(default_factory=dict)
    """List of robots to include in the simulation."""
    path: str | None = None
    """Path to the scene file (e.g., MJCF file)."""
