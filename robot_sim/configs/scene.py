from dataclasses import MISSING, dataclass, field

from .object import ObjectConfig
from .robot import RobotConfig
from .sensor import SensorConfig
from .terrain import TerrainConfig


@dataclass
class SceneConfig:
    terrain: TerrainConfig = MISSING
    """Terrain configuration."""
    objects: dict[str, ObjectConfig] = field(default_factory=dict)
    """List of objects to include in the simulation."""
    robots: dict[str, RobotConfig] = field(default_factory=dict)
    """List of robots to include in the simulation."""
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    """Sensor configurations for the robot."""
    path: str | None = None
    """Path to the scene file (e.g., MJCF file)."""
