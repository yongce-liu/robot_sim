from dataclasses import dataclass, field

from .object import ObjectConfig
from .sensor import SensorConfig
from .terrain import TerrainConfig


@dataclass
class SceneConfig:
    terrain: TerrainConfig | None = None
    """Terrain configuration.
       If None, the terrain will be loaded from the scene file."""
    objects: dict[str, ObjectConfig] = field(default_factory=dict)
    """List of objects and robots to include in the simulation."""
    # robots: dict[str, RobotConfig] = field(default_factory=dict)
    # """List of robots to include in the simulation."""
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    """Sensor configurations for the robot."""
    path: str | None = None
    """Path to the scene file (e.g., MJCF file)."""
