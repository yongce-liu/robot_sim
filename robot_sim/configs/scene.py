from dataclasses import dataclass, field

from .object import ObjectConfig
from .sensor import SensorConfig
from .terrain import TerrainConfig
from .visual import VisualConfig


@dataclass
class SceneConfig:
    path: str | None = None
    """Path to the scene file (e.g., MJCF file)."""
    terrain: TerrainConfig | None = None
    """Terrain configuration. If None, the terrain will be loaded from the scene file."""
    visual: VisualConfig | None = None
    """Visual configuration for the scene. If None, default visual settings or scene file settings will be used."""
    objects: dict[str, ObjectConfig] = field(default_factory=dict)
    """List of objects and robots to include in the simulation."""
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    """Sensor configurations for the robot."""
