from dataclasses import MISSING, dataclass


@dataclass
class TerrainConfig:
    type: str = MISSING
    """Type of the terrain (e.g., 'heightfield', 'plane', 'trimesh')."""
    # size: list[float] = field(default_factory=lambda: [10.0, 10.0])
    # """Size of the terrain [length, width]."""
    # heightmap_path: str | None = None
    # """Path to the heightmap file (if applicable)."""
    # friction: float = 1.0
    # """Friction coefficient of the terrain."""
    # restitution: float = 0.0
    # """Restitution coefficient of the terrain."""
    # texture_path: str | None = None
    # """Path to the texture file for the terrain surface."""
