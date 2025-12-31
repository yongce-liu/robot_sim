from dataclasses import dataclass, field
from enum import Enum


class TerrainType(Enum):
    PLANE = "plane"
    CUSTOM = "custom"


@dataclass
class TerrainConfig:
    type: TerrainType | None = None
    path: str | None = None
    """Path to a custom terrain mesh file. Used only if terrain type is CUSTOM."""
    properties: dict = field(default_factory=dict)
    """take effect when path is None."""

    def __post_init__(self):
        if self.type == TerrainType.CUSTOM:
            assert self.path is not None, "Custom terrain type requires a valid mesh file path."
