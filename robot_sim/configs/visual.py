from dataclasses import dataclass, field
from typing import Any


@dataclass
class LightConfig:
    """Light configuration for the renderer."""

    diffuse: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    ambient: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    specular: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional backend-specific headlight options."""


@dataclass
class VisualConfig:
    """Scene-level visual configuration."""

    light: LightConfig
    """Headlight configuration."""
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional backend-specific visual options."""
