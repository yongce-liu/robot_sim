from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisualConfig:
    """Scene-level visual configuration."""

    light: dict[str, Any] = field(default_factory=dict)
    """Headlight configuration."""
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional backend-specific visual options."""
