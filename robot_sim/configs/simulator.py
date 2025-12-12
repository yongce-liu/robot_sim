from dataclasses import MISSING, field
from enum import Enum

from .base import configclass
from .scene import SceneConfig


class BackendType(Enum):
    """Enumeration of supported backend types."""

    ISAAC = "isaac"
    MUJOCO = "mujoco"


@configclass
class PhysicsConfig:
    dt: float = MISSING
    """Simulation timestep."""
    render_interval: int = MISSING
    """Interval (in steps) at which to render the simulation."""
    device: str = MISSING
    """Device to run the simulation on (e.g., 'cpu', 'cuda:0')."""
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    """Gravity vector for the simulation."""
    headless: bool = False
    """Whether to run the simulation in headless mode (no GUI)."""
    num_envs: int = 1
    """Number of parallel simulation environments."""


@configclass
class SimulatorConfig:
    backend: BackendType = MISSING
    sim: PhysicsConfig = MISSING
    """Configuration for the physics simulation."""
    scene: SceneConfig = MISSING
    """Configuration for the simulation scene."""
