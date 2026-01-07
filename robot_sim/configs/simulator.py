from dataclasses import dataclass, field
from enum import Enum

from robot_sim.utils.config import configclass

from .scene import SceneConfig


class BackendType(Enum):
    """Enumeration of supported backend types."""

    ISAAC = "isaac"
    MUJOCO = "mujoco"
    UNITREE = "unitree"
    ELASTIC = "elastic"


@dataclass
class PhysicsConfig:
    dt: float
    """Simulation timestep."""
    render_interval: int
    """Interval (in steps) at which to render the simulation."""
    device: str
    """Device to run the simulation on (e.g., 'cpu', 'cuda:0')."""
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    """Gravity vector for the simulation."""
    headless: bool = False
    """Whether to run the simulation in headless mode (no GUI)."""
    num_envs: int = 1
    """Number of parallel simulation environments."""


@configclass
class SimulatorConfig:
    backend: BackendType
    """Type of simulation backend to use."""
    sim: PhysicsConfig
    """Configuration for the physics simulation."""
    scene: SceneConfig
    """Configuration for the simulation scene."""
    spec: dict[str, dict] = field(default_factory=dict)
    """Backend-specific configuration options."""
    extras: dict = field(default_factory=dict)
    """Additional task-specific configuration options, e.g., you can assign decimation, max_episode_steps here."""

    @staticmethod
    def get_dacite_config():
        import dacite

        from robot_sim.configs.sensor import CameraConfig, SensorConfig, SensorType

        base_dacite_config = dacite.Config(
            cast=[Enum],
            strict=True,
        )

        # Register sensor types
        def sensor_config_hook(_data_dict: dict) -> SensorConfig:
            sensor_type = SensorType(_data_dict.get("type"))
            if sensor_type == SensorType.CAMERA:
                return dacite.from_dict(data_class=CameraConfig, data=_data_dict, config=base_dacite_config)
            else:
                raise ValueError(f"Unsupported sensor type: {sensor_type}")

        dacite_config = dacite.Config(
            type_hooks={
                SensorConfig: sensor_config_hook,
            },
            cast=[Enum],
            strict=True,
        )
        # dacite will not call __post_init__
        return dacite_config
