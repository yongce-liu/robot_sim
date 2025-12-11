from pathlib import Path

from robot_sim.configs import SimulatorConfig


def test_load_config(default_config_path: Path) -> None:
    """Load the default simulator configuration from YAML."""

    cfg = SimulatorConfig.from_yaml(default_config_path)
    assert isinstance(cfg, SimulatorConfig)
