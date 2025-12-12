from pathlib import Path

from robot_sim.configs import SimulatorConfig

from .conftest import assert_class_type


def test_load_config(default_config_path: Path) -> None:
    """Load the default simulator configuration from YAML."""

    cfg = SimulatorConfig.from_yaml(default_config_path)
    assert_class_type(cfg, SimulatorConfig)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
