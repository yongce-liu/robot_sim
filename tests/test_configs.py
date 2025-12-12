from omegaconf import DictConfig

from robot_sim.configs import SimulatorConfig

from .conftest import assert_class_type


def test_load_config_with_hydra(hydra_config: DictConfig) -> None:
    """Load configuration using Hydra compose."""

    assert hydra_config is not None
    assert "backend" in hydra_config
    assert "sim" in hydra_config
    assert "scene" in hydra_config

    assert hydra_config.backend == "mujoco"


def test_hydra_config_to_simulator_config(hydra_config_dict: dict) -> None:
    """Convert Hydra config to SimulatorConfig."""

    cfg = SimulatorConfig.from_dict(hydra_config_dict)
    assert_class_type(cfg, SimulatorConfig)
