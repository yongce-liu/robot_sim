"""Tests for backend implementations."""

import numpy as np
from omegaconf import DictConfig

from robot_sim.backends import BackendFactory, ObjectState
from robot_sim.configs import SimulatorConfig


class TestMuJoCoBackend:
    """Test MuJoCo backend."""

    def test_init(self, hydra_config_dict: DictConfig) -> None:
        """Test backend initialization."""
        cfg = SimulatorConfig.from_dict(hydra_config_dict)
        backend = BackendFactory(config=cfg).backend
        assert backend is not None

    def test_launch(self, hydra_config_dict: DictConfig) -> None:
        """Test simulation setup."""
        cfg = SimulatorConfig.from_dict(hydra_config_dict)
        backend = BackendFactory(config=cfg).backend
        backend.launch()

    def test_simulate(self, mujoco_backend) -> None:
        """Test a single simulation step."""
        mujoco_backend.simulate()

    # def test_set_root_state(self, mujoco_backend, obj_state: ObjectState) -> None:
    #     """Test setting root state of an object."""
    #     mujoco_backend._set_root_state("g1", obj_state, env_ids=np.array([0]))


class TestIsaacBackend:
    """Test Isaac Lab backend."""

    def test_init(self) -> None:
        """Test backend initialization."""
        # TODO: Implement test
        pass

    def test_setup(self) -> None:
        """Test simulation setup."""
        # TODO: Implement test
        pass
