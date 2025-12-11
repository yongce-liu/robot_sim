"""Tests for unified backend system."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from robot_sim.backends import BaseBackend, IsaacBackend, MuJoCoBackend, SimulationManager, create_backend


class TestBaseBackend:
    """Test base backend interface."""

    def test_backend_creation(self) -> None:
        """Test creating backend instances."""
        isaac = IsaacBackend()
        mujoco = MuJoCoBackend()

        assert isinstance(isaac, BaseBackend)
        assert isinstance(mujoco, BaseBackend)
        assert isaac.backend_name == "IsaacBackend"
        assert mujoco.backend_name == "MuJoCoBackend"

    def test_factory_creation(self) -> None:
        """Test backend factory."""
        isaac = create_backend("isaac")
        mujoco = create_backend("mujoco")

        assert isinstance(isaac, IsaacBackend)
        assert isinstance(mujoco, MuJoCoBackend)

        with pytest.raises(ValueError):
            create_backend("unknown")


class TestSimulationManager:
    """Test simulation manager."""

    def test_manager_creation(self) -> None:
        """Test creating simulation manager."""
        config = OmegaConf.create(
            {
                "simulation": {"backend": "mujoco", "timestep": 0.001, "num_steps": 100},
                "robot": {"type": "go2"},
            }
        )

        manager = SimulationManager(config)
        assert manager.num_backends == 0

    def test_add_backend(self) -> None:
        """Test adding backends to manager."""
        config = OmegaConf.create(
            {
                "simulation": {"backend": "mujoco", "timestep": 0.001, "num_steps": 100},
                "robot": {"type": "go2"},
            }
        )

        manager = SimulationManager(config)
        manager.add_backend("backend1", "mujoco")

        assert manager.num_backends == 1
        assert "backend1" in manager.backend_names

    def test_multi_backend(self) -> None:
        """Test adding multiple backends."""
        config = OmegaConf.create(
            {
                "simulation": {"backend": "mujoco", "timestep": 0.001, "num_steps": 100},
                "robot": {"type": "go2"},
            }
        )

        manager = SimulationManager(config)
        manager.add_backend("isaac", "isaac")
        manager.add_backend("mujoco", "mujoco")

        assert manager.num_backends == 2
        assert set(manager.backend_names) == {"isaac", "mujoco"}

    def test_setup_and_step(self) -> None:
        """Test setup and stepping simulation."""
        config = OmegaConf.create(
            {
                "simulation": {"backend": "mujoco", "timestep": 0.001, "num_steps": 10},
                "robot": {"type": "go2"},
            }
        )

        manager = SimulationManager(config)
        manager.add_backend("test", "mujoco")

        # Setup
        manager.setup()
        assert manager._is_setup

        # Step
        results = manager.step()
        assert "test" in results
        assert "observation" in results["test"]

        # Close
        manager.close()


class TestBackendMethods:
    """Test backend methods."""

    def test_get_state(self) -> None:
        """Test getting state from backend."""
        backend = MuJoCoBackend()
        state = backend.get_state()

        assert "joint_positions" in state
        assert "joint_velocities" in state
        assert "base_position" in state
        assert "base_orientation" in state

    def test_apply_action(self) -> None:
        """Test applying action."""
        backend = MuJoCoBackend()
        action = np.zeros(12)

        # Should not raise error
        backend.apply_action(action)

    def test_get_observation(self) -> None:
        """Test getting observation."""
        backend = IsaacBackend()
        obs = backend.get_observation()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (48,)  # Placeholder size
