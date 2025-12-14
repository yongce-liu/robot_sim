"""Tests for backend implementations."""

import numpy as np

from robot_sim.backends.types import ArrayState


class TestMuJoCoBackend:
    """Test MuJoCo backend."""

    def test_simulate(self, mujoco_backend) -> None:
        """Test a single simulation step."""
        mujoco_backend.simulate()

    def test_set_root_state(self, mujoco_backend, array_state: ArrayState, robot_name: str) -> None:
        """Test setting root state of an object."""

        mujoco_backend._set_root_state(robot_name, array_state.objects[robot_name], env_ids=np.array([0]))
        mujoco_backend.simulate()

    def test_set_joint_state(self, mujoco_backend, array_state: ArrayState, robot_name: str) -> None:
        """Test setting joint state of an object."""
        mujoco_backend._set_joint_state(robot_name, array_state.objects[robot_name], env_ids=np.array([0]))
        mujoco_backend.simulate()

    def test_set_actions(self, mujoco_backend, array_state: ArrayState, robot_name: str) -> None:
        """Test setting actions for an object."""
        actions = {robot_name: array_state.objects[robot_name].joint_pos}
        mujoco_backend._set_actions(actions, env_ids=np.array([0]))
        mujoco_backend.simulate()


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
