"""Test communication between server and client."""

from omegaconf import OmegaConf
from robot_sim.client.model_client import ModelClient
from robot_sim.server.sim_server import SimulationServer
from robot_sim.utils.comm import ControlCommand, RobotState

from robot_sim.backends import SimulationManager


def run_server(num_steps: int = 10):
    """Run simulation server in subprocess."""
    cfg = OmegaConf.create(
        {
            "simulation": {"backend": "mujoco", "timestep": 0.001, "num_steps": num_steps},
            "robot": {"type": "go2"},
        }
    )

    manager = SimulationManager(cfg)
    manager.add_backend("main", "mujoco", cfg)
    manager.setup()

    server = SimulationServer(sim_manager=manager, port=5556)
    server.run(num_steps=num_steps)


def run_client(num_steps: int = 10):
    """Run model client in subprocess."""
    client = ModelClient(host="localhost", port=5556)
    client.run(num_steps=num_steps)


class TestCommunication:
    """Test server-client communication."""

    def test_message_serialization(self):
        """Test message serialization."""
        import numpy as np
        from robot_sim.utils.comm.messages import SimulationMessage

        robot_state = RobotState(
            joint_positions=np.zeros(12),
            joint_velocities=np.zeros(12),
            timestamp=0.0,
        )

        msg = SimulationMessage(msg_type="state", robot_state=robot_state)
        msg_dict = msg.to_dict()

        # Deserialize
        msg2 = SimulationMessage.from_dict(msg_dict)
        assert msg2.msg_type == "state"
        assert msg2.robot_state is not None

    def test_control_command(self):
        """Test control command."""
        import numpy as np

        cmd = ControlCommand(
            control_mode="position",
            joint_targets=np.ones(12),
        )

        cmd_dict = cmd.to_dict()
        cmd2 = ControlCommand.from_dict(cmd_dict)

        assert cmd2.control_mode == "position"
        assert len(cmd2.joint_targets) == 12

    # Skip actual server-client test as it requires running processes
    # def test_server_client_communication(self):
    #     """Test full server-client communication."""
    #     # Start server in subprocess
    #     server_proc = mp.Process(target=run_server, args=(5,))
    #     server_proc.start()
    #
    #     time.sleep(1)  # Wait for server to start
    #
    #     # Run client
    #     client_proc = mp.Process(target=run_client, args=(5,))
    #     client_proc.start()
    #
    #     client_proc.join(timeout=10)
    #     server_proc.terminate()
