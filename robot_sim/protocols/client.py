"""Model client base for connecting to simulation server."""

any, dict


class ModelClient:
    """Base class for model clients that connect to simulation server.

    Example:
        class MyVLAClient(ModelClient):
            def compute_control(self, robot_state):
                action = self.vla_model.predict(robot_state)
                return ControlCommand(joint_targets=action)

        client = MyVLAClient(server_address="tcp://localhost:5555")
        client.run()
    """

    def __init__(self, server_address: str = "tcp://localhost:5555") -> None:
        """Initialize model client.

        Args:
            server_address: ZMQ server address
        """
        self.server_address = server_address
        self.protocol = None

    def connect(self) -> None:
        """Connect to simulation server."""
        from robot_sim.protocols.messages import ZMQProtocol

        self.protocol = ZMQProtocol(is_server=False, port=self.server_address)
        print(f"Connected to simulation server at {self.server_address}")

    def compute_control(self, robot_state: dict[str, any]) -> dict[str, any]:
        """Compute control commands based on robot state.

        Override this method to implement your model logic.

        Args:
            robot_state: Current robot state from simulation

        Returns:
            Control commands to send to simulation
        """
        raise NotImplementedError("Must implement compute_control method")

    def run(self, num_steps: int | None = None) -> None:
        """Run client control loop.

        Args:
            num_steps: Number of steps to run (None for infinite)
        """
        self.connect()

        step = 0
        try:
            while num_steps is None or step < num_steps:
                # Receive robot state
                robot_state = self.protocol.receive()

                # Compute control
                control_cmd = self.compute_control(robot_state)

                # Send control command
                self.protocol.send(control_cmd)

                step += 1

        except KeyboardInterrupt:
            print("\nClient interrupted")
        finally:
            self.close()

    def close(self) -> None:
        """Close connection to server."""
        if self.protocol:
            self.protocol.close()
