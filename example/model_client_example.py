"""Example: Model client connecting to simulation."""

import numpy as np

from robot_sim.client.model_client import ModelClient, VLAModelWrapper
from robot_sim.utils.comm.messages import ControlCommand, RobotState


class SimpleController:
    """Simple controller example (replace with your VLA model)."""
    
    def __call__(self, robot_state: RobotState) -> np.ndarray:
        """Compute action.
        
        Args:
            robot_state: Current robot state
            
        Returns:
            Joint position targets
        """
        # Example: Simple sine wave motion
        t = robot_state.timestamp
        amplitude = 0.1
        frequency = 1.0
        
        action = robot_state.joint_positions.copy()
        action += amplitude * np.sin(2 * np.pi * frequency * t)
        
        return action


class MyModelClient(ModelClient):
    """Custom model client with your inference logic."""
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        # Initialize your model here
        model = SimpleController()
        super().__init__(host=host, port=port, model=model)
    
    def compute_control(self, robot_state: RobotState) -> ControlCommand:
        """Compute control from state.
        
        Args:
            robot_state: Current robot state
            
        Returns:
            Control command
        """
        # Your model inference
        action = self.model(robot_state)
        
        return ControlCommand(
            control_mode="position",
            joint_targets=action,
        )


def main() -> None:
    """Run model client.
    
    Make sure simulation server is running first:
        python example/model_server.py
    """
    print("=" * 60)
    print("Model Client Example")
    print("=" * 60)
    
    # Create client
    client = MyModelClient(host="localhost", port=5555)
    
    print("\nConnecting to simulation server...")
    print("Running control loop...")
    
    # Run for 1000 steps
    try:
        client.run(num_steps=1000)
    except KeyboardInterrupt:
        print("\nClient interrupted")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\nClient finished")


if __name__ == "__main__":
    main()
