# Quick Start Guide

## 5-Minute Quick Start

### 1. Install Dependencies

```bash
cd robot_sim
pip install -e .
```

### 2. Test Basic Simulation

```bash
# Run MuJoCo simulation
python scripts/run_sim.py backend=mujoco simulation.num_steps=100
```

### 3. Test Model Communication

**Terminal 1 - Start simulation server:**
```bash
python example/model_server.py simulation.num_steps=500
```

**Terminal 2 - Start model client:**
```bash
python example/model_client_example.py
```

You should see:
- Server: Sending robot state
- Client: Receive state → Compute control → Send back to server
- Server: Apply control → Update simulation

### 4. Test Scene Building

```bash
python example/scene_builder_demo.py
```

## Integrate Your Model

### Method 1: Modify ModelClient

Edit `example/model_client_example.py`:

```python
class MyModelClient(ModelClient):
    def __init__(self):
        # Load your model
        import torch
        self.model = torch.load("your_model.pt")
        super().__init__(model=self.model)
    
    def compute_control(self, robot_state: RobotState) -> ControlCommand:
        # Prepare input
        obs = np.concatenate([
            robot_state.joint_positions,
            robot_state.joint_velocities,
            robot_state.base_linear_velocity,
        ])
        
        # Model inference
        with torch.no_grad():
            action = self.model(torch.from_numpy(obs)).numpy()
        
        return ControlCommand(
            control_mode="position",
            joint_targets=action,
        )
```

### Method 2: Standalone Script

```python
# my_vla_controller.py
from robot_sim.utils.comm import ZMQProtocol, SimulationMessage, ControlCommand
import numpy as np

# Connect to simulation
comm = ZMQProtocol(host="localhost", port=5555, mode="client")

for step in range(1000):
    # Request state
    request = SimulationMessage(msg_type="request_state")
    comm.send(request.to_dict())
    
    # Receive state
    response = comm.receive()
    msg = SimulationMessage.from_dict(response)
    state = msg.robot_state
    
    # Your model inference
    action = your_model.predict(state)
    
    # Send control
    control = ControlCommand(control_mode="position", joint_targets=action)
    control_msg = SimulationMessage(msg_type="control", control_command=control)
    comm.send(control_msg.to_dict())
    
    # Wait for confirmation
    comm.receive()

comm.close()
```

## Common Configurations

### Change Communication Port

```bash
# Server
python example/model_server.py communication.port=6666

# Client (modify port parameter in code)
client = ModelClient(port=6666)
```

### Add Camera

```python
# In model_server.py
from robot_sim.utils.sensors import Camera

sensor_mgr = SensorManager()
sensor_mgr.add_sensor(Camera(name="front_cam", width=640, height=480))
sensor_mgr.add_sensor(Camera(name="wrist_cam", width=320, height=240))
```

### Use Different Backends

```bash
# Isaac Lab
python example/model_server.py backend=isaac

# MuJoCo
python example/model_server.py backend=mujoco
```

### Custom Scene

```python
# Modify model_server.py
from robot_sim.utils.scene_builder import SceneBuilder

scene = (
    SceneBuilder()
    .add_ground_plane()
    .add_box("obstacle", position=(2, 0, 0.5), size=(1, 1, 1))
    .add_stairs(num_steps=3)
)

# Load scene to backend (TODO: implement loading logic)
```

## Debugging Tips

### 1. Check Message Content

```python
# On client side
print(f"Received state: {msg.robot_state.joint_positions}")
print(f"Base velocity: {msg.robot_state.base_linear_velocity}")
```

### 2. Visualization

```python
# Add simple state printing
if step % 10 == 0:
    print(f"Step {step}: joint_pos = {state.joint_positions[:3]}")
```

### 3. Test Different Control Modes

```python
# Position control
ControlCommand(control_mode="position", joint_targets=target_pos)

# Velocity control
ControlCommand(control_mode="velocity", joint_targets=target_vel)

# Torque control
ControlCommand(control_mode="torque", joint_targets=target_torque)
```

## Next Steps

- Read [Model Communication Documentation](model_communication.md)
- Check [Unified Backend System](unified_backend.md)
- Explore more examples in the `example/` directory
- Implement specific backend logic as needed (isaac.py, mujoco.py)

## Troubleshooting

**Connection Timeout:**
- Make sure server starts first
- Check if port is occupied: `lsof -i :5555`