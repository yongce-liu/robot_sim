# Model Communication Workflow

## Project Architecture Description

This project supports **bidirectional communication between the simulator and external models (VLA/World Model/Planner)**:

```
┌─────────────────┐         ZMQ          ┌──────────────────┐
│  robot_sim      │◄─────────────────────►│  Your Model      │
│  (Simulation)   │    State/Control      │  (VLA/Planner)   │
└─────────────────┘                       └──────────────────┘
```

## Core Components

### 1. **Communication Layer** (`robot_sim/utils/comm/`)
- **ZMQProtocol** - ZMQ communication protocol (lightweight, efficient)
- **Messages** - Standardized message formats (RobotState, ControlCommand, etc.)

### 2. **Server Side** (`robot_sim/server/`)
- **SimulationServer** - Runs simulation, sends state, receives control

### 3. **Client Side** (`robot_sim/client/`)
- **ModelClient** - Connects to simulation, receives state, sends control
- **VLAModelWrapper** - VLA model wrapper

### 4. **Sensors** (`robot_sim/utils/sensors.py`)
- Camera, IMU, ContactSensor, etc.
- SensorManager - Unified management of multiple sensors

### 5. **Scene Building** (`robot_sim/utils/scene_builder.py`)
- SceneBuilder - Quickly build simulation scenes
- Preset scenes (obstacles, stairs, etc.)

## Usage Workflow

### Step 1: Start the Simulation Server

```bash
# Terminal 1
python example/model_server.py
```

The server will:
- Initialize the simulation environment
- Set up sensors
- Wait for model client connection
- Loop: Send state → Receive control → Execute control → Update simulation

### Step 2: Run the Model Client

```bash
# Terminal 2
python example/model_client_example.py
```

The client will:
- Connect to the simulation server
- Loop: Request state → Model inference → Send control

### Step 3: Integrate Your Model

Modify `model_client_example.py`:

```python
from robot_sim.client import ModelClient
from robot_sim.utils.comm.messages import RobotState, ControlCommand

class MyVLAClient(ModelClient):
    def __init__(self):
        # Load your VLA model
        from your_vla_model import VLAModel
        self.vla = VLAModel.load("model.pt")
        super().__init__(model=self.vla)
    
    def compute_control(self, robot_state: RobotState) -> ControlCommand:
        # VLA model inference
        observation = {
            "joint_pos": robot_state.joint_positions,
            "joint_vel": robot_state.joint_velocities,
            "base_vel": robot_state.base_linear_velocity,
        }
        
        action = self.vla.predict(observation)
        
        return ControlCommand(
            control_mode="position",
            joint_targets=action,
        )
```

## Message Formats

### State Information Sent to the Model

```python
RobotState:
    - joint_positions: [n_joints]
    - joint_velocities: [n_joints]
    - joint_torques: [n_joints] (optional)
    - base_position: [3]
    - base_orientation: [4] (quaternion)
    - base_linear_velocity: [3]
    - base_angular_velocity: [3]
    - imu_data: dict
    - contact_forces: array
    - timestamp: float
```

### Control Signals Received from the Model

```python
ControlCommand:
    - control_mode: "position" / "velocity" / "torque"
    - joint_targets: [n_joints]
    - kp: [n_joints] (optional PD gains)
    - kd: [n_joints] (optional PD gains)
    - base_velocity_cmd: [3] (optional for mobile base)
```

## Scene Building Example

```python
from robot_sim.utils.scene_builder import SceneBuilder

scene = (
    SceneBuilder()
    .add_ground_plane()
    .add_box("obstacle1", position=(2, 0, 0.5), size=(1, 1, 1))
    .add_stairs(num_steps=5)
    .build()
)
```

## Configuration File

```yaml
# configs/model_comm.yaml
simulation:
  backend: mujoco
  timestep: 0.001
  num_steps: 10000

communication:
  port: 5555
  host: localhost
  serialization: json  # or pickle

sensors:
  camera:
    enabled: true
    width: 640
    height: 480
  imu:
    enabled: true
    update_freq: 100
```

## Lightweight Design

✅ **ZMQ Communication** - Lightweight, fast, cross-language  
✅ **JSON Serialization** - Human-readable, easy to debug  
✅ **Modular** - Only load needed components  
✅ **Minimal Dependencies** - numpy, pyzmq, hydra  
✅ **Plug-and-Play** - Directly replace your model

## Extensions

### Adding Camera Images

```python
# Server side
from robot_sim.utils.sensors import Camera
camera = Camera(width=640, height=480)
sensor_mgr.add_sensor(camera)

# Images will be automatically included in RobotState
```

### Supporting Multiple Cameras

```python
sensor_mgr.add_sensor(Camera(name="front_cam"))
sensor_mgr.add_sensor(Camera(name="wrist_cam"))
```

### Custom Sensors

```python
from robot_sim.utils.sensors import BaseSensor

class CustomSensor(BaseSensor):
    def update(self, sim_state):
        # Your sensor logic
        pass
    
    def get_data(self):
        return self._data
```
