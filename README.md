# Unified Robot Simulation Platform

A unified robot simulation platform that supports Isaac Lab and MuJoCo, primarily designed for Unitree robots and interaction with VLA/World Model/planners.

## ✨ Core Features

- 🔄 **Unified Backend Interface** - Seamless switching between Isaac Lab and MuJoCo
- 🤖 **Model Communication** - Interact with external models (VLA/planners) via ZMQ
- 📡 **Sensor Support** - Camera, IMU, Contact and other sensors
- 🏗️ **Quick Scene Building** - SceneBuilder tool for rapid test environment setup
- ⚡ **Lightweight Design** - Minimal dependencies, modular architecture
- 🎯 **Joint Simulation** - Support for running multiple simulators simultaneously

## 📦 Installation

```bash
# Basic installation
pip install -e .

# Development mode
pip install -e ".[dev]"
```

## 🚀 Quick Start

### 1. Basic Simulation

```bash
# MuJoCo simulation
python scripts/run_sim.py backend=mujoco

# Isaac Lab simulation
python scripts/run_sim.py backend=isaac

# Switch robots
python scripts/run_sim.py backend=isaac robot=h1
```

### 2. Model Interaction

**Terminal 1 - Start simulation server:**
```bash
python example/model_server.py
```

**Terminal 2 - Run model client:**
```bash
python example/model_client_example.py
```

### 3. Integrate Your VLA Model

```python
from robot_sim.communication import ModelClient

class MyVLAClient(ModelClient):
    def compute_control(self, robot_state):
        # Your model inference
        action = self.vla_model.predict(robot_state)
        return {"control_mode": "position", "joint_targets": action}

client = MyVLAClient(server_address="tcp://localhost:5555")
client.run()
```

## 📁 Project Structure

```
robot_sim/
├── robot_sim/
│   ├── backends/          # Simulation backends (Isaac Lab, MuJoCo)
│   │   ├── base.py       # Unified backend interface
│   │   ├── isaac.py      # Isaac Lab implementation
│   │   ├── mujoco.py     # MuJoCo implementation
│   │   ├── manager.py    # Simulation manager
│   │   └── factory.py    # Backend factory
│   ├── communication/     # Communication layer
│   │   ├── server.py     # Simulation server - sends states, receives controls
│   │   ├── client.py     # Model client - connects to sim, runs model
│   │   ├── protocol.py   # Base communication protocol
│   │   └── messages.py   # ZMQ implementation & message formats
│   ├── config/            # Configuration management
│   │   └── loader.py     # Hydra configuration loader
│   ├── controllers/       # Robot controllers
│   │   └── controller.py # PD, trajectory controllers
│   ├── scenes/            # Scene building utilities
│   │   └── builder.py    # Quick environment setup (SceneBuilder)
│   ├── sensors/           # Sensor implementations
│   │   └── base.py       # Camera, IMU, Contact sensors
│   └── presets/           # Robot presets (Unitree Go2, H1, etc.)
├── example/               # Example code
│   ├── basic_sim.py      # Basic simulation examples
│   ├── single_backend.py # Backend switching demo
│   ├── joint_simulation.py # Multi-simulator demo
│   └── scene_builder_demo.py # Scene building examples
├── configs/               # Hydra configuration files
└── docs/                  # Documentation
```

## 🔌 Communication Protocol

### State Information (Sim → Model)

```python
RobotState:
    joint_positions, joint_velocities
    base_position, base_orientation
    base_linear_velocity, base_angular_velocity
    imu_data, contact_forces
    timestamp
```

### Control Signals (Model → Sim)

```python
ControlCommand:
    control_mode: "position" / "velocity" / "torque"
    joint_targets: [n_joints]
    kp, kd: PD gains (optional)
```

## 🏗️ Scene Building

```python
from robot_sim.scenes import SceneBuilder

scene = (
    SceneBuilder()
    .add_ground_plane()
    .add_box("target", position=(5, 0, 0.5), size=(1, 1, 1))
    .add_stairs(num_steps=5)
    .build()
)
```

## 📚 Documentation

- [Unified Backend System](docs/unified_backend.md)
- [Model Communication Workflow](docs/model_communication.md)
- [API Reference](docs/api.md)

## 🎮 Simulator Backends

### IsaacSim
- Supports Isaac Lab 4.0+
- GPU-accelerated physics simulation
- Best for complex scenes and visual rendering

### MuJoCo
- Supports MuJoCo 3.0+
- Fast CPU-based simulation
- Ideal for quick iterations and RL training

## ⚙️ Configuration System

Uses **Hydra** for flexible configuration management:
- YAML-based configuration files
- Command-line parameter overrides
- Hierarchical config composition
- Easy backend/robot switching

