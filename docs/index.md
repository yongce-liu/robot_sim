# Robot Simulation Platform Documentation

## Overview

Unified Robot Simulation Platform for Isaac Lab and MuJoCo, mainly supporting Unitree robots.

## Features

- **Multiple Backends**: Support for both Isaac Lab and MuJoCo simulators
- **Robot Presets**: Pre-configured settings for Unitree Go2 and H1 robots
- **Flexible Configuration**: YAML and Hydra-based configuration system
- **Control Utilities**: Built-in controllers and communication protocols

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd robot_sim

# Install with uv
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

## Quick Start

### Basic Simulation

```python
from robot_sim.backend.mujoco import MuJoCoBackend

backend = MuJoCoBackend()
backend.setup()

for _ in range(1000):
    backend.step()

backend.close()
```

### Using Robot Presets

```python
from robot_sim.backend.isaac import IsaacBackend
from robot_sim.presets.unitree.go2 import Go2Config

robot_config = Go2Config()
backend = IsaacBackend()
backend.setup()

# Run simulation...
```

## Configuration

Configuration files are located in the `configs/` directory. You can use YAML files with Hydra for flexible configuration management.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
ruff check .
ruff format .
```

### Type Checking

```bash
mypy robot_sim/
```

## Project Structure

```
robot_sim/
├── robot_sim/         # Main package
│   ├── backend/       # Simulator backends
│   ├── presets/       # Robot configurations
│   ├── utils/         # Utility functions
│   └── config/        # Configuration system
├── example/           # Example scripts
├── scripts/           # Utility scripts
├── tests/             # Test suite
├── configs/           # Configuration files
└── docs/              # Documentation
```

## License

See LICENSE file for details.
