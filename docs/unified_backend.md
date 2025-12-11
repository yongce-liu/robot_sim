# Unified Backend System

This document explains the design and usage of the unified backend system.

## Architecture Overview

### Core Components

1. **BaseBackend** - Abstract base class, defines unified interface
2. **IsaacBackend** - Isaac Lab backend implementation
3. **MuJoCoBackend** - MuJoCo backend implementation
4. **SimulationManager** - Simulation manager, supports single/multi-backend
5. **create_backend()** - Factory function, convenient backend creation

## Unified Interface

All backends implement the following interface:

```python
class BaseBackend(ABC):
    def setup() -> None
    def step() -> Dict[str, Any]
    def reset() -> Dict[str, Any]
    def close() -> None
    def get_state() -> Dict[str, Any]
    def set_state(state: Dict[str, Any]) -> None
    def apply_action(action: np.ndarray) -> None
    def get_observation() -> np.ndarray
```

## Usage Methods

### 1. Single Backend Simulation (Easy Switching)

Switch simulators by modifying configuration parameters:

```bash
# Use MuJoCo
python scripts/run_sim.py backend=mujoco

# Use Isaac Lab
python scripts/run_sim.py backend=isaac

# Switch robot
python scripts/run_sim.py backend=isaac robot=h1
```

Code example:

```python
from robot_sim.backends import SimulationManager

# Create manager
manager = SimulationManager(cfg)

# Add backend (automatically selected from config)
manager.add_backend(
    name="main",
    backend_type=cfg.simulation.backend,
    config=cfg
)

# Run simulation
manager.setup()
for _ in range(num_steps):
    results = manager.step()
manager.close()
```

### 2. Joint Simulation (Multiple Backends Running Simultaneously)

Run multiple simulators simultaneously for comparison and validation:

```bash
# Run joint simulation
python example/joint_simulation.py

# Use specific config
python example/joint_simulation.py --config-name=joint_sim
```

Code example:

```python
from robot_sim.backends import SimulationManager

# Create manager
manager = SimulationManager(cfg)

# Add multiple backends
manager.add_backend("isaac", "isaac", cfg)
manager.add_backend("mujoco", "mujoco", cfg)

# Initialize all backends
manager.setup()

# Run synchronously
for step in range(num_steps):
    # Apply same action to all backends
    manager.apply_action(action)
    
    # All backends step simultaneously
    results = manager.step()
    
    # Compare states
    if step % 100 == 0:
        differences = manager.compare_states()
        print(differences)

manager.close()
```

### 3. State Synchronization

Synchronize states in joint simulation:

```python
# Copy mujoco state to isaac
manager.synchronize_states(source="mujoco", targets=["isaac"])

# Synchronize to all other backends
manager.synchronize_states(source="mujoco")
```

### 4. Using Factory Function

```python
from robot_sim.backends import create_backend

# Directly create backend
backend = create_backend("mujoco", config)
backend.setup()
backend.step()
```

## Example Programs

### Single Backend Examples
- `example/single_backend.py` - Single backend simulation, supports parameter switching
- `scripts/run_sim.py` - Main run script

### Joint Simulation Examples
- `example/joint_simulation.py` - Multiple backends running simultaneously
- Config file: `configs/joint_sim.yaml`

## Extending New Backend

Add support for new simulators:

1. Create new backend class inheriting from `BaseBackend`
2. Implement all abstract methods
3. Register in `factory.py`
4. Create corresponding configuration file

```python
from robot_sim.backends.base import BaseBackend

class NewBackend(BaseBackend):
    def setup(self) -> None:
        # Implement setup logic
        self._is_initialized = True
    
    def step(self) -> Dict[str, Any]:
        # Implement step logic
        self._step_count += 1
        return {"observation": ..., "reward": ..., "done": ...}
    
    # Implement other methods...
```

## Configuration File Structure

```yaml
simulation:
  backend: mujoco  # or isaac
  timestep: 0.001
  num_steps: 1000

robot:
  type: go2
  
# Joint simulation specific config
joint_simulation:
  enabled: true
  backends: ["isaac", "mujoco"]
  synchronize_states: false
  compare_interval: 100
```

## Advantages

✅ **Unified Interface** - All backends use the same API  
✅ **Easy Switching** - Switch simulators via config parameters  
✅ **Joint Simulation** - Run multiple simulators simultaneously for comparison  
✅ **Easy Extension** - Add new backends by implementing base class  
✅ **Type Safety** - Full type annotation support  
✅ **State Management** - Unified state get/set interface
