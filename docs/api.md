# API Reference

## Backend

### MuJoCoBackend

```python
class MuJoCoBackend:
    """MuJoCo simulation backend."""
    
    def __init__(self) -> None:
        """Initialize MuJoCo backend."""
        
    def setup(self) -> None:
        """Setup the simulation environment."""
        
    def step(self) -> None:
        """Step the simulation."""
        
    def reset(self) -> None:
        """Reset the simulation."""
        
    def close(self) -> None:
        """Close the simulation."""
```

### IsaacBackend

```python
class IsaacBackend:
    """Isaac Lab simulation backend."""
    
    def __init__(self) -> None:
        """Initialize Isaac Lab backend."""
        
    def setup(self) -> None:
        """Setup the simulation environment."""
        
    def step(self) -> None:
        """Step the simulation."""
        
    def reset(self) -> None:
        """Reset the simulation."""
        
    def close(self) -> None:
        """Close the simulation."""
```

## Robot Presets

### Go2Config

Configuration for Unitree Go2 quadruped robot.

### H1Config

Configuration for Unitree H1 humanoid robot.

## Utilities

### Controllers

- `BaseController`: Base class for controllers
- `PDController`: PD controller implementation

### Communication

- `CommunicationProtocol`: Base protocol class

## Configuration

### ConfigLoader

Load and save configuration files in YAML format.
