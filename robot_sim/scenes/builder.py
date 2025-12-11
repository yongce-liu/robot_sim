"""Scene building utilities for quick environment setup."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Object:
    """Scene object definition."""
    
    name: str
    type: str  # "box", "sphere", "cylinder", "plane", "mesh"
    position: np.ndarray  # [3]
    orientation: np.ndarray  # [4] quaternion
    size: Optional[np.ndarray] = None  # Dimensions
    mass: float = 1.0
    color: Optional[Tuple[float, float, float]] = None
    mesh_path: Optional[str] = None
    friction: float = 0.5
    restitution: float = 0.0


class SceneBuilder:
    """Utility for building simulation scenes quickly."""
    
    def __init__(self) -> None:
        """Initialize scene builder."""
        self.objects: List[Object] = []
        self.robot = None
    
    def add_ground_plane(
        self,
        size: float = 100.0,
        height: float = 0.0,
        friction: float = 1.0,
    ) -> "SceneBuilder":
        """Add ground plane to scene.
        
        Args:
            size: Plane size
            height: Height of plane
            friction: Friction coefficient
            
        Returns:
            Self for chaining
        """
        self.objects.append(Object(
            name="ground",
            type="plane",
            position=np.array([0, 0, height]),
            orientation=np.array([0, 0, 0, 1]),
            size=np.array([size, size, 0.01]),
            mass=0.0,  # Static
            friction=friction,
        ))
        return self
    
    def add_box(
        self,
        name: str,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float],
        mass: float = 1.0,
        color: Optional[Tuple[float, float, float]] = None,
    ) -> "SceneBuilder":
        """Add box to scene.
        
        Args:
            name: Object name
            position: Position (x, y, z)
            size: Box dimensions (width, depth, height)
            mass: Mass in kg
            color: RGB color (0-1 range)
            
        Returns:
            Self for chaining
        """
        self.objects.append(Object(
            name=name,
            type="box",
            position=np.array(position),
            orientation=np.array([0, 0, 0, 1]),
            size=np.array(size),
            mass=mass,
            color=color,
        ))
        return self
    
    def add_sphere(
        self,
        name: str,
        position: Tuple[float, float, float],
        radius: float,
        mass: float = 1.0,
        color: Optional[Tuple[float, float, float]] = None,
    ) -> "SceneBuilder":
        """Add sphere to scene.
        
        Args:
            name: Object name
            position: Position (x, y, z)
            radius: Sphere radius
            mass: Mass in kg
            color: RGB color (0-1 range)
            
        Returns:
            Self for chaining
        """
        self.objects.append(Object(
            name=name,
            type="sphere",
            position=np.array(position),
            orientation=np.array([0, 0, 0, 1]),
            size=np.array([radius, radius, radius]),
            mass=mass,
            color=color,
        ))
        return self
    
    def add_obstacle_course(
        self,
        num_obstacles: int = 5,
        spacing: float = 2.0,
    ) -> "SceneBuilder":
        """Add simple obstacle course.
        
        Args:
            num_obstacles: Number of obstacles
            spacing: Spacing between obstacles
            
        Returns:
            Self for chaining
        """
        for i in range(num_obstacles):
            x = i * spacing
            height = np.random.uniform(0.1, 0.5)
            self.add_box(
                name=f"obstacle_{i}",
                position=(x, 0, height / 2),
                size=(0.5, 0.5, height),
                mass=0.0,  # Static
                color=(0.8, 0.3, 0.3),
            )
        return self
    
    def add_stairs(
        self,
        num_steps: int = 5,
        step_height: float = 0.15,
        step_depth: float = 0.3,
        step_width: float = 1.0,
    ) -> "SceneBuilder":
        """Add stairs to scene.
        
        Args:
            num_steps: Number of steps
            step_height: Height of each step
            step_depth: Depth of each step
            step_width: Width of steps
            
        Returns:
            Self for chaining
        """
        for i in range(num_steps):
            self.add_box(
                name=f"step_{i}",
                position=(i * step_depth, 0, (i + 0.5) * step_height),
                size=(step_depth, step_width, step_height),
                mass=0.0,  # Static
                color=(0.6, 0.6, 0.6),
            )
        return self
    
    def add_robot(self, robot_config: Dict[str, Any]) -> "SceneBuilder":
        """Add robot to scene.
        
        Args:
            robot_config: Robot configuration dictionary
            
        Returns:
            Self for chaining
        """
        self.robot = robot_config
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return scene configuration.
        
        Returns:
            Scene configuration dictionary
        """
        return {
            "objects": [
                {
                    "name": obj.name,
                    "type": obj.type,
                    "position": obj.position.tolist(),
                    "orientation": obj.orientation.tolist(),
                    "size": obj.size.tolist() if obj.size is not None else None,
                    "mass": obj.mass,
                    "color": obj.color,
                    "mesh_path": obj.mesh_path,
                    "friction": obj.friction,
                    "restitution": obj.restitution,
                }
                for obj in self.objects
            ],
            "robot": self.robot,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Alias for build()."""
        return self.build()


# Preset scenes
def create_empty_scene() -> SceneBuilder:
    """Create empty scene with just ground plane."""
    return SceneBuilder().add_ground_plane()


def create_obstacle_scene() -> SceneBuilder:
    """Create scene with obstacles."""
    return (
        SceneBuilder()
        .add_ground_plane()
        .add_obstacle_course(num_obstacles=5)
    )


def create_stairs_scene() -> SceneBuilder:
    """Create scene with stairs."""
    return (
        SceneBuilder()
        .add_ground_plane()
        .add_stairs(num_steps=5)
    )
