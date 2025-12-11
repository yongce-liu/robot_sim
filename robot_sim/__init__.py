"""Unified Robot Simulation Platform.

A unified robot simulation platform that supports Isaac Lab and MuJoCo,
designed for Unitree robots and interaction with VLA/World Model/planners.
"""

__version__ = "0.1.0"

# Import main components for easy access
from robot_sim.backends import SimulationManager
from robot_sim.scenes import SceneBuilder
from robot_sim.sensors import IMU, Camera, ContactSensor, SensorManager

__all__ = [
    "SimulationManager",
    "SceneBuilder",
    "Camera",
    "IMU",
    "ContactSensor",
    "SensorManager",
]
