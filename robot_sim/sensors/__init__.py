"""Sensor module for camera, IMU, contact, and other sensors.

This module provides various sensor implementations for robot simulation,
including cameras, IMUs, contact sensors, and a sensor manager.
"""

from robot_sim.sensors.base import (
    IMU,
    BaseSensor,
    Camera,
    ContactSensor,
    SensorManager,
)

__all__ = [
    "BaseSensor",
    "Camera",
    "IMU",
    "ContactSensor",
    "SensorManager",
]
