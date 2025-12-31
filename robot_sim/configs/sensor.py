"""Sensor module for camera, IMU, and other sensors."""

from dataclasses import dataclass, field
from enum import Enum


class SensorType(Enum):
    """Enumeration of available sensor types."""

    CAMERA = "camera"
    CONTACT_FORCE = "contact_force"


@dataclass
class SensorConfig:
    """Base class for all sensors."""

    type: SensorType
    """Type of the sensor."""
    freq: float | None = None
    """Update frequency in Hz. It should less than or equal to the simulation frequency."""
    data_buffer_length: int = 1
    """Maximum length of the data queue."""


@dataclass
class CameraConfig(SensorConfig):
    """Camera sensor for RGB, depth, and segmentation."""

    width: int = 640
    """Image width in pixels."""
    height: int = 480
    """Image height in pixels."""

    mount_to: str | None = None
    """Mount the camera to a specific link of object or robot. Defaults to None (world frame camera)."""

    # if mount_to is None, it represents a world frame camera
    # elif mount_to is not None, it represents a pose with respect to the link frame

    # position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # """Camera position [x, y, z] in world frame. Used when camera is not mounted."""
    # orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    # """Camera orientation [w, x, y, z] in world frame. Used when camera is not mounted."""
    pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    """Camera pose [x, y, z, qw, qx, qy, qz] in world frame. Used when camera is not mounted."""

    look_at: list[float] | None = None
    """Point [x, y, z] for the camera to look at in world frame. Used when camera is not mounted."""

    # camera parameters
    vertical_fov: float = 45.0
    """Vertical field of view in degrees."""
    data_types: list[str] = field(default_factory=lambda: ["rgb"])
    """Data types to capture: ['rgb', 'depth', 'segmentation']."""
