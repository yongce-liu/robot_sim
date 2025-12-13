"""Sensor module for camera, IMU, and other sensors."""

from dataclasses import MISSING, dataclass, field
from enum import Enum

from loguru import logger


class SensorType(Enum):
    """Enumeration of available sensor types."""

    CAMERA = "camera"
    CONTACT_FORCE = "contact_force"


@dataclass
class SensorConfig:
    """Base class for all sensors."""

    type: SensorType = MISSING
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
    """Mount the camera to a specific object or robot. Defaults to None (world frame camera)."""

    # when you want to mount the camera to a robot or object, set mount_to and mount_link
    position: list[float] | None = None
    """Camera position [x, y, z] in world frame. Used when camera is not mounted."""
    look_at: list[float] | None = None
    """Camera look-at point [x, y, z] in world frame. Used when camera is not mounted."""

    # when you want to mount the camera to a robot or object, set mount_to and mount_link
    mount_link: str | None = None
    """Specify the link name to mount the camera to. Defaults to None."""
    mount_pos: list[float] | None = None
    """Position of the camera relative to the mount link. Defaults to (0, 0, 0)."""
    mount_quat: list[float] | None = None
    """Quaternion [w, x, y, z] of the camera relative to the mount link. Defaults to (1, 0, 0, 0)."""

    # camera parameters
    vertical_fov: float = 45.0
    """Vertical field of view in degrees."""
    data_types: list[str] = field(default_factory=lambda: ["rgb"])
    """Data types to capture: ['rgb', 'depth', 'segmentation']."""