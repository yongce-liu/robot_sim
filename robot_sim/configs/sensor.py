"""Sensor module for camera, IMU, and other sensors."""

from dataclasses import MISSING, field
from enum import Enum

from robot_sim.utils import configclass


class SensorType(Enum):
    """Enumeration of available sensor types."""

    CAMERA = "camera"
    CONTACTSENSOR = "contactsensor"


# Registry to map sensor type to concrete class
_SENSOR_TYPE_REGISTRY: dict[SensorType, type] = {}


@configclass
class SensorConfig:
    """Base class for all sensors."""

    type: SensorType = MISSING
    """Type of the sensor."""
    freq: float | None = None
    """Update frequency in Hz. It should less than or equal to the simulation frequency."""

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "SensorConfig":
        """Override from_dict to handle polymorphic sensor types."""
        if cls is not SensorConfig:
            # If called on a concrete subclass, use default behavior
            return super().from_dict(cfg_dict)

        # If called on base class, determine concrete type from 'type' field
        if "type" not in cfg_dict:
            raise ValueError("Sensor configuration must include 'type' field")

        sensor_type_str = cfg_dict["type"]
        sensor_type = SensorType(sensor_type_str)

        if sensor_type not in _SENSOR_TYPE_REGISTRY:
            raise ValueError(
                f"Unknown sensor type: {sensor_type}. Available types: {list(_SENSOR_TYPE_REGISTRY.keys())}"
            )

        concrete_class = _SENSOR_TYPE_REGISTRY[sensor_type]
        return concrete_class.from_dict(cfg_dict)


@configclass
class Camera(SensorConfig):
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


# Register sensor types
_SENSOR_TYPE_REGISTRY[SensorType.CAMERA] = Camera
