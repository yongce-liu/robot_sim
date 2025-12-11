"""Sensor module for camera, IMU, and other sensors."""

from abc import ABC, abstractmethod

any, dict

import numpy as np


class BaseSensor(ABC):
    """Base class for all sensors."""

    def __init__(self, name: str, update_freq: float = 100.0) -> None:
        """Initialize sensor.

        Args:
            name: Sensor name
            update_freq: Update frequency in Hz
        """
        self.name = name
        self.update_freq = update_freq
        self._data = None

    @abstractmethod
    def update(self, sim_state: dict[str, any]) -> None:
        """Update sensor data from simulation state."""
        pass

    @abstractmethod
    def get_data(self) -> any:
        """Get current sensor data."""
        pass


class Camera(BaseSensor):
    """Camera sensor for RGB, depth, and segmentation."""

    def __init__(
        self,
        name: str = "camera",
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        update_freq: float = 30.0,
        mode: str = "rgb",  # "rgb", "depth", "rgbd", "all"
    ) -> None:
        """Initialize camera.

        Args:
            name: Camera name
            width: Image width
            height: Image height
            fov: Field of view in degrees
            update_freq: Update frequency in Hz
            mode: Camera mode
        """
        super().__init__(name, update_freq)
        self.width = width
        self.height = height
        self.fov = fov
        self.mode = mode

    def update(self, sim_state: dict[str, any]) -> None:
        """Update camera data from simulation."""
        # TODO: Implement actual camera rendering from backend
        self._data = {
            "rgb": np.zeros((self.height, self.width, 3), dtype=np.uint8) if "rgb" in self.mode else None,
            "depth": np.zeros((self.height, self.width), dtype=np.float32) if "depth" in self.mode else None,
            "segmentation": None,
            "timestamp": sim_state.get("timestamp", 0.0),
        }

    def get_data(self) -> dict[str, np.ndarray]:
        """Get camera data."""
        return self._data if self._data is not None else {}

    def get_intrinsics(self) -> np.ndarray:
        """Get camera intrinsic matrix.

        Returns:
            3x3 intrinsic matrix
        """
        f = self.width / (2 * np.tan(np.radians(self.fov) / 2))
        cx = self.width / 2
        cy = self.height / 2

        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


class IMU(BaseSensor):
    """IMU sensor for acceleration and angular velocity."""

    def __init__(self, name: str = "imu", update_freq: float = 100.0) -> None:
        """Initialize IMU.

        Args:
            name: IMU name
            update_freq: Update frequency in Hz
        """
        super().__init__(name, update_freq)

    def update(self, sim_state: dict[str, any]) -> None:
        """Update IMU data from simulation."""
        # TODO: Implement actual IMU data extraction from backend
        self._data = {
            "linear_acceleration": np.zeros(3),  # [ax, ay, az]
            "angular_velocity": np.zeros(3),  # [wx, wy, wz]
            "orientation": np.array([0, 0, 0, 1]),  # quaternion
            "timestamp": sim_state.get("timestamp", 0.0),
        }

    def get_data(self) -> dict[str, np.ndarray]:
        """Get IMU data."""
        return self._data if self._data is not None else {}


class ContactSensor(BaseSensor):
    """Contact/force sensor."""

    def __init__(self, name: str = "contact", body_names: list | None = None, update_freq: float = 100.0) -> None:
        """Initialize contact sensor.

        Args:
            name: Sensor name
            body_names: list of body names to monitor
            update_freq: Update frequency in Hz
        """
        super().__init__(name, update_freq)
        self.body_names = body_names or []

    def update(self, sim_state: dict[str, any]) -> None:
        """Update contact data from simulation."""
        # TODO: Implement actual contact force extraction
        self._data = {
            "forces": np.zeros(len(self.body_names)),
            "in_contact": np.zeros(len(self.body_names), dtype=bool),
            "timestamp": sim_state.get("timestamp", 0.0),
        }

    def get_data(self) -> dict[str, np.ndarray]:
        """Get contact data."""
        return self._data if self._data is not None else {}


class SensorManager:
    """Manager for multiple sensors."""

    def __init__(self) -> None:
        """Initialize sensor manager."""
        self.sensors: dict[str, BaseSensor] = {}

    def add_sensor(self, sensor: BaseSensor) -> None:
        """Add a sensor.

        Args:
            sensor: Sensor instance
        """
        self.sensors[sensor.name] = sensor

    def update_all(self, sim_state: dict[str, any]) -> None:
        """Update all sensors.

        Args:
            sim_state: Current simulation state
        """
        for sensor in self.sensors.values():
            sensor.update(sim_state)

    def get_sensor_data(self, name: str) -> any:
        """Get data from specific sensor.

        Args:
            name: Sensor name

        Returns:
            Sensor data
        """
        if name in self.sensors:
            return self.sensors[name].get_data()
        return None

    def get_all_data(self) -> dict[str, any]:
        """Get data from all sensors.

        Returns:
            dictionary mapping sensor names to their data
        """
        return {name: sensor.get_data() for name, sensor in self.sensors.items()}
