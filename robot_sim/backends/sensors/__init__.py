from .base import BaseSensor
from .camera import Camera
from .contact_force import ContactForce

__all__ = ["BaseSensor", "Camera", "ContactForce"]


from robot_sim.configs import SensorType

# Registry to map sensor type to concrete class
_SENSOR_TYPE_REGISTRY: dict[SensorType, type] = {
    SensorType.CAMERA: Camera,
    SensorType.CONTACT_FORCE: ContactForce,
}
