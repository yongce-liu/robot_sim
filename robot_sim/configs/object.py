from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import numpy as np
import regex as re
from loguru import logger

from .sensor import SensorConfig


class ObjectType(Enum):
    """Enumeration of available object types in the simulation."""

    ROBOT = "robot"
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CUSTOM = "custom"


class ControlType(Enum):
    """Enumeration of available control types for robot actuators."""

    POSITION = "position"
    TORQUE = "torque"
    VELOCITY = "velocity"


@dataclass
class JointConfig:
    """Configuration for a single joint (actuator) in the robot."""

    torque_limit: float
    """Maximum torque of the actuator."""
    velocity_limit: float
    """Maximum velocity of the actuator."""
    position_limit: list[float]
    """Position limits of the actuator [min, max]."""
    control_type: ControlType | None
    """Control type of the actuator."""
    default_position: float = 0.0
    """Default position of the actuator."""
    stiffness: float | None = None
    """Stiffness of the actuator."""
    damping: float | None = None
    """Damping of the actuator."""
    actuated: bool = True
    """Whether the actuator is can be actuated."""
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional properties specific to the joint."""

    def __post_init__(self):
        if ControlType(self.control_type) in [ControlType.POSITION]:
            assert self.stiffness is not None, "Stiffness must be defined for position control."
            assert self.damping is not None, "Damping must be defined for position control."


@dataclass
class ObjectConfig:
    # name: str
    # """Name of the robot."""
    type: ObjectType = ObjectType.CUSTOM
    """Type of the object. Can be a ObjectType or 'custom'."""
    path: str | None = None
    """Path to the robot's model file."""
    pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    """Initial pose of the robot/object in the simulation [x, y, z, qw, qx, qy, qz]."""
    twist: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    """Initial twist (linear and angular velocity) of the robot/object in the simulation [vx, vy, vz, wx, wy, wz]."""
    joints: dict[str, JointConfig] | None = None
    """List of actuators (joints) in the robot/object. If it is None, no actuators are defined."""
    bodies: dict[str, Any] | None = None
    """List of bodies in the robot/object. If it is None, no bodies are defined."""
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    """Sensor configurations for the robot."""
    properties: dict[str, Any] = field(default_factory=dict)
    """Additional properties specific to the robot."""
    extras: dict[str, Any] = field(default_factory=dict)
    """Extra information for custom use cases."""

    def __post_init__(self):
        if self.path is None and self.type in [ObjectType.CUSTOM, ObjectType.ROBOT]:
            raise ValueError("For custom object type, a valid path to the model file must be provided.")
        assert len(self.pose) == 7, "Pose must be a list of 7 elements [x, y, z, qw, qx, qy, qz]."
        assert len(self.twist) == 6, "Twist must be a list of 6 elements [vx, vy, vz, wx, wy, wz]."


class ObjectModel:
    def __init__(self, config: ObjectConfig):
        self.cfg = config
        self._cache: dict[str, Any] = {}
        self.initialize()

    def initialize(self) -> None:
        """Initialize the robot model."""
        _ = self.num_dofs
        _ = self.default_joint_positions
        _ = self.joint_names
        _ = self.actuator_names
        _ = self.actuator_indices
        _ = self.stiffness
        _ = self.damping
        for t in ControlType:
            _ = self.get_joint_limits(t)

    @property
    def cfg_sensors(self) -> dict[str, SensorConfig]:
        return self.cfg.sensors

    @property
    def num_dofs(self) -> int:
        if "num_dofs" in self._cache:
            return cast(int, self._cache["num_dofs"])
        self._cache["num_dofs"] = num_dofs = len(self.cfg.joints.values()) if self.cfg.joints else 0
        return num_dofs

    @property
    def default_joint_positions(self):
        return None

    @property
    def joint_names(self):
        return None

    @property
    def actuator_names(self):
        return None

    @property
    def actuator_indices(self):
        return None

    @property
    def stiffness(self):
        return None

    @property
    def damping(self):
        return None

    @property
    def sensor_names(self):
        return None

    def get_joint_limits(self, *args, **kwargs):
        return None

    def get_group_joint_indices(self, *args, **kwargs):
        return None

    def get_joint_names(self, *args, **kwargs):
        return None

    def get_actuator_names(self, *args, **kwargs):
        return None

    @property
    def body_names(self) -> list[str] | None:
        return self._cache.get("body_names")

    @body_names.setter
    def body_names(self, value: list[str]) -> None:
        self._cache["body_names"] = value

    def get_body_names(self, prefix: str | None = None) -> list[str]:
        """Get the names of all bodies."""
        if prefix is None:
            return cast(list[str], self._cache["body_names"])
        hashed_key = f"body_names_with_prefix_{prefix}"
        if hashed_key in self._cache:
            return cast(list[str], self._cache[hashed_key])
        self._cache[hashed_key] = names = [f"{prefix}{name}" for name in cast(list[str], self._cache["body_names"])]
        return names


class RobotModel(ObjectModel):
    def __init__(self, config: ObjectConfig):
        assert config.type == ObjectType.ROBOT, "RobotModel must be initialized with a robot ObjectConfig."
        super().__init__(config=config)

    @property
    def default_joint_positions(self) -> np.ndarray:
        assert self.cfg.joints is not None, "Robot configuration must have joints defined."
        if "default_joint_positions" in self._cache:
            return cast(np.ndarray, self._cache["default_joint_positions"])
        self._cache["default_joint_positions"] = default_positions = np.array(
            [obj.default_position for obj in self.cfg.joints.values() if obj.actuated], dtype=np.float32
        )
        return default_positions

    @property
    def joint_names(self) -> list[str]:
        assert self.cfg.joints is not None, "Robot configuration must have joints defined."
        if "joint_names" in self._cache:
            return cast(list[str], self._cache["joint_names"])
        self._cache["joint_names"] = names = list(self.cfg.joints.keys())
        return names

    @joint_names.setter
    def joint_names(self, value: list[str]) -> None:
        self._cache["joint_names"] = value

    @property
    def actuator_names(self) -> list[str]:
        assert self.cfg.joints is not None, "Robot configuration must have joints defined."
        if "actuator_names" in self._cache:
            return cast(list[str], self._cache["actuator_names"])
        self._cache["actuator_names"] = names = [name for name, obj in self.cfg.joints.items() if obj.actuated]
        return names

    @actuator_names.setter
    def actuator_names(self, value: list[str]) -> None:
        self._cache["actuator_names"] = value

    def get_actuator_names(self, prefix: str | None = None) -> list[str]:
        """Get the names of actuated joints."""
        if prefix is None:
            return self.actuator_names
        hashed_key = f"actuator_names_with_prefix_{prefix}"
        if hashed_key in self._cache:
            return cast(list[str], self._cache[hashed_key])
        self._cache[hashed_key] = names = [f"{prefix}{name}" for name in self.actuator_names]
        return names

    def get_joint_names(self, prefix: str | None = None) -> list[str]:
        """Get the names of all joints."""
        if prefix is None:
            return self.joint_names
        hashed_key = f"joint_names_with_prefix_{prefix}"
        if hashed_key in self._cache:
            return cast(list[str], self._cache[hashed_key])
        self._cache[hashed_key] = names = [f"{prefix}{name}" for name in self.joint_names]
        return names

    @property
    def actuator_indices(self) -> np.ndarray:
        assert self.cfg.joints is not None, "Robot configuration must have joints defined."
        if "actuator_indices" in self._cache:
            return cast(np.ndarray, self._cache["actuator_indices"])
        self._cache["actuator_indices"] = indices = np.array(
            [i for i, obj in enumerate(self.cfg.joints.values()) if obj.actuated], dtype=np.int32
        )
        return indices

    @property
    def stiffness(self) -> np.ndarray:
        assert self.cfg.joints is not None, "Robot configuration must have joints defined."
        if "stiffness" in self._cache:
            return cast(np.ndarray, self._cache["stiffness"])
        self._cache["stiffness"] = stiffness = np.array(
            [obj.stiffness if obj.stiffness is not None else 0.0 for obj in self.cfg.joints.values() if obj.actuated],
            dtype=np.float32,
        )
        return stiffness

    @property
    def damping(self) -> np.ndarray:
        assert self.cfg.joints is not None, "Robot configuration must have joints defined."
        if "damping" in self._cache:
            return cast(np.ndarray, self._cache["damping"])
        self._cache["damping"] = damping = np.array(
            [obj.damping if obj.damping is not None else 0.0 for obj in self.cfg.joints.values() if obj.actuated],
            dtype=np.float32,
        )
        return damping

    def get_joint_limits(self, key: ControlType | str, coeff=1.0) -> tuple[np.ndarray, np.ndarray]:
        """Get joint limits for the specified control type."""
        key = ControlType(key)
        hashed_key = f"joint_limits/{key.value}/{coeff}"
        if hashed_key in self._cache:
            return cast(tuple[np.ndarray, np.ndarray], self._cache[hashed_key])
        assert self.cfg.joints is not None, "Robot configuration must have joints defined."
        joints: dict[str, JointConfig] = self.cfg.joints
        if key == ControlType.TORQUE:
            tmp = np.array([obj.torque_limit * coeff for obj in joints.values() if obj.actuated], dtype=np.float32)
            self._cache[hashed_key] = (-tmp, tmp)
        elif key == ControlType.POSITION:
            position_limits = np.array(
                [obj.position_limit for obj in joints.values() if obj.actuated], dtype=np.float32
            )
            mid = (position_limits[:, 0] + position_limits[:, 1]) / 2
            range_2 = (position_limits[:, 1] - position_limits[:, 0]) * 0.5
            position_limits[:, 0] = mid - coeff * range_2
            position_limits[:, 1] = mid + coeff * range_2
            self._cache[hashed_key] = (position_limits[:, 0], position_limits[:, 1])
        elif key == ControlType.VELOCITY:
            velocity_limits = np.array(
                [obj.velocity_limit for obj in joints.values() if obj.actuated], dtype=np.float32
            )
            self._cache[hashed_key] = (-velocity_limits * coeff, velocity_limits * coeff)

        return cast(tuple[np.ndarray, np.ndarray], self._cache[hashed_key])

    def get_group_joint_indices(self, group_name: str, patterns: str | list[str] | None = None) -> np.ndarray:
        """Get joint indices for a specific group."""
        hased_key = f"group_joint_indices/{group_name}"
        if hased_key in self._cache:
            return cast(np.ndarray, self._cache[hased_key])
        assert self.cfg.joints is not None, "Robot configuration must have joints defined."
        if patterns is None:
            patterns = [group_name]
        elif isinstance(patterns, str):
            patterns = [patterns]

        compiled = [re.compile(p) for p in patterns]
        indices_list: list[int] = []
        for i, name in enumerate(self.cfg.joints.keys()):
            if any(rx.search(name) for rx in compiled):
                indices_list.append(i)

        self._cache[hased_key] = ans = np.array(indices_list, dtype=np.int32)
        if len(ans) == 0:
            logger.error(f"No joints found for group '{group_name}' with patterns {patterns}.")
        else:
            logger.info(f"Found joints indices: {ans} for group '{group_name}' with patterns {patterns}.")

        return ans
