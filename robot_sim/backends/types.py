from abc import ABC
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
import torch

########################## Data Structures ##########################

ArrayType = torch.Tensor | np.ndarray


@dataclass
class BaseState(ABC):
    """Base class for states."""

    def to_numpy(self) -> "BaseState":
        """Convert all tensor attributes to numpy arrays recursively."""
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = self._convert_to_numpy(value)
        return replace(self, **kwargs)

    def to_tensor(self, device: str | torch.device = "cpu") -> "BaseState":
        """Convert all numpy array attributes to tensors recursively."""

        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = self._convert_to_tensor(value, device)
        return replace(self, **kwargs)

    def clone(self) -> "BaseState":
        """Create a deep copy of the state."""
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = self._deep_clone(value)
        return replace(self, **kwargs)

    @staticmethod
    def _deep_clone(value: Any) -> Any:
        """Recursively clone a value."""
        if isinstance(value, BaseState):
            return value.clone()
        elif isinstance(value, torch.Tensor):
            return value.clone()
        elif isinstance(value, np.ndarray):
            return value.copy()
        elif isinstance(value, dict):
            return {k: BaseState._deep_clone(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            converted = [BaseState._deep_clone(v) for v in value]
            return type(value)(converted)
        else:
            return value

    @staticmethod
    def _convert_to_numpy(value: Any) -> Any:
        """Recursively convert value to numpy."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        elif isinstance(value, dict):
            return {k: BaseState._convert_to_numpy(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            converted = [BaseState._convert_to_numpy(v) for v in value]
            return type(value)(converted)
        else:
            return value

    @staticmethod
    def _convert_to_tensor(value: Any, device: str | torch.device = "cpu") -> Any:
        """Recursively convert value to tensor."""
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device)
        elif isinstance(value, dict):
            return {k: BaseState._convert_to_tensor(v, device) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            converted = [BaseState._convert_to_tensor(v, device) for v in value]
            return type(value)(converted)
        else:
            return value


@dataclass
class ObjectState(BaseState):
    """State of a single robot/object."""

    root_state: ArrayType
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_state: ArrayType
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13)."""
    joint_pos: ArrayType | None = None
    """Joint positions. Shape is (num_envs, num_joints). If it is None, the object has no joints."""
    joint_vel: ArrayType | None = None
    """Joint velocities. Shape is (num_envs, num_joints). If it is None, same as above."""
    # joint_pos_target: ArrayType | None = None
    # """Joint positions target. Shape is (num_envs, num_joints)."""
    # joint_vel_target: ArrayType | None = None
    # """Joint velocities target. Shape is (num_envs, num_joints)."""
    # joint_effort_target: ArrayType | None = None
    # """Joint effort targets. Shape is (num_envs, num_joints)."""
    joint_action: ArrayType | None = None
    """Joint actions. Shape is (num_envs, num_joints). If it is None, no action is applied."""
    sensors: dict[str, Any] = field(default_factory=dict)
    """Sensor readings. Each sensor has shape (num_envs, sensor_dim)."""
    extras: dict[str, Any] = field(default_factory=dict)
    """Extra information."""


StatesType = dict[str, ObjectState]
ActionsType = dict[str, ArrayType]


# @dataclass
# class StatesType(BaseState):
#     """A dictionary that holds the states of all robots and objects in tensor format.

#     The keys are the names of the robots and objects, and the values are tensors representing their states.
#     The tensor shape is (num_envs, state_dim), where num_envs is the number of environments, and state_dim is the dimension of the state for each robot or object.
#     """

#     objects: dict[str, ObjectState]
#     extras: dict[str, Any] = field(default_factory=dict)


# robot/object name
@dataclass
class Buffer:
    # config: ObjectConfig | None = None
    # sensors: dict[str, BaseSensor] | None = field(default_factory=dict)  # buffer -> Sensor Instance
    # the joint order follows the config.joints order
    joint_names: list[str] | None = None  # buffer -> list[joint name]
    default_joint_positions: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )  # buffer -> (num_joints,)
    # the body order follows the urdf/sdf/model file order if not specified in the config
    body_names: list[str] | None = None  # buffer -> list[body name]
    actuator_names: list[str] | None = None  # buffer -> list[actuator name]
    # source: backend order, target: config order
    joint_indices: list[int] | None = None  # source -> target
    joint_indices_reverse: list[int] | None = None  # target -> source
    body_indices: list[int] | None = None  # source -> target
    body_indices_reverse: list[int] | None = None  # target -> source
    action_indices: list[int] | None = None  # source -> target
    action_indices_reverse: list[int] | None = None  # target -> source
