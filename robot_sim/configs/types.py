from abc import ABC
from dataclasses import dataclass, field, fields, replace
from typing import Any

import msgpack
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

    def to_bytes(self) -> bytes:
        """Serialize the state into msgpack bytes with numpy buffers."""
        payload = _pack_value(self)
        return msgpack.packb(payload, use_bin_type=True)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ObjectState":
        """Deserialize msgpack bytes into an ObjectState."""
        data = msgpack.unpackb(payload, raw=False)
        state = _unpack_value(data)
        if not isinstance(state, ObjectState):
            raise ValueError("Payload does not contain an ObjectState.")
        return state

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
    body_state: ArrayType | None = None
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13)."""
    joint_pos: ArrayType | None = None
    """Joint positions. Shape is (num_envs, num_joints). If it is None, the object has no joints."""
    joint_vel: ArrayType | None = None
    """Joint velocities. Shape is (num_envs, num_joints). If it is None, same as above."""
    joint_action: ArrayType | None = None
    """Joint actions. Shape is (num_envs, num_joints). If it is None, no action is applied."""
    sensors: dict[str, Any] | None = None
    """Sensor readings. Each sensor has shape (num_envs, sensor_dim)."""
    extras: dict[str, Any] | None = None
    """Extra information."""


StatesType = dict[str, ObjectState]
ActionsType = dict[str, ArrayType]


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


########################### Helper Functions ##########################
def _pack_array(value: np.ndarray) -> dict[str, Any]:
    if not value.flags["C_CONTIGUOUS"]:
        value = np.ascontiguousarray(value)
    return {
        "__ndarray__": True,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data": value.tobytes(),
    }


def _pack_state(state: ObjectState) -> dict[str, Any]:
    state_np = state.to_numpy()
    payload: dict[str, Any] = {}
    for f in fields(state_np):
        payload[f.name] = _pack_value(getattr(state_np, f.name))
    return payload


def _pack_value(value: Any) -> Any:
    if isinstance(value, ObjectState):
        return {"__object_state__": True, "data": _pack_state(value)}
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _pack_array(value)
    if isinstance(value, dict):
        return {k: _pack_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack_value(v) for v in value]
    return value


def _unpack_array(value: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(value["dtype"])
    data = value["data"]
    array = np.frombuffer(data, dtype=dtype)
    shape = value.get("shape", [])
    if shape:
        array = array.reshape(tuple(shape))
    return array.copy()


def _unpack_state(payload: dict[str, Any]) -> ObjectState:
    kwargs: dict[str, Any] = {}
    for f in fields(ObjectState):
        kwargs[f.name] = _unpack_value(payload.get(f.name))
    return ObjectState(**kwargs)


def _unpack_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__object_state__") is True:
            return _unpack_state(value.get("data", {}))
        if value.get("__ndarray__") is True:
            return _unpack_array(value)
        return {k: _unpack_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack_value(v) for v in value]
    return value


def states_to_bytes(states: StatesType) -> bytes:
    payload = {name: _pack_value(state) for name, state in states.items()}
    return msgpack.packb(payload, use_bin_type=True)


def bytes_to_states(payload: bytes) -> StatesType:
    data = msgpack.unpackb(payload, raw=False)
    states: StatesType = {}
    for name, value in data.items():
        states[name] = _unpack_value(value)
    return states


def actions_to_bytes(actions: ActionsType) -> bytes:
    payload = {name: _pack_value(action) for name, action in actions.items()}
    return msgpack.packb(payload, use_bin_type=True)


def bytes_to_actions(payload: bytes) -> ActionsType:
    data = msgpack.unpackb(payload, raw=False)
    actions: ActionsType = {}
    for name, value in data.items():
        decoded = _unpack_value(value)
        if not isinstance(decoded, np.ndarray):
            decoded = np.asarray(decoded, dtype=np.float32)
        actions[name] = decoded
    return actions
