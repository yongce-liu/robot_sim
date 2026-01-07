import threading
import time
from dataclasses import dataclass
from typing import Any

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
import msgpack
import numpy as np
import torch
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber

from robot_sim.backends.types import ActionsType, ObjectState, StatesType

from .base import BaseBridge, bridge_register


@dataclass
@annotate.final
@annotate.autoid("sequential")
class StatePacket(idl.IdlStruct, typename="robot_sim.msg.dds_.StatePacket"):
    tick: types.uint64
    stamp_ns: types.uint64
    payload: types.sequence[types.uint8]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class ActionPacket(idl.IdlStruct, typename="robot_sim.msg.dds_.ActionPacket"):
    tick: types.uint64
    stamp_ns: types.uint64
    payload: types.sequence[types.uint8]


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, ObjectState):
        obj = obj.to_numpy()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def states_to_payload(states: StatesType) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, state in states.items():
        if isinstance(state, ObjectState):
            state_np = state.to_numpy()
            payload[name] = {
                "root_state": _to_builtin(state_np.root_state),
                "body_state": _to_builtin(state_np.body_state),
                "joint_pos": _to_builtin(state_np.joint_pos),
                "joint_vel": _to_builtin(state_np.joint_vel),
                "joint_action": _to_builtin(state_np.joint_action),
                "sensors": _to_builtin(state_np.sensors),
                "extras": _to_builtin(state_np.extras),
            }
        else:
            payload[name] = _to_builtin(state)
    return payload


def payload_to_states(payload: dict[str, Any]) -> StatesType:
    states: StatesType = {}
    for name, data in payload.items():
        state = ObjectState(
            root_state=np.asarray(data["root_state"], dtype=np.float32),
            body_state=np.asarray(data["body_state"], dtype=np.float32),
            joint_pos=np.asarray(data["joint_pos"], dtype=np.float32) if data.get("joint_pos") is not None else None,
            joint_vel=np.asarray(data["joint_vel"], dtype=np.float32) if data.get("joint_vel") is not None else None,
            joint_action=(
                np.asarray(data["joint_action"], dtype=np.float32) if data.get("joint_action") is not None else None
            ),
            sensors=data.get("sensors", {}),
            extras=data.get("extras", {}),
        )
        states[name] = state
    return states


def actions_to_payload(actions: ActionsType) -> dict[str, Any]:
    return _to_builtin(actions)


def payload_to_actions(payload: dict[str, Any]) -> ActionsType:
    actions: ActionsType = {}
    for name, value in payload.items():
        actions[name] = np.asarray(value, dtype=np.float32)
    return actions


@bridge_register("dds")
class UnitreeDDSBridge(BaseBridge):
    """DDS bridge using Unitree SDK channel APIs."""

    def __init__(self, mode: str, state_topic: str, action_topic: str):
        super().__init__(mode=mode)
        self._state_topic = state_topic
        self._action_topic = action_topic
        self._mode = mode
        self._state_pub: ChannelPublisher | None = None
        self._action_pub: ChannelPublisher | None = None
        self._state_sub: ChannelSubscriber | None = None
        self._action_sub: ChannelSubscriber | None = None
        self._lock = threading.Lock()
        self._latest_state: tuple[int, int, dict[str, Any]] | None = None
        self._latest_action: tuple[int, int, dict[str, Any]] | None = None
        self._state_tick = 0
        self._action_tick = 0

    def launch(self) -> None:
        if self._mode in ("robot", "client"):
            self._state_pub = ChannelPublisher(self._state_topic, StatePacket)
            self._state_pub.Init()
            self._action_sub = ChannelSubscriber(self._action_topic, ActionPacket)
            self._action_sub.Init(self._on_action, 1)
        elif self._mode == "server":
            self._action_pub = ChannelPublisher(self._action_topic, ActionPacket)
            self._action_pub.Init()
            self._state_sub = ChannelSubscriber(self._state_topic, StatePacket)
            self._state_sub.Init(self._on_state, 1)
        else:
            raise ValueError(f"Unknown DDS bridge mode: {self._mode}")

    def bind(self) -> None:
        self.launch()

    def close(self) -> None:
        return None

    def _pack(self, payload: dict[str, Any]) -> list[int]:
        packed = msgpack.packb(payload, use_bin_type=True)
        return list(packed)

    def _unpack(self, payload: list[int]) -> dict[str, Any]:
        raw = bytes(payload)
        return msgpack.unpackb(raw, raw=False)

    def _on_state(self, msg: StatePacket) -> None:
        payload = self._unpack(list(msg.payload))
        with self._lock:
            self._latest_state = (int(msg.tick), int(msg.stamp_ns), payload)

    def _on_action(self, msg: ActionPacket) -> None:
        payload = self._unpack(list(msg.payload))
        with self._lock:
            self._latest_action = (int(msg.tick), int(msg.stamp_ns), payload)

    def publish_state(self, states: StatesType, tick: int, stamp_ns: int) -> None:
        if self._state_pub is None:
            return
        payload = states_to_payload(states)
        msg = StatePacket(tick=tick, stamp_ns=stamp_ns, payload=self._pack(payload))
        self._state_pub.Write(msg)

    def publish_action(self, actions: ActionsType, tick: int, stamp_ns: int) -> None:
        if self._action_pub is None:
            return
        payload = actions_to_payload(actions)
        msg = ActionPacket(tick=tick, stamp_ns=stamp_ns, payload=self._pack(payload))
        self._action_pub.Write(msg)

    def get_latest_state(self) -> tuple[int, int, dict[str, Any]] | None:
        with self._lock:
            return self._latest_state

    def get_latest_action(self) -> tuple[int, int, dict[str, Any]] | None:
        with self._lock:
            return self._latest_action

    def send_state(
        self,
        topic: str,
        state: StatesType,
        tick: int | None = None,
        stamp_ns: int | None = None,
    ) -> None:
        if tick is None:
            self._state_tick += 1
            tick = self._state_tick
        if stamp_ns is None:
            stamp_ns = time.monotonic_ns()
        self.publish_state(state, tick=tick, stamp_ns=stamp_ns)

    def get_state(self, topic: str) -> tuple[int, int, StatesType] | None:
        msg = self.get_latest_state()
        if msg is None:
            return None
        tick, stamp_ns, payload = msg
        return tick, stamp_ns, payload_to_states(payload)

    def send_action(
        self,
        topic: str,
        action: ActionsType,
        tick: int | None = None,
        stamp_ns: int | None = None,
    ) -> None:
        if tick is None:
            self._action_tick += 1
            tick = self._action_tick
        if stamp_ns is None:
            stamp_ns = time.monotonic_ns()
        self.publish_action(action, tick=tick, stamp_ns=stamp_ns)

    def get_action(self, topic: str) -> tuple[int, int, ActionsType] | None:
        msg = self.get_latest_action()
        if msg is None:
            return None
        tick, stamp_ns, payload = msg
        return tick, stamp_ns, payload_to_actions(payload)
