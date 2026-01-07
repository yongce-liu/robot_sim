import threading
import time
from typing import Any

import numpy as np
from loguru import logger

from robot_sim.backends.base import BaseBackend
from robot_sim.backends.types import ActionsType, ArrayType, ObjectState, StatesType
from robot_sim.bridges.unitree import UnitreeDDSBridge, payload_to_actions
from robot_sim.configs import SimulatorConfig


def _load_unitree_types(robot_type: str):
    if "g1" in robot_type or "h1-2" in robot_type:
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

        low_cmd = unitree_hg_msg_dds__LowCmd_()
        low_state = LowState_default()
        return low_cmd, low_state, LowCmd_, LowState_
    if "h1" == robot_type or "go2" == robot_type:
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

        low_cmd = unitree_go_msg_dds__LowCmd_()
        low_state = LowState_default()
        return low_cmd, low_state, LowCmd_, LowState_
    raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'.")


class UnitreeBackend(BaseBackend):
    """Unitree hardware backend using DDS for command/state."""

    def __init__(self, config: SimulatorConfig, optional_queries: dict[str, Any] | None = None):
        super().__init__(config, optional_queries)
        assert self.num_envs == 1, f"Unitree backend supports a single env, got {self.num_envs}."

        spec = self.sim_config.backend_spec.get("unitree", {})
        self._robot_type = spec.get("robot_type", "g1")
        self._action_mode = spec.get("action_mode", "position")
        self._action_topic = spec.get("action_topic", "rt/lowcmd")
        self._state_topic = spec.get("state_topic", "rt/lowstate")
        dds_cfg = spec.get("dds", {})
        self._dds_enable = bool(dds_cfg.get("enable", False))
        self._dds_state_topic = dds_cfg.get("state_topic", "rt/robot_sim/state")
        self._dds_action_topic = dds_cfg.get("action_topic", "rt/robot_sim/action")

        self._motor_kp = np.asarray(spec.get("motor_kp", []), dtype=np.float32)
        self._motor_kd = np.asarray(spec.get("motor_kd", []), dtype=np.float32)

        self._robot_name = spec.get("robot_name")
        if self._robot_name is None:
            if len(self.objects) != 1:
                logger.warning("Unitree backend expects one robot; using the first object as robot.")
            self._robot_name = list(self.objects.keys())[0]

        self._state_lock = threading.Lock()
        self._have_state = False
        self._pending_action: ActionsType | None = None

        self._low_cmd, self._low_state, self._low_cmd_type, self._low_state_type = _load_unitree_types(self._robot_type)
        self._low_cmd_puber = None
        self._low_state_suber = None
        self._dds_bridge: UnitreeDDSBridge | None = None
        self._dds_tick = 0

    def _launch(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber

        self._low_cmd_puber = ChannelPublisher(self._action_topic, self._low_cmd_type)
        self._low_cmd_puber.Init()
        self._low_state_suber = ChannelSubscriber(self._state_topic, self._low_state_type)
        self._low_state_suber.Init(self._on_low_state, 1)
        if self._dds_enable:
            self._dds_bridge = UnitreeDDSBridge(
                state_topic=self._dds_state_topic,
                action_topic=self._dds_action_topic,
                mode="robot",
            )
            self._dds_bridge.bind()

    def _render(self) -> None:
        return None

    def get_rgb_image(self) -> np.ndarray | Any | None:
        return None

    def close(self) -> None:
        return None

    def _on_low_state(self, msg) -> None:
        with self._state_lock:
            self._low_state = msg
            self._have_state = True

    def _simulate(self):
        if self._dds_bridge is not None:
            msg = self._dds_bridge.get_latest_action()
            if msg is not None:
                _, _, payload = msg
                self._pending_action = payload_to_actions(payload)

        if self._pending_action is None or self._low_cmd_puber is None:
            return

        actions = self._pending_action.get(self._robot_name)
        if actions is None:
            return
        actions = np.asarray(actions).reshape(-1)

        num_joints = len(self.get_joint_names(self._robot_name))
        if actions.shape[0] < num_joints:
            padded = np.zeros((num_joints,), dtype=np.float32)
            padded[: actions.shape[0]] = actions
            actions = padded

        for i in range(num_joints):
            cmd = self._low_cmd.motor_cmd[i]
            if i < self._motor_kp.shape[0]:
                cmd.kp = float(self._motor_kp[i])
            if i < self._motor_kd.shape[0]:
                cmd.kd = float(self._motor_kd[i])

            if self._action_mode == "torque":
                cmd.tau = float(actions[i])
                cmd.q = 0.0
                cmd.dq = 0.0
            elif self._action_mode == "velocity":
                cmd.dq = float(actions[i])
                cmd.q = 0.0
                cmd.tau = 0.0
            else:
                cmd.q = float(actions[i])
                cmd.dq = 0.0
                cmd.tau = 0.0

        self._low_cmd_puber.Write(self._low_cmd)
        if self._dds_bridge is not None:
            self._dds_tick += 1
            self._dds_bridge.publish_state(self.get_states(), self._dds_tick, time.monotonic_ns())

    def _set_states(self, states: StatesType, env_ids: ArrayType | None = None) -> None:
        logger.warning("Unitree backend does not support setting states on hardware.")

    def _set_actions(self, actions: ActionsType) -> None:
        self._pending_action = actions

    def _get_states(self) -> StatesType:
        num_joints = len(self.get_joint_names(self._robot_name))
        num_bodies = len(self.get_body_names(self._robot_name))

        joint_pos = np.zeros((1, num_joints), dtype=np.float32)
        joint_vel = np.zeros((1, num_joints), dtype=np.float32)
        root_state = np.zeros((1, 13), dtype=np.float32)
        body_state = np.zeros((1, num_bodies, 13), dtype=np.float32)

        with self._state_lock:
            low_state = self._low_state
            have_state = self._have_state

        if have_state:
            for i in range(num_joints):
                joint_pos[0, i] = low_state.motor_state[i].q
                joint_vel[0, i] = low_state.motor_state[i].dq

            root_state[0, 3:7] = np.asarray(low_state.imu_state.quaternion, dtype=np.float32)
            root_state[0, 10:13] = np.asarray(low_state.imu_state.gyroscope, dtype=np.float32)

        state = ObjectState(
            root_state=root_state,
            body_state=body_state,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_action=None,
            sensors={},
        )
        return {self._robot_name: state}

    def _update_buffer_indices(self) -> None:
        for obj_name, buf in self._buffer_dict.items():
            joint_count = len(buf.joint_names) if buf.joint_names else 0
            body_count = len(buf.body_names) if buf.body_names else 0
            action_count = len(buf.actuator_names) if buf.actuator_names else 0

            buf.joint_indices = list(range(joint_count))
            buf.joint_indices_reverse = list(range(joint_count))
            buf.body_indices = list(range(body_count))
            buf.body_indices_reverse = list(range(body_count))
            buf.action_indices = list(range(action_count))
            buf.action_indices_reverse = list(range(action_count))
