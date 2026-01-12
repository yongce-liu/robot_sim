import threading
import time
from copy import deepcopy
from typing import Any, Callable, TypeAlias

import numpy as np
from cyclonedds.idl import IdlStruct
from loguru import logger
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core import channel
from unitree_sdk2py.idl import default as unitree_default
from unitree_sdk2py.idl import unitree_go, unitree_hg
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from robot_sim.backends.base import BaseBackend
from robot_sim.configs import SimulatorConfig
from robot_sim.configs.types import ActionsType, ArrayType, ObjectState, StatesType
from robot_sim.controllers import CompositeController

TopicSpec: TypeAlias = tuple[
    type[IdlStruct],
    type[IdlStruct],
    Callable[[], Any],
    Callable[[], Any],
    list[str],
]


def get_default_map(robot_type: str | None) -> dict[str, TopicSpec]:
    _UNITREE_HG_LOW_PAIR = (
        unitree_hg.msg.dds_.LowState_,
        unitree_hg.msg.dds_.LowCmd_,
        unitree_default.unitree_hg_msg_dds__LowState_,
        unitree_default.unitree_hg_msg_dds__LowCmd_,
    )
    _UNITREE_HG_HAND_PAIR = (
        unitree_hg.msg.dds_.HandState_,
        unitree_hg.msg.dds_.HandCmd_,
        unitree_default.unitree_hg_msg_dds__HandState_,
        unitree_default.unitree_hg_msg_dds__HandCmd_,
    )
    _UNITREE_GO_PAIR = (
        unitree_go.msg.dds_.LowState_,
        unitree_go.msg.dds_.LowCmd_,
        unitree_default.unitree_go_msg_dds__LowState_,
        unitree_default.unitree_go_msg_dds__LowCmd_,
    )
    _DEFAULT_TOPIC_MSG_JOINT_MAP: dict[str, dict[str, TopicSpec]] = {
        "g1_dex3": {
            "rt/low": (
                *_UNITREE_HG_LOW_PAIR,
                [
                    "left_hip_pitch",
                    "left_hip_roll",
                    "left_hip_yaw",
                    "left_knee",
                    "left_ankle_pitch",
                    "left_ankle_roll",
                    "right_hip_pitch",
                    "right_hip_roll",
                    "right_hip_yaw",
                    "right_knee",
                    "right_ankle_pitch",
                    "right_ankle_roll",
                    "waist_yaw",
                    "waist_roll",
                    "waist_pitch",
                    "left_shoulder_pitch",
                    "left_shoulder_roll",
                    "left_shoulder_yaw",
                    "left_elbow",
                    "left_wrist_roll",
                    "left_wrist_pitch",
                    "left_wrist_yaw",
                    "right_shoulder_pitch",
                    "right_shoulder_roll",
                    "right_shoulder_yaw",
                    "right_elbow",
                    "right_wrist_roll",
                    "right_wrist_pitch",
                    "right_wrist_yaw",
                ],
            ),
            "rt/dex3/left/": (
                *_UNITREE_HG_HAND_PAIR,
                [
                    "left_hand_thumb_0",
                    "left_hand_thumb_1",
                    "left_hand_thumb_2",
                    "left_hand_middle_0",
                    "left_hand_middle_1",
                    "left_hand_index_0",
                    "left_hand_index_1",
                ],
            ),
            "rt/dex3/right/": (
                *_UNITREE_HG_HAND_PAIR,
                [
                    "right_hand_thumb_0",
                    "right_hand_thumb_1",
                    "right_hand_thumb_2",
                    "right_hand_middle_0",
                    "right_hand_middle_1",
                    "right_hand_index_0",
                    "right_hand_index_1",
                ],
            ),
        }
    }

    assert robot_type in _DEFAULT_TOPIC_MSG_JOINT_MAP, f"Unsupported robot type '{robot_type}' for UnitreeBackend."

    return _DEFAULT_TOPIC_MSG_JOINT_MAP[robot_type]


class UnitreeRemoteController:
    def __init__(self) -> None:
        self.Lx = 0.0
        self.Rx = 0.0
        self.Ry = 0.0
        self.Ly = 0.0

        self.L1 = 0
        self.L2 = 0
        self.R1 = 0
        self.R2 = 0
        self.A = 0
        self.B = 0
        self.X = 0
        self.Y = 0
        self.Up = 0
        self.Down = 0
        self.Left = 0
        self.Right = 0
        self.Select = 0
        self.F1 = 0
        self.F3 = 0
        self.Start = 0

    def _parse_buttons(self, data1: int, data2: int) -> None:
        self.R1 = (data1 >> 0) & 1
        self.L1 = (data1 >> 1) & 1
        self.Start = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2 = (data1 >> 4) & 1
        self.L2 = (data1 >> 5) & 1
        self.F1 = (data1 >> 6) & 1
        self.F3 = (data1 >> 7) & 1
        self.A = (data2 >> 0) & 1
        self.B = (data2 >> 1) & 1
        self.X = (data2 >> 2) & 1
        self.Y = (data2 >> 3) & 1
        self.Up = (data2 >> 4) & 1
        self.Right = (data2 >> 5) & 1
        self.Down = (data2 >> 6) & 1
        self.Left = (data2 >> 7) & 1

    def _parse_axes(self, data: bytes) -> None:
        self.Lx = float(np.frombuffer(data, dtype=np.float32, count=1, offset=4)[0])
        self.Rx = float(np.frombuffer(data, dtype=np.float32, count=1, offset=8)[0])
        self.Ry = float(np.frombuffer(data, dtype=np.float32, count=1, offset=12)[0])
        self.Ly = float(np.frombuffer(data, dtype=np.float32, count=1, offset=20)[0])

    def parse(self, remote_data: Any) -> None:
        data = bytes(remote_data)
        if len(data) < 24:
            raise ValueError(f"Remote data too short: {len(data)} bytes.")
        self._parse_axes(data)
        self._parse_buttons(data[2], data[3])

    def get_dict(self) -> dict[str, float | int]:
        return {
            "Lx": self.Lx,
            "Rx": self.Rx,
            "Ry": self.Ry,
            "Ly": self.Ly,
            "L1": self.L1,
            "L2": self.L2,
            "R1": self.R1,
            "R2": self.R2,
            "A": self.A,
            "B": self.B,
            "X": self.X,
            "Y": self.Y,
            "Up": self.Up,
            "Down": self.Down,
            "Left": self.Left,
            "Right": self.Right,
            "Select": self.Select,
            "F1": self.F1,
            "F3": self.F3,
            "Start": self.Start,
        }


class UnitreeLowLevelBackend(BaseBackend):
    def __init__(
        self, controllers: dict[str, CompositeController] | None = None, disable_controllers: bool = True, **kwargs
    ):
        logger.warning("UnitreeLowLevelBackend initialized; using low-level DDS control.")
        if disable_controllers and (controllers is not None and len(controllers) > 0):
            raise ValueError(
                f"UnitreeBackend only support position control. But you provided controllers {controllers} for low-level/high-frequency control. If you want use the provided controllers, please set disable_controllers=False. If you want to disable the controllers, please pass controllers=None."
            )
        super().__init__(controllers=controllers, **kwargs)
        assert self.num_envs == 1, f"UnitreeBackend only supports num_envs=1, got {self.num_envs}."
        assert len(self.robots) == 1, "UnitreeBackend only support single robot simulation."
        robot_type = self.cfg_spec.get("robot_type")
        self._robot_name = self.robot_names[0]
        self._robot = self.robots[self._robot_name]
        self._topic_joint_msg_map = get_default_map(robot_type)

        self._mode_pr = int(self.cfg_spec.get("mode_pr", 0))
        self._mode_machine = int(self.cfg_spec.get("mode_machine", 0))
        self._state_queue_len = int(self.cfg_spec.get("state_queue_len", 10))

        self._control_dt = self.cfg_sim.dt
        self._cmd_cnt = 0
        self._decimation = self.cfg_extras.get("decimation", 10)

        self._topic_joint_indices: dict[str, list[int]] = {}
        self._publishers: dict[str, channel.ChannelPublisher] = {}
        self._cmd_msgs: dict[str, Any] = {}
        self._subscribers: dict[str, channel.ChannelSubscriber] = {}
        self._state_msgs: dict[str, Any] = {}

        self._state_lock = threading.Lock()
        self._latest_states = self._build_dummy_state()
        self._latest_robot_state = deepcopy(self._latest_states[self._robot_name])

        self._latest_robot_action: np.ndarray = np.zeros((self.num_envs, self._robot.num_dofs), dtype=np.float32)
        self._latest_actions = {self._robot_name: self._latest_robot_action}
        self._action_lock = threading.Lock()

        self._state_ready = threading.Event()
        self._start_task = threading.Event()
        self._crc = CRC()

        self._remote_controller = UnitreeRemoteController()

        self._joystick_thread: RecurrentThread | None = None
        self._control_thread: RecurrentThread | None = None

        self._default_joint_positions = self._robot.default_joint_positions[np.newaxis, :].repeat(self.num_envs, axis=0)
        self._joint_kp, self._joint_kd = self._robot.stiffness, self._robot.damping
        self._pos_limits = self._robot.get_joint_limits("position")
        self._next_sim_time = None

        self._init_dds(**self.cfg_spec.get("dds", {}))

    def _init_dds(self, domain_id: int = 0, interface: str = "lo") -> None:
        channel.ChannelFactoryInitialize(domain_id, interface)
        for topic, values in self._topic_joint_msg_map.items():
            state_type, cmd_type, state_msg_cls, cmd_msg_cls, joint_names = values
            self._subscribers[topic] = channel.ChannelSubscriber(f"{topic}state", state_type)
            self._publishers[topic] = channel.ChannelPublisher(f"{topic}cmd", cmd_type)
            self._cmd_msgs[topic] = cmd_msg_cls()

            self._topic_joint_indices[topic] = self._robot.get_group_joint_indices(
                group_name=topic, patterns=joint_names
            )
            assert (
                len(self._topic_joint_indices[topic]) == len(joint_names) == len(set(self._topic_joint_indices[topic]))
            ), (
                f"Joint count mismatch for topic '{topic}': expected {joint_names}, got "
                f"{self._topic_joint_indices[topic]}."
            )

    def _launch(self) -> None:
        # release previous mode
        msc = MotionSwitcherClient()
        msc.SetTimeout(float(self.cfg_spec.get("motion_switcher_timeout", 5.0)))
        msc.Init()
        status, result = msc.CheckMode()
        while result and result.get("name"):
            logger.info(f"Releasing previous mode '{result.get('name')}'...")
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1.0)
        # Initialize the callbacks of subscribers and publishers
        for topic, subscriber in self._subscribers.items():
            joint_indices = self._topic_joint_indices[topic]
            subscriber.Init(
                lambda msg, topic=topic, indices=joint_indices: self._assign_robot_state(
                    topic=topic, msg=msg, indices=indices
                ),
                self._state_queue_len,
            )
        for publisher in self._publishers.values():
            publisher.Init()

        self._init_cmd_msgs(joint_targets=self._default_joint_positions)
        self._control_thread = RecurrentThread(interval=self._control_dt, target=self._write_cmd_loop, name="unitree")
        self._control_thread.Start()

        self._joystick_thread = RecurrentThread(
            interval=self._control_dt * 10, target=self._joystick_loop, name="unitree_joystick"
        )  # 10 * control dt = 10 * 0.002 = 0.02s
        self._joystick_thread.Start()

        while not self._state_ready.wait(timeout=2.0):
            logger.warning(
                "UnitreeBackend did not receive initial state in time; commands will be sent after state arrives."
            )

    def _render(self) -> None:
        # logger.error("UnitreeBackend does not support rendering.")
        return None

    def close(self) -> None:
        """Close the simulation."""
        if self._control_thread is not None:
            self._control_thread.Wait(1.0)
            self._control_thread = None
        if self._joystick_thread is not None:
            self._joystick_thread.Wait(1.0)
            self._joystick_thread = None

        for subscriber in self._subscribers.values():
            subscriber.Close()
        for publisher in self._publishers.values():
            publisher.Close()

    def _simulate(self):
        """Simulate the environment for one time step."""
        now = time.monotonic()
        if self._next_sim_time is None:
            self._next_sim_time = now + self._control_dt
            return

        sleep_time = self._next_sim_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._next_sim_time += self._control_dt

    def _set_states(self, states: StatesType, env_ids: ArrayType) -> None:
        """Set the states of the environment."""
        joint_pos = states[self._robot_name].joint_pos
        if joint_pos is not None:
            joint_pos = joint_pos.clip(self._pos_limits[0], self._pos_limits[1])
        else:
            joint_pos = self._default_joint_positions
        self.set_actions({self._robot_name: joint_pos}, env_ids)

    def _get_states(self) -> StatesType:
        """Get the states of the environment."""
        with self._action_lock:
            last_action = self._latest_robot_action
        with self._state_lock:
            self._latest_robot_state.joint_action = last_action
            self._latest_robot_state.sensors = {k: deepcopy(v.data) for k, v in self._sensors[self._robot_name].items()}
            state = deepcopy(self._latest_robot_state)

        self._latest_states[self._robot_name] = state
        return self._latest_states

    def _set_actions(self, actions: ActionsType, env_ids: ArrayType) -> None:
        """Set the dof targets of the environment."""
        if self._cmd_cnt % self._decimation == 0:
            with self._action_lock:
                self._latest_robot_action = actions[self._robot_name].clip(self._pos_limits[0], self._pos_limits[1])
                targets = self._latest_robot_action if self._state_ready.is_set() else self._default_joint_positions
            self._build_cmd_msgs(targets)
        self._cmd_cnt = (self._cmd_cnt + 1) % self._decimation

    def _init_cmd_msgs(self, joint_targets: np.ndarray) -> None:
        for topic, cmd_msg in self._cmd_msgs.items():
            indices = self._topic_joint_indices[topic]
            targets = joint_targets[0, indices]
            kps = self._joint_kp[indices]
            kds = self._joint_kd[indices]
            for i, val in enumerate(targets):
                cmd_msg.motor_cmd[i].mode = 1  # 1:Enable, 0:Disable
                cmd_msg.motor_cmd[i].q = float(val)
                cmd_msg.motor_cmd[i].dq = 0.0
                cmd_msg.motor_cmd[i].tau = 0.0
                cmd_msg.motor_cmd[i].kp = float(kps[i])
                cmd_msg.motor_cmd[i].kd = float(kds[i])
            if hasattr(cmd_msg, "mode_pr"):
                cmd_msg.mode_pr = self._mode_pr
            if hasattr(cmd_msg, "mode_machine"):
                cmd_msg.mode_machine = self._mode_machine
            if hasattr(cmd_msg, "crc"):
                cmd_msg.crc = self._crc.Crc(cmd_msg)

    def _build_cmd_msgs(self, joint_targets: np.ndarray) -> None:
        for topic, cmd_msg in self._cmd_msgs.items():
            targets = joint_targets[0, self._topic_joint_indices[topic]]
            for i, val in enumerate(targets):
                cmd_msg.motor_cmd[i].q = float(val)
            if hasattr(cmd_msg, "mode_pr"):
                cmd_msg.mode_pr = self._mode_pr
            if hasattr(cmd_msg, "mode_machine"):
                cmd_msg.mode_machine = self._mode_machine
            if hasattr(cmd_msg, "crc"):
                cmd_msg.crc = self._crc.Crc(cmd_msg)

    def _assign_robot_state(
        self, topic: str, msg: unitree_hg.msg.dds_.LowState_ | unitree_hg.msg.dds_.HandState_, indices: np.ndarray
    ) -> None:
        self._latest_robot_state.joint_pos[0, indices] = np.array([msg.motor_state[i].q for i in range(len(indices))])
        self._latest_robot_state.joint_vel[0, indices] = np.array([msg.motor_state[i].dq for i in range(len(indices))])
        with self._state_lock:
            # self._latest_robot_state.extras[f"{topic}/raw_msg"] = msg
            self._state_msgs[topic] = msg
            self._latest_robot_state.extras["time_ns"] = time.monotonic_ns()
            if hasattr(msg, "imu_state"):
                self._latest_robot_state.root_state[0, 3:] = np.concatenate(
                    [msg.imu_state.quaternion, msg.imu_state.rpy, msg.imu_state.gyroscope]
                )
            if hasattr(msg, "mode_machine"):
                self._latest_robot_state.extras["mode_machine"] = int(msg.mode_machine)
                self._mode_machine = int(msg.mode_machine)
            if hasattr(msg, "tick"):
                self._latest_robot_state.extras["tick"] = int(msg.tick)

        if self._state_ready.is_set() is False:
            self._state_ready.set()

    def _build_dummy_state(self) -> StatesType:
        states = {}
        for name, cfg in self.cfg_objects.items():
            root_state = (
                np.concatenate([cfg.pose, cfg.twist], axis=-1)[np.newaxis, :]
                .repeat(self.num_envs, axis=0)
                .astype(np.float32)
            )
            states[name] = ObjectState(root_state=root_state)
            if cfg.joints is not None:
                states[name].joint_pos = (
                    self.robots[name].default_joint_positions[np.newaxis, :].repeat(self.num_envs, axis=0)
                )
                states[name].joint_vel = np.zeros_like(states[name].joint_pos)
                states[name].extras = {}
        return states

    def _write_cmd_loop(self) -> None:
        for topic, publisher in self._publishers.items():
            cmd_msg = self._cmd_msgs[topic]
            if hasattr(cmd_msg, "crc"):
                cmd_msg.crc = self._crc.Crc(cmd_msg)
            publisher.Write(cmd_msg)

    def _joystick_loop(self) -> None:
        last_start = 0
        last_select = 0

        msg = self._state_msgs.get("rt/low", None)

        if msg is None or not hasattr(msg, "wireless_remote"):
            return

        try:
            self._remote_controller.parse(msg.wireless_remote)
        except Exception as e:
            logger.warning(f"Failed to parse remote data: {e}")

        start = self._remote_controller.Start
        select = self._remote_controller.Select

        # rising edge trigger
        if start == 1 and last_start == 0:
            logger.debug("Remote: START pressed -> Start task")
            self._start_task.set()

        if select == 1 and last_select == 0:
            logger.debug("Remote: SELECT pressed -> Stop task")
            self._start_task.clear()

        last_start = start
        last_select = select


class UnitreeHighLevelBackend(BaseBackend):
    def __init__(self, config: SimulatorConfig, **kwargs):
        logger.warning("UnitreeHighLevelBackend is not fully implemented; falling back to low-level DDS control.")
        super().__init__(config=config, **kwargs)


class UnitreeFactory:
    @staticmethod
    def create(config: SimulatorConfig, **kwargs) -> BaseBackend:
        spec = config.spec.get("unitree", {})
        mode = spec.get("mode", "low_level").lower()
        if mode == "low_level":
            return UnitreeLowLevelBackend(config=config, **kwargs)
        elif mode == "high_level":
            raise NotImplementedError("UnitreeHighLevelBackend is not yet implemented.")
            # return UnitreeHighLevelBackend(config=config, **kwargs)
        else:
            raise ValueError(f"Invalid control_level: {mode}")
