import time
from typing import Literal

import numpy as np
from loguru import logger

from robot_sim.bridges import BridgeFactory
from robot_sim.configs import SimulatorConfig

from .base import BaseBackend
from .types import ActionsType, ArrayType, StatesType


class ElasticBackend(BaseBackend):
    """Backend that exchanges state/action via a communication bridge."""

    def __init__(
        self,
        config: SimulatorConfig,
        mode: Literal["server", "client"] | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        assert len(self.robot_names) == 1, "Elastic backend only supports single robot."
        spec = self.config.spec.get(self.type.value, {})
        bridge_cfg = spec.get("bridge", {})
        self._bridge = BridgeFactory.create(**bridge_cfg)

        raw_mode = mode or bridge_cfg.get("mode") or getattr(self._bridge, "mode", None)
        if raw_mode == "robot":
            raw_mode = "client"
        if raw_mode not in ("server", "client"):
            raise ValueError(f"Unknown elastic backend mode: {raw_mode}")
        self._mode: Literal["server", "client"] = raw_mode

        self._state_topic = spec.get("state_topic", "rt/robot_sim/state")
        self._action_topic = spec.get("action_topic", "rt/robot_sim/action")
        self._blocking = bool(spec.get("blocking", True))
        self._step_timeout_s = float(spec.get("step_timeout_s", 2.0))

        self._state_tick = 0
        self._action_tick = 0
        self._last_state_tick = -1
        self._last_action_tick = -1
        self._states_cache: StatesType | None = None
        self._actions_cache: ActionsType | None = None

    def _launch(self) -> None:
        self._bridge.launch()

    def _render(self) -> None:
        return None

    def get_rgb_image(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        self._bridge.close()

    def _poll_state(self) -> bool:
        msg = self._bridge.get_state(self._state_topic)
        if msg is None:
            return False
        tick, _, states = msg
        if tick <= self._last_state_tick:
            return False
        self._last_state_tick = tick
        self._states_cache = states
        return True

    def _poll_action(self) -> bool:
        msg = self._bridge.get_action(self._action_topic)
        if msg is None:
            return False
        tick, _, actions = msg
        if tick <= self._last_action_tick:
            return False
        self._last_action_tick = tick
        self._actions_cache = actions
        return True

    def _wait_for_state(self) -> None:
        if not self._blocking:
            self._poll_state()
            return
        deadline = time.monotonic() + self._step_timeout_s
        while time.monotonic() < deadline:
            if self._poll_state():
                return
            time.sleep(0.001)
        logger.warning("Elastic backend state timeout.")

    def _wait_for_action(self) -> None:
        if not self._blocking:
            self._poll_action()
            return
        deadline = time.monotonic() + self._step_timeout_s
        while time.monotonic() < deadline:
            if self._poll_action():
                return
            time.sleep(0.001)
        logger.warning("Elastic backend action timeout.")

    def _simulate(self):
        if self._mode == "server":
            self._wait_for_state()
        else:
            self._wait_for_action()

    def _set_states(self, states: StatesType, env_ids: ArrayType | None = None) -> None:
        if self._mode != "client":
            logger.warning("Elastic server does not support setting states.")
            return
        if env_ids is not None:
            logger.warning("Elastic client ignores env_ids when sending states.")
        self._states_cache = states
        self._state_tick += 1
        self._bridge.send_state(
            self._state_topic,
            states,
            tick=self._state_tick,
            stamp_ns=time.monotonic_ns(),
        )

    def _set_actions(self, actions: ActionsType) -> None:
        if self._mode != "server":
            logger.warning("Elastic client ignores set_actions; actions come from bridge.")
            return
        self._action_tick += 1
        self._bridge.send_action(
            self._action_topic,
            actions,
            tick=self._action_tick,
            stamp_ns=time.monotonic_ns(),
        )

    def _get_states(self) -> StatesType:
        if self._mode == "server":
            if self._states_cache is None:
                self._wait_for_state()
            if self._states_cache is None:
                raise RuntimeError("No states received from elastic bridge.")
            return self._states_cache
        if self._states_cache is None:
            raise RuntimeError("Elastic client has no local state to return.")
        return self._states_cache

    def get_actions(self) -> ActionsType | None:
        """Client helper: fetch the latest action from the bridge."""
        if self._mode != "client":
            return None
        if self._actions_cache is None:
            self._wait_for_action()
        return self._actions_cache

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
