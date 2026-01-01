"""Gr00t Teleoperation task for Gr00t adapter using Pico Ultra 4 and Motion Tracker."""

import json
from typing import Any

import numpy as np
import redis
from loguru import logger

from robot_sim.adapters.gr00t.env import Gr00tEnv
from robot_sim.backends.types import ArrayType
from robot_sim.configs import SimulatorConfig
from robot_sim.utils.helper import task_register


@task_register("Gr00tTeleoperation-v0")
class TeleoperationTask(Gr00tEnv):
    def __init__(
        self,
        config: SimulatorConfig,
        body2part_indices: dict[str, list[int] | ArrayType],
        redis_config: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        logger.info(f"TeleoperationTask with robot: {self.robot_name}")
        for k, v in body2part_indices.items():
            body2part_indices[k] = np.array(v, dtype=np.int32)
        self.body2part_indices = body2part_indices
        self._init_redis(**redis_config)

    def get_action(self) -> dict[str, ArrayType]:
        # Get mimic obs from Redis
        for k in self.redis_recv_keys:
            self.redis_pipeline.get(k)
        redis_results = self.redis_pipeline.execute()

        for i, k in enumerate(self.redis_recv_keys):
            self.__recv_buffer[k] = json.loads(redis_results[i])
        # xy velocity, z position, roll/pitch, yaw angular velocity, 29 dof
        action_mimic = np.array(self.__recv_buffer[f"action_body_{self.robot_name}"])  # 35 dims
        nav_cmd, height_cmd, rp_cmd, body_dof_cmd = self.unpack_action(action_mimic)
        action_left_hand = np.array(self.__recv_buffer[f"action_hand_left_{self.robot_name}"])[np.newaxis, :]
        action_right_hand = np.array(self.__recv_buffer[f"action_hand_right_{self.robot_name}"])[np.newaxis, :]

        dof_pos = np.concatenate([body_dof_cmd, action_left_hand, action_right_hand], axis=-1)

        action = {
            "action.navigate_command": nav_cmd,
            "action.base_height_command": height_cmd,
        }
        for k, v in self.body2part_indices.items():
            action[k] = dof_pos[..., v]

        return action

    def unpack_action(self, action_mimic: np.ndarray):
        nav_cmd = action_mimic[0:2][np.newaxis, :]
        height_cmd = action_mimic[2:3][np.newaxis, :]
        rp_cmd = action_mimic[3:5][np.newaxis, :]
        yaw_cmd = action_mimic[5:6][np.newaxis, :]
        body_dof_cmd = action_mimic[6:][np.newaxis, :]
        return np.concatenate([nav_cmd, yaw_cmd], axis=-1), height_cmd, rp_cmd, body_dof_cmd

    def step(self, action=None):
        action = self.get_action()
        return super().step(action)

    def _init_redis(
        self,
        # send_keys: list[str],
        # send_size: list[int],
        recv_keys: list[str],
        host: str = "localhost",
        port: int = 8888,
        db: int = 0,
    ):
        # redis settings
        redis_client = redis.Redis(host=host, port=port, db=db)
        self.redis_pipeline = redis_client.pipeline()

        # self.redis_send_keys = [f"{k}_{self.robot_name}" for k in send_keys]
        # self.__send_buffer = {k: json.dumps([0] * send_size[i]) for i, k in enumerate(self.redis_send_keys)}
        # for k in self.redis_send_keys:
        #     self.redis_pipeline.set(k, self.__send_buffer.get(k, None))
        # self.redis_pipeline.execute()

        self.redis_recv_keys = [f"{k}_{self.robot_name}" for k in recv_keys]
        self.__recv_buffer = {k: None for k in self.redis_recv_keys}
