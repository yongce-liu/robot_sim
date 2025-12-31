import json
import time
from collections import deque
from typing import Any

import numpy as np
import redis
import torch
from loguru import logger

from robot_sim.configs import ObjectConfig

from .utils import OnnxPolicyWrapper


class Twist2Policy:
    def __init__(
        self,
        robot_name: str,
        policy_path: str,
        robot_config: ObjectConfig,
        mdp_config: dict[str, Any] = None,
        redis_config: dict[str, Any] = None,
        logs_config: dict[str, Any] = None,
        device="cuda",
    ):
        self.policy = OnnxPolicyWrapper.load_onnx_policy(policy_path, device)
        self.device = device
        self.robot_name = robot_name
        self._init_robot_info(robot_config)
        self._init_mdp(**mdp_config)
        self._init_logs(**logs_config)
        self._init_redis(**redis_config)

    def unpack_data(self, observation: dict[str, Any]):
        """Extract robot state data"""
        body_dof_pos = observation["body_dof_pos"]
        body_dof_vel = observation["body_dof_vel"]
        rpy = observation["rpy"]
        ang_vel = observation["ang_vel"]
        left_hand_dof_pos = observation.get("left_hand_dof_pos", None)
        right_hand_dof_pos = observation.get("right_hand_dof_pos", None)

        return body_dof_pos, body_dof_vel, rpy, ang_vel, left_hand_dof_pos, right_hand_dof_pos

    def run_once(self, observation: dict[str, Any]):
        # Add policy execution FPS tracking for frequent printing
        # Build proprioceptive observation
        dof_pos, dof_vel, rpy, ang_vel, left_hand_state, right_hand_state = self.unpack_data(observation)
        dof_vel[self.ankle_idx] = 0.0
        obs_proprio = np.concatenate(
            [
                ang_vel * 0.25,
                rpy[:2],  # only use roll and pitch
                (dof_pos - self.default_dof_pos),
                dof_vel * 0.05,
                self.last_action,
            ]
        )

        self.__send_buffer[f"state_body_{self.robot_name}"] = json.dumps(
            np.concatenate([ang_vel, rpy[:2], dof_pos]).tolist()
        )  # 3+2+29 = 34 dims
        if left_hand_state is not None:
            self.__send_buffer[f"state_hand_left_{self.robot_name}"] = json.dumps(left_hand_state.tolist())  # 7 dims
        if right_hand_state is not None:
            self.__send_buffer[f"state_hand_right_{self.robot_name}"] = json.dumps(right_hand_state.tolist())  # 7 dims

        # Send proprio to redis
        for k in self.redis_send_keys:
            self.redis_pipeline.set(k, self.__send_buffer[k])
        self.redis_pipeline.set("t_state", int(time.time() * 1000))  # current timestamp in ms
        self.redis_pipeline.execute()

        # Get mimic obs from Redis
        for k in self.redis_recv_keys:
            self.redis_pipeline.get(k)
        redis_results = self.redis_pipeline.execute()

        for i, k in enumerate(self.redis_recv_keys):
            self.__recv_buffer[k] = json.loads(redis_results[i])
        action_mimic = np.array(self.__recv_buffer[f"action_body_{self.robot_name}"])  # 35 dims
        action_left_hand = np.array(self.__recv_buffer[f"action_hand_left_{self.robot_name}"]) * self.use_hand_coeff
        action_right_hand = np.array(self.__recv_buffer[f"action_hand_right_{self.robot_name}"]) * self.use_hand_coeff
        # action_neck = json.loads(redis_results[3])

        # Construct observation for TWIST2 controller
        obs_full = np.concatenate([action_mimic, obs_proprio])
        # Update history
        obs_hist = np.array(self.proprio_history_buf).flatten()
        self.proprio_history_buf.append(obs_full)
        future_obs = action_mimic.copy()
        # Combine all observations: current + history + future (set to current frame for now)
        obs_buf = np.concatenate([obs_full, obs_hist, future_obs])

        # Run policy
        obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

        # Measure and track policy execution FPS
        current_time = time.time()
        if self.policy_time is not None:
            if self.policy_execution_times is not None:
                policy_interval = current_time - self.policy_time
                current_policy_fps = 1.0 / policy_interval

                # For frequent printing (every 100 steps)
                self.policy_execution_times.append(policy_interval)

                # Print policy execution FPS every 100 steps
                if len(self.policy_execution_times) == self.policy_execution_times.maxlen:
                    avg_interval = np.mean(self.policy_execution_times)
                    avg_execution_fps = 1.0 / avg_interval
                    logger.info(
                        f"Policy Execution FPS (last {self.policy_execution_times.maxlen} steps): {avg_execution_fps:.2f} Hz (avg interval: {avg_interval * 1000:.2f}ms)"
                    )
                    self.policy_execution_times.clear()

            # For detailed measurement (every 1000 steps)
            if self.fps_measurements is not None:
                self.fps_measurements.append(current_policy_fps)
                if len(self.fps_measurements) == self.fps_measurements.maxlen:
                    avg_fps = np.mean(self.fps_measurements)
                    max_fps = np.max(self.fps_measurements)
                    min_fps = np.min(self.fps_measurements)
                    std_fps = np.std(self.fps_measurements)
                    logger.info(
                        f"Average Policy FPS: {avg_fps:.2f}\nMax Policy FPS: {max_fps:.2f}\nMin Policy FPS: {min_fps:.2f}\nStd Policy FPS: {std_fps:.2f}\nExpected FPS (from decimation): {1.0 / (self.expected_fps):.2f}"
                    )
                    self.fps_measurements.clear()

        self.policy_time = current_time
        self.last_action = raw_action

        # self.redis_client.set("action_low_level_unitree_g1", json.dumps(raw_action.tolist()))
        # Record proprio if enabled
        if self.proprio_recordings is not None:
            proprio_data = {
                "timestamp": time.time(),
                "dof_pos": dof_pos.tolist(),
                "dof_vel": dof_vel.tolist(),
                "rpy": rpy.tolist(),
                "ang_vel": ang_vel.tolist(),
                "target_dof_pos": action_mimic.tolist()[-29:],
            }
            self.proprio_recordings.append(proprio_data)

        target_body_dof_pos = (
            raw_action.clip(-self.action_clip, self.action_clip) * self.action_scale + self.default_dof_pos
        )
        target_left_hand_dof_pos = action_left_hand
        target_right_hand_dof_pos = action_right_hand

        return {
            "dof_pos": np.concatenate(
                [target_body_dof_pos, target_left_hand_dof_pos, target_right_hand_dof_pos], axis=-1
            )[np.newaxis, :],
        }

    ############### HELPER FUNCTION FOR RUN ###############
    def _init_robot_info(self, robot_config: ObjectConfig):
        # G1 specific configuration
        self.default_dof_pos = np.array(
            [joint.default_position for name, joint in robot_config.joints.items() if "hand" not in name],
            dtype=np.float32,
        )
        self.last_action = np.zeros_like(self.default_dof_pos)

        self.stiffness = np.array(
            [joint.stiffness for name, joint in robot_config.joints.items() if "hand" not in name],
            dtype=np.float32,
        )
        self.damping = np.array(
            [joint.damping for name, joint in robot_config.joints.items() if "hand" not in name],
            dtype=np.float32,
        )
        self.torque_limits = np.array(
            [joint.torque_limit for name, joint in robot_config.joints.items() if "hand" not in name],
            dtype=np.float32,
        )

        self.ankle_idx = robot_config.extra.get("ankle_idx", [4, 5, 10, 11])

    def _init_mdp(
        self,
        n_mimic_obs: int,
        n_proprio: int,
        history_len: int,
        action_clip: float | list[float] = 10.0,
        action_scale: float | list[float] = 0.5,
        use_hand_action: bool = False,
    ):
        n_obs_single = n_mimic_obs + n_proprio  # n_mimic_obs + n_proprio = 35 + 92 = 127
        self.total_obs_size = n_obs_single * (history_len + 1) + n_mimic_obs  # 127*11 + 35 = 1402
        self.action_clip = np.array(action_clip)
        self.action_scale = np.array(action_scale)
        self.use_hand_coeff = 1.0 if use_hand_action else 0.0

        # Initialize history buffer
        self.proprio_history_buf = deque(
            [np.zeros(n_obs_single, dtype=np.float32) for _ in range(history_len)], maxlen=history_len
        )

        logger.info(
            f"TWIST2 Controller Configuration:\n\tn_mimic_obs: {n_mimic_obs}\n\tn_proprio: {n_proprio}\n\tn_obs_single: {n_obs_single}\n\thistory_len: {history_len}\n\ttotal_obs_size: {self.total_obs_size}"
        )

    def _init_logs(
        self,
        record_proprio: bool = False,
        mesure_fps_size: int = None,
        expected_fps: int = None,
        measure_policy_size: int = None,
    ):
        self.policy_time = None
        self.proprio_recordings = [] if record_proprio else None
        self.fps_measurements = deque(maxlen=mesure_fps_size) if mesure_fps_size else None
        self.expected_fps = expected_fps if expected_fps > 0 else None
        self.policy_execution_times = deque(maxlen=measure_policy_size) if measure_policy_size else None

    def _init_redis(
        self,
        send_keys: list[str],
        send_size: list[int],
        recv_keys: list[str],
        host: str = "localhost",
        port: int = 8888,
        db: int = 0,
    ):
        # redis settings
        redis_client = redis.Redis(host=host, port=port, db=db)
        self.redis_pipeline = redis_client.pipeline()
        self.redis_send_keys = [f"{k}_{self.robot_name}" for k in send_keys]
        self.__send_buffer = {k: json.dumps([0] * send_size[i]) for i, k in enumerate(self.redis_send_keys)}
        self.redis_recv_keys = [f"{k}_{self.robot_name}" for k in recv_keys]
        self.__recv_buffer = {k: None for k in self.redis_recv_keys}

        for k in self.redis_send_keys:
            self.redis_pipeline.set(k, self.__send_buffer.get(k, None))
        self.redis_pipeline.execute()
