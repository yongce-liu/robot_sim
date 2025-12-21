from typing import Literal

import gymnasium as gym
import numpy as np
import regex as re

from robot_sim.backends.types import ArrayState
from robot_sim.configs import MapFunc, SensorType
from robot_sim.envs import MapEnv

"""
Two Phases:
1. Initialize _BUFFER through the key
2. Output the observation value of the corresponding group once called
"""


class obs_joint_state_split(MapFunc):
    def init(
        self,
        env: MapEnv,
        group_name: str,
        joint_patterns: list[str] | str,
        **kwargs,
    ) -> None:
        # Initialization buffer phase
        self.group_name = group_name
        joint_names = [joint.split("/")[-1].split(".")[-1] for joint in env.backend.get_joint_names(env.robot_name)]
        if isinstance(joint_patterns, str):
            joint_patterns = [joint_patterns]
        buffer = []
        for pattern in joint_patterns:
            rx = re.compile(pattern)
            matched_joint_indices = [joint_names.index(name) for name in joint_names if rx.fullmatch(name)]
            assert len(matched_joint_indices) == 1, (
                f"Expected exactly one joint to match pattern '{pattern}', but found {len(matched_joint_indices)}."
            )
            buffer.extend(matched_joint_indices)
        # initialize observation space
        low_val = [env.robot_cfg.joints[joint_names[idx]].position_limit[0] for idx in buffer]
        high_val = [env.robot_cfg.joints[joint_names[idx]].position_limit[1] for idx in buffer]
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=np.array(low_val, dtype=np.float32),
            high=np.array(high_val, dtype=np.float32),
            shape=(len(buffer),),
            dtype=np.float32,
        )
        self.group_joint_indices = buffer
        return super().init(env, group_name, **kwargs)

    def __call__(
        self,
        states: ArrayState,
        mode: Literal["position", "torque"] = "position",
        **kwargs,
    ) -> np.ndarray:
        robot_state = states.objects[self.env.robot_name]
        if mode == "torque":
            return robot_state.joint_tau[..., self.group_joint_indices]
        elif mode == "position":
            return robot_state.joint_pos[..., self.group_joint_indices]
        else:
            raise ValueError(f"Unsupported mode '{mode}' for joint_map. Available modes are 'position' and 'torque'.")


class obs_body_state_split(MapFunc):
    def init(
        self,
        env: MapEnv,
        group_name: str,
        body_patterns: list[str] | str,
        **kwargs,
    ) -> np.ndarray | None:
        # Initialization buffer phase
        body_names = [body.split("/")[-1].split(".")[-1] for body in env.backend.get_body_names(env.robot_name)]
        if isinstance(body_patterns, str):
            body_patterns = [body_patterns]
        buffer = []
        for pattern in body_patterns:
            rx = re.compile(pattern)
            matched_body_indices = [body_names.index(name) for name in body_names if rx.fullmatch(name)]
            buffer.extend(matched_body_indices)
        self.group_body_indices = buffer
        # initialize observation space
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.group_body_indices), 7),  # Assuming position (3) + orientation (4 as quaternion)
            dtype=np.float32,
        )
        return super().init(env, group_name, **kwargs)

    def __call__(
        self, states: ArrayState, mode: Literal["position", "quaternion", "pose"] = "pose", **kwargs
    ) -> np.ndarray:
        robot_state = states.objects[self.env.robot_name]

        if mode == "position":
            return robot_state.body_state[..., self.group_body_indices, :3]
        elif mode == "quaternion":  # [w, x, y, z]
            return robot_state.body_quat[..., self.group_body_indices, 3:7]
        elif mode == "pose":
            return robot_state.body_state[..., self.group_body_indices, :7]
        else:
            raise ValueError(
                f"Unsupported mode '{mode}' for body_map. Available modes are 'position', 'quaternion', and 'pose'."
            )


class obs_video_map(MapFunc):
    def init(
        self,
        env: MapEnv,
        group_name: str,
        camera_name: str,
        **kwargs,
    ) -> np.ndarray | None:
        assert isinstance(camera_name, str), "Camera name must be a string"
        # Initialization buffer phase
        rx = re.compile(camera_name)
        available_cameras = [
            name.split("/")[-1].split(".")[-1]
            for name, cfg in env.robot_cfg.sensors.items()
            if cfg.type == SensorType.CAMERA
        ]
        matched_camera_names = [name for name in available_cameras if rx.fullmatch(name)]
        assert len(matched_camera_names) == 1, (
            f"Expected exactly one camera to match camera '{camera_name}', but found {len(matched_camera_names)}."
        )
        self.camera_name = matched_camera_names[0]
        # initialize observation space
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                env.robot_cfg.sensors[self.camera_name].height,
                env.robot_cfg.sensors[self.camera_name].width,
                3,
            ),  # Assuming fixed camera resolution; adjust as needed
            dtype=np.uint8,
        )
        return super().init(env, group_name, **kwargs)

    def __call__(self, states: ArrayState, **kwargs) -> np.ndarray:
        robot_state = states.objects[self.env.robot_name]

        return robot_state.sensors[self.camera_name]["rgb"]


class obs_annotation_map(MapFunc):
    def init(
        self,
        env: MapEnv,
        group_name: str,
        description: str,
        allowed_language_charset: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:",
        **kwargs,
    ) -> str | None:
        # Initialization buffer phase
        assert isinstance(description, str), "Annotation text must be a string"
        self.description = description
        # initialize observation space
        env._observation_space_dict[group_name] = gym.spaces.Text(
            max_length=kwargs.get("max_length", 1024), charset=kwargs.get("charset", allowed_language_charset)
        )
        return super().init(env, group_name, **kwargs)

    def __call__(self, **kwargs) -> str:
        return self.description


class act_actuator_space_init(MapFunc):
    def init(
        self,
        env: MapEnv,
        group_name: str,
        actuator_patterns: list[str] | str,
        **kwargs,
    ) -> np.ndarray | None:
        # Initialization buffer phase
        actuator_names = [
            actuator.split("/")[-1].split(".")[-1] for actuator in env.backend.get_actuator_names(env.robot_name)
        ]
        if isinstance(actuator_patterns, str):
            actuator_patterns = [actuator_patterns]
        buffer = []
        for pattern in actuator_patterns:
            rx = re.compile(pattern)
            matched_actuator_indices = [actuator_names.index(name) for name in actuator_names if rx.fullmatch(name)]
            assert len(matched_actuator_indices) == 1, (
                f"Expected exactly one actuator to match pattern '{pattern}', but found {len(matched_actuator_indices)}."
            )
            buffer.extend(matched_actuator_indices)
        # initialize action space
        low_val = [env.robot_cfg.joints[actuator_names[idx]].position_limit[0] for idx in buffer]
        high_val = [env.robot_cfg.joints[actuator_names[idx]].position_limit[1] for idx in buffer]
        env._action_space_dict[group_name] = gym.spaces.Box(
            low=np.array(low_val, dtype=np.float32),
            high=np.array(high_val, dtype=np.float32),
            shape=(len(buffer),),
            dtype=np.float32,
        )
        self.group_actuator_indices = buffer
        return super().init(env, group_name, **kwargs)


class act_command_space_init(MapFunc):
    def init(
        self,
        env: MapEnv,
        group_name: str,
        command_dim: int,
        bound: dict[Literal["min", "max"], float | int | list[float] | list[int]] = {"min": -np.inf, "max": np.inf},
        **kwargs,
    ) -> np.ndarray | None:
        # initialize action space
        env._action_space_dict[group_name] = gym.spaces.Box(
            low=np.array(bound.get("min"), dtype=np.float32),
            high=np.array(bound.get("max"), dtype=np.float32),
            shape=(command_dim,),
            dtype=np.float32,
        )
        return super().init(env, group_name, **kwargs)
