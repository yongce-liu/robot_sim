from typing import Literal

import gymnasium as gym
import numpy as np
import regex as re

from robot_sim.adapters.gr00t.env import Gr00tEnv
from robot_sim.backends.types import ArrayState
from robot_sim.configs.sensor import SensorType

_ALLOWED_LANGUAGE_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
_BUFFER: dict[str, any] = {}


def obs_joint_state_split(
    group_name: str,
    env: Gr00tEnv,
    joint_patterns: list[str] | str,
    mode: Literal["position", "torque"] = "position",
    **kwargs,
) -> np.ndarray | None:
    """Two Phases:
    1. Initialize _BUFFER through the key
    2. Output the observation value of the corresponding group once called
    """
    if not hasattr(env.observation_mapping, group_name):
        # Initialization buffer phase
        if _BUFFER.get("joint_names") is None:
            _BUFFER["joint_names"] = [
                joint.split("/")[-1].split(".")[-1] for joint in env.backend.get_joint_names(env.robot_name)
            ]
        joint_names = _BUFFER.get("joint_names")
        if isinstance(joint_patterns, str):
            joint_patterns = [joint_patterns]
        _BUFFER[group_name] = []
        for pattern in joint_patterns:
            rx = re.compile(pattern)
            matched_joint_indices = [joint_names.index(name) for name in joint_names if rx.fullmatch(name)]
            assert len(matched_joint_indices) == 1, (
                f"Expected exactly one joint to match pattern '{pattern}', but found {len(matched_joint_indices)}."
            )
            _BUFFER[group_name].extend(matched_joint_indices)
        # initialize observation space
        low_val = [env.robot_cfg.joints[joint_names[idx]].position_limit[0] for idx in _BUFFER[group_name]]
        high_val = [env.robot_cfg.joints[joint_names[idx]].position_limit[1] for idx in _BUFFER[group_name]]
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=np.array(low_val, dtype=np.float32),
            high=np.array(high_val, dtype=np.float32),
            shape=(len(_BUFFER[group_name]),),
            dtype=np.float32,
        )
        return None

    states: ArrayState = kwargs.get("robot_state")
    robot_state = states.objects[env.robot_name]
    if mode == "torque":
        return robot_state.joint_tau[..., _BUFFER[group_name]]
    elif mode == "position":
        return robot_state.joint_pos[..., _BUFFER[group_name]]
    else:
        raise ValueError(f"Unsupported mode '{mode}' for joint_mapping. Available modes are 'position' and 'torque'.")


def obs_body_state_split(
    group_name: str,
    env: Gr00tEnv,
    body_patterns: list[str] | str,
    mode: Literal["position", "quaternion", "pose"] = "pose",
    **kwargs,
) -> np.ndarray | None:
    if not hasattr(env.observation_mapping, group_name):
        # Initialization buffer phase
        if _BUFFER.get("body_names") is None:
            _BUFFER["body_names"] = [
                body.split("/")[-1].split(".")[-1] for body in env.backend.get_body_names(env.robot_name)
            ]
        body_names = _BUFFER.get("body_names")
        if isinstance(body_patterns, str):
            body_patterns = [body_patterns]
        _BUFFER[group_name] = []
        for pattern in body_patterns:
            rx = re.compile(pattern)
            matched_body_indices = [body_names.index(name) for name in body_names if rx.fullmatch(name)]
            _BUFFER[group_name].extend(matched_body_indices)
        # initialize observation space
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(_BUFFER[group_name]), 7),  # Assuming position (3) + orientation (4 as quaternion)
            dtype=np.float32,
        )
        return None
    states: ArrayState = kwargs.get("robot_state")
    robot_state = states.objects[env.robot_name]

    if mode == "position":
        return robot_state.body_state[..., _BUFFER[group_name], :3]
    elif mode == "quaternion":  # [w, x, y, z]
        return robot_state.body_quat[..., _BUFFER[group_name], 3:7]
    elif mode == "pose":
        return robot_state.body_state[..., _BUFFER[group_name], :7]
    else:
        raise ValueError(
            f"Unsupported mode '{mode}' for body_mapping. Available modes are 'position', 'quaternion', and 'pose'."
        )


def obs_video_mapping(
    group_name: str,
    env: Gr00tEnv,
    camera_name: str,
    **kwargs,
) -> np.ndarray | None:
    if not hasattr(env.observation_mapping, group_name):
        assert isinstance(camera_name, str), "Camera name must be a string"
        # Initialization buffer phase
        rx = re.compile(camera_name)
        if _BUFFER.get("camera_names") is None:
            _BUFFER["camera_names"] = [
                name.split("/")[-1].split(".")[-1]
                for name, cfg in env.robot_cfg.sensors.items()
                if cfg.type == SensorType.CAMERA
            ]
        available_cameras = _BUFFER["camera_names"]
        matched_camera_names = [name for name in available_cameras if rx.fullmatch(name)]
        assert len(matched_camera_names) == 1, (
            f"Expected exactly one camera to match camera '{camera_name}', but found {len(matched_camera_names)}."
        )
        _BUFFER[group_name] = matched_camera_names[0]
        # initialize observation space
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                env.robot_cfg.sensors[_BUFFER[group_name]].height,
                env.robot_cfg.sensors[_BUFFER[group_name]].width,
                3,
            ),  # Assuming fixed camera resolution; adjust as needed
            dtype=np.uint8,
        )
        return None
    states = kwargs.get("robot_state")
    robot_state = states.objects[env.robot_name]

    return robot_state.sensors[_BUFFER[group_name]]["rgb"]


def obs_annotation_mapping(
    group_name: str,
    env: Gr00tEnv,
    description: str,
    **kwargs,
) -> str | None:
    if not hasattr(env.observation_mapping, group_name):
        # Initialization buffer phase
        assert isinstance(description, str), "Annotation text must be a string"
        _BUFFER[group_name] = description
        # initialize observation space
        env._observation_space_dict[group_name] = gym.spaces.Text(
            max_length=kwargs.get("max_length", 1024), charset=kwargs.get("charset", _ALLOWED_LANGUAGE_CHARSET)
        )
        return None
    return _BUFFER[group_name]


def act_actuator_space_init(
    group_name: str,
    env: Gr00tEnv,
    actuator_patterns: list[str] | str,
    **kwargs,
) -> np.ndarray | None:
    if not hasattr(env.action_mapping, group_name):
        # Initialization buffer phase
        if _BUFFER.get("actuator_names") is None:
            _BUFFER["actuator_names"] = [
                actuator.split("/")[-1].split(".")[-1] for actuator in env.backend.get_actuator_names(env.robot_name)
            ]
        actuator_names = _BUFFER.get("actuator_names")  # default order of actuators
        if isinstance(actuator_patterns, str):
            actuator_patterns = [actuator_patterns]
        _BUFFER[group_name] = []
        for pattern in actuator_patterns:
            rx = re.compile(pattern)
            matched_actuator_indices = [actuator_names.index(name) for name in actuator_names if rx.fullmatch(name)]
            assert len(matched_actuator_indices) == 1, (
                f"Expected exactly one actuator to match pattern '{pattern}', but found {len(matched_actuator_indices)}."
            )
            _BUFFER[group_name].extend(matched_actuator_indices)
        # initialize action space
        low_val = [env.robot_cfg.joints[actuator_names[idx]].position_limit[0] for idx in _BUFFER[group_name]]
        high_val = [env.robot_cfg.joints[actuator_names[idx]].position_limit[1] for idx in _BUFFER[group_name]]
        env._action_space_dict[group_name] = gym.spaces.Box(
            low=np.array(low_val, dtype=np.float32),
            high=np.array(high_val, dtype=np.float32),
            shape=(len(_BUFFER[group_name]),),
            dtype=np.float32,
        )
        return None


def act_command_space_init(
    group_name: str,
    env: Gr00tEnv,
    command_dim: int,
    bound: dict[Literal["min", "max"], float | int | list[float] | list[int]] = {"min": -np.inf, "max": np.inf},
    **kwargs,
) -> np.ndarray | None:
    if not hasattr(env.action_mapping, group_name):
        # initialize action space
        env._action_space_dict[group_name] = gym.spaces.Box(
            low=np.array(bound.get("min"), dtype=np.float32),
            high=np.array(bound.get("max"), dtype=np.float32),
            shape=(command_dim,),
            dtype=np.float32,
        )
        return None
