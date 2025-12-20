import gymnasium as gym
import numpy as np
import regex as re

from robot_sim.adapters.gr00t.env import Gr00tEnv
from robot_sim.configs.sensor import SensorType

_ALLOWED_LANGUAGE_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
_BUFFER: dict[str, any] = {}


def joint_mapping(
    group_name: str,
    env: Gr00tEnv,
    joint_patterns: list[str] | str,
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
            _BUFFER[group_name].extend(matched_joint_indices)
        # initialize observation space
        low_val = [env.robot_cfg.joints[name].position_limit[0] for name in joint_names]
        high_val = [env.robot_cfg.joints[name].position_limit[1] for name in joint_names]
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=np.array(low_val, dtype=np.float32),
            high=np.array(high_val, dtype=np.float32),
            shape=(len(_BUFFER[group_name]),),
            dtype=np.float32,
        )
        return None
    states = kwargs.get("robot_state")
    robot_state = states.objects[env.robot_name]
    return robot_state.joint_pos[..., _BUFFER[group_name]]


def video_mapping(
    group_name: str,
    env: Gr00tEnv,
    camera_pattern: str,
    **kwargs,
) -> np.ndarray | None:
    if not hasattr(env.observation_mapping, group_name):
        assert isinstance(camera_pattern, str), "Camera pattern must be a string"
        # Initialization buffer phase
        rx = re.compile(camera_pattern)
        if _BUFFER.get("camera_names") is None:
            _BUFFER["camera_names"] = [
                name.split("/")[-1].split(".")[-1]
                for name, cfg in env.robot_cfg.sensors.items()
                if cfg.type == SensorType.CAMERA
            ]
        available_cameras = _BUFFER["camera_names"]
        matched_camera_names = [name for name in available_cameras if rx.fullmatch(name)]
        assert len(matched_camera_names) == 1, (
            f"Expected exactly one camera to match pattern '{camera_pattern}', but found {len(matched_camera_names)}."
        )
        _BUFFER[group_name] = matched_camera_names[0]
        # initialize observation space
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                env.robot_cfg.sensors[_BUFFER[group_name]].get("height"),
                env.robot_cfg.sensors[_BUFFER[group_name]].get("width"),
                3,
            ),  # Assuming fixed camera resolution; adjust as needed
            dtype=np.uint8,
        )
        return None
    states = kwargs.get("robot_state")
    robot_state = states.objects[env.robot_name]

    return robot_state.sensors[_BUFFER[group_name]]["rgb"]


def annotation_mapping(
    group_name: str,
    env: Gr00tEnv,
    annotation_text: str,
    **kwargs,
) -> str | None:
    if not hasattr(env.observation_mapping, group_name):
        # Initialization buffer phase
        assert isinstance(annotation_text, str), "Annotation text must be a string"
        _BUFFER[group_name] = annotation_text
        # initialize observation space
        env._observation_space_dict[group_name] = gym.spaces.Text(
            max_length=kwargs.get("max_length", 1024), charset=kwargs.get("charset", _ALLOWED_LANGUAGE_CHARSET)
        )
        return None
    return _BUFFER[group_name]
