from typing import Literal

import gymnasium as gym
import numpy as np
import regex as re

from robot_sim.backends.types import StatesType
from robot_sim.configs import MapFunc, SensorType
from robot_sim.envs import MapEnv

"""
Two Phases:
1. Initialize _BUFFER through the key
2. Output the observation value of the corresponding group once called
"""


def rpy_to_rotmat(rpy):
    """
    Convert roll, pitch, yaw (RPY) angles to a 3x3 rotation matrix.
    Uses ZYX (yaw-pitch-roll) intrinsic rotation order, which is standard in robotics.

    Parameters:
        rpy : array-like, shape (3,)
            [roll, pitch, yaw] in radians.

    Returns:
        R : np.ndarray, shape (3, 3)
            Rotation matrix (float64).
    """
    roll, pitch, yaw = rpy

    # Precompute sines and cosines
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )

    return R


def rotmat_to_rpy(R):
    """
    Convert rotation matrix to roll, pitch, yaw (ZYX convention).
    Returns: np.array([roll, pitch, yaw])
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert a WXYZ quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


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
        joint_names = [joint for joint in env.get_joint_names(env.robot_name)]
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
        states: StatesType,
        mode: Literal["position", "torque"] = "position",
        **kwargs,
    ) -> np.ndarray:
        robot_state = states[self.env.robot_name]
        if mode == "torque":
            return robot_state.joint_action[0, self.group_joint_indices]
        elif mode == "position":
            return robot_state.joint_pos[0, self.group_joint_indices]
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
        body_names = [body for body in env.get_body_names(env.robot_name)]
        if isinstance(body_patterns, str):
            body_patterns = [body_patterns]
        buffer = []
        for pattern in body_patterns:
            rx = re.compile(pattern)
            matched_body_indices = [body_names.index(name) for name in body_names if rx.fullmatch(name)]
            buffer.extend(matched_body_indices)
        self.group_body_indices = buffer
        # initialize observation space
        mode = kwargs.get("mode", "pose")
        vec_dim = 7 if mode == "pose" else 3 if mode == "position" else 4 if mode == "quaternion" else None
        env._observation_space_dict[group_name] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.group_body_indices), vec_dim),  # Assuming position (3) + orientation (4 as quaternion)
            dtype=np.float32,
        )
        return super().init(env, group_name, **kwargs)

    def __call__(
        self, states: StatesType, mode: Literal["position", "quaternion", "pose"] = "pose", **kwargs
    ) -> np.ndarray:
        robot_state = states[self.env.robot_name]

        if mode == "position":
            return robot_state.body_state[0, self.group_body_indices, :3]
        elif mode == "quaternion":  # [w, x, y, z]
            return robot_state.body_state[0, self.group_body_indices, 3:7]
        elif mode == "pose":
            return robot_state.body_state[0, self.group_body_indices, :7]
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
        available_cameras = [name for name, cfg in env.robot_cfg.sensors.items() if cfg.type == SensorType.CAMERA]
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

    def __call__(self, states: StatesType, **kwargs) -> np.ndarray | dict:
        robot_state = states[self.env.robot_name]

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


class act_actuator_map(MapFunc):
    def init(
        self,
        env: MapEnv,
        group_name: str,
        actuator_patterns: list[str] | str,
        **kwargs,
    ) -> np.ndarray | None:
        # Initialization buffer phase
        actuator_names = [actuator for actuator in env.get_actuator_names(env.robot_name)]
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

    def __call__(self, action: dict[str, np.ndarray], **kwargs) -> None:
        # Set the corresponding actuator commands in the backend action array
        action[self.env.robot_name][..., self.group_actuator_indices] = action[self.group_name]


class act_command_map(MapFunc):
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

    def __call__(self, action: dict[str, np.ndarray], **kwargs) -> None:
        if self.group_name == "action.base_height_command":
            return None
        elif self.group_name == "action.navigate_command":
            return None
        elif self.group_name == "action.rpy_command":
            # Compute torso orientation relative to waist, to pass to lower body policy
            body_state = self.env.states[self.env.robot_name].body_state
            # Get torso and waist indices
            torso_index = self.env.get_body_names(self.env.robot_name).index("torso_link")
            pelvis_index = self.env.get_body_names(self.env.robot_name).index("pelvis")
            torso_orientation = quat_wxyz_to_rotmat(body_state[0, torso_index, 3:7])
            waist_orientation = quat_wxyz_to_rotmat(body_state[0, pelvis_index, 3:7])
            # Extract yaw from rotation matrix and create a rotation with only yaw
            # The rotation property is a 3x3 numpy array
            waist_yaw = np.arctan2(waist_orientation[1, 0], waist_orientation[0, 0])
            # Create a rotation matrix with only yaw using Pinocchio's rpy functions
            waist_yaw_only_rotation = rpy_to_rotmat([0, 0, float(waist_yaw)])
            yaw_only_waist_from_torso = waist_yaw_only_rotation.T @ torso_orientation
            torso_orientation_rpy = rotmat_to_rpy(yaw_only_waist_from_torso)

            return torso_orientation_rpy
        else:
            raise ValueError(f"Unsupported command group '{self.group_name}' in act_command_map.")
