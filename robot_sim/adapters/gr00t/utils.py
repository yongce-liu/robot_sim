from typing import Literal

import numpy as np

import robot_sim.utils.math_array as rmath
from robot_sim.backends.types import ActionsType, ArrayType, StatesType


def obs_joint_extract(
    name: str,
    states: StatesType,
    indices: list[int],
    mode: Literal["position", "torque"] = "position",
) -> ArrayType:
    robot_state = states[name]
    if mode == "torque":
        return robot_state.joint_action[0, indices]
    elif mode == "position":
        return robot_state.joint_pos[0, indices]
    else:
        raise ValueError(f"Unsupported mode '{mode}' for joint_map. Available modes are 'position' and 'torque'.")


def act_joint_assign(
    name: str,
    group_name: str,
    indices: list[int],
    action: ActionsType,
    **kwargs,
):
    action[name][..., indices] = action[group_name]


def rpy_cmd_from_waist(
    name: str,
    torso_index: int,
    pelvis_index: int,
    action: ActionsType,
    states: StatesType,
) -> np.ndarray:
    # Compute torso orientation relative to waist, to pass to lower body policy
    body_state = states[name].body_state
    #################### Previouos ######################################
    # torso_orientation = matrix_from_quat(body_state[..., torso_index, 3:7])
    # waist_orientation = matrix_from_quat(body_state[..., pelvis_index, 3:7])
    # # Extract yaw from rotation matrix and create a rotation with only yaw
    # # The rotation property is a 3x3 numpy array
    # waist_yaw = np.arctan2(waist_orientation[1, 0], waist_orientation[0, 0])
    # # Create a rotation matrix with only yaw using Pinocchio's rpy functions
    # waist_yaw_only_rotation = matrix_from_euler([0, 0, float(waist_yaw)])
    # yaw_only_waist_from_torso = waist_yaw_only_rotation.T @ torso_orientation
    # torso_orientation_rpy = rotmat_to_rpy(yaw_only_waist_from_torso)
    #################### Current ######################################
    torso_quat = body_state[..., torso_index, 3:7]  # (B, 4), (w, x, y, z)
    waist_quat = body_state[..., pelvis_index, 3:7]  # (B, 4)
    waist_yaw_quat = rmath.yaw_quat(waist_quat)  # (B, 4)
    waist_yaw_quat_inv = rmath.quat_conjugate(waist_yaw_quat)  # (B, 4)
    yaw_only_waist_from_torso_quat = rmath.quat_mul(waist_yaw_quat_inv, torso_quat)
    rpy_cmd = np.stack(rmath.euler_xyz_from_quat(yaw_only_waist_from_torso_quat), axis=-1)

    return rpy_cmd
