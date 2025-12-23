from typing import Literal

import numpy as np


def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    x_normalized = x / np.clip(norm, a_min=eps, a_max=None)
    return x_normalized


def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    # Store original shape of vec
    shape = vec.shape
    # Reshape to (-1, 4) and (-1, 3)
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # Extract xyz part (imaginary components)
    xyz = quat[:, 1:]  # shape: (N, 3)
    w = quat[:, 0:1]  # shape: (N, 1)
    # Compute t = 2 * cross(xyz, vec)
    t = 2.0 * np.cross(xyz, vec)  # shape: (N, 3)
    # Compute result: vec - w * t + cross(xyz, t)
    result = vec - w * t + np.cross(xyz, t)
    # Restore original shape
    return result.reshape(shape)


def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _axis_angle_rotation(axis: Literal["X", "Y", "Z"], angle: np.ndarray) -> np.ndarray:
    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)
    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_from_euler(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, np.moveaxis(euler_angles, -1, 0))]
    # return functools.reduce(torch.matmul, matrices)
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])


def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = np.zeros_like(quat_yaw)
    quat_yaw[:, 3] = np.sin(yaw / 2)
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.reshape(shape)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def copysign(mag: float, other: np.ndarray) -> np.ndarray:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    mag_array = abs(mag) * np.ones_like(other)
    return np.copysign(mag_array, other)


def euler_xyz_from_quat(quat: np.ndarray, wrap_to_2pi: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ extrinsic convention.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
        wrap_to_2pi (bool): Whether to wrap output Euler angles into [0, 2π). If
            False, angles are returned in the default range (−π, π]. Defaults to
            False.

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (N,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sin_pitch) >= 1, copysign(np.pi / 2.0, sin_pitch), np.arcsin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    return roll, pitch, yaw
