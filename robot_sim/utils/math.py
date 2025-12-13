import torch


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    mag_torch = abs(mag) * torch.ones_like(other)
    return torch.copysign(mag_torch, other)


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def euler_xyz_from_quat(
    quat: torch.Tensor, wrap_to_2pi: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw
