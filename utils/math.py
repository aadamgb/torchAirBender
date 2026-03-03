import torch
import torch.nn.functional as F
from torch import Tensor

@torch.jit.script
def quat_to_rotmat(q: Tensor) -> Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        q : (B, 4)  [w, x, y, z]

    Returns:
        R : (B, 3, 3)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)], dim=-1),  # row 0
        torch.stack([    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)], dim=-1),  # row 1
        torch.stack([    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=-1),  # row 2
    ], dim=-2)  # (B, 3, 3)

    return R

@torch.jit.script
def quat_derivative(q: Tensor, w: Tensor) -> Tensor:
    """
    Quaternion kinematic equation: q_dot = 0.5 * q ⊗ [0, w]

    Args:
        q : (B, 4)  [w, x, y, z]
        w : (B, 3)  angular velocity in body frame

    Returns:
        q_dot : (B, 4)
    """
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    q_dot = 0.5 * torch.stack([
        -qx*wx - qy*wy - qz*wz,
         qw*wx + qy*wz - qz*wy,
         qw*wy - qx*wz + qz*wx,
         qw*wz + qx*wy - qy*wx,
    ], dim=-1)

    return q_dot

@torch.jit.script
def integrate_euler(
    dt: float,
    p: Tensor, v: Tensor, q: Tensor, w: Tensor,
    p_dot: Tensor, v_dot: Tensor, q_dot: Tensor, w_dot: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    p_next = p + p_dot * dt
    v_next = v + v_dot * dt
    q_next = F.normalize(q + q_dot * dt, dim=-1)               # renormalize quaternion
    w_next = w + w_dot * dt
    return p_next, v_next, q_next, w_next