import torch
import torch.nn.functional as F
from torch import Tensor


def acc_to_quat2(acc_ref: Tensor, v_ref: Tensor, g: float = 9.81) -> Tensor:
    """
    Computes desired quaternion from reference acceleration and velocity.
    Heading is derived from v_ref (projected onto XY plane).
    Builds R_des the same way as the geometric controller.

    Args:
        acc_ref : (N, 3)  reference acceleration
        v_ref   : (N, 3)  reference velocity (used for heading)
    Returns:
        q_des   : (N, 4)  [w, x, y, z]
    """
    # heading from velocity reference, projected onto XY plane
    v_heading       = v_ref.clone()
    v_heading[:, 2] = 0.0
    b1d             = F.normalize(v_heading, dim=-1)                    # (N, 3)

    # thrust direction from acceleration + gravity
    gravity    = torch.tensor([0.0, 0.0, g], device=acc_ref.device)
    A          = acc_ref + gravity                                      # (N, 3)

    b3d        = F.normalize(A, dim=-1)                                 # (N, 3)
    b2d        = F.normalize(torch.linalg.cross(b3d, b1d), dim=-1)     # (N, 3)
    b1d_ort    = torch.linalg.cross(b2d, b3d)                          # (N, 3)

    R_des      = torch.stack([b1d_ort, b2d, b3d], dim=-1)              # (N, 3, 3)

    return rotmat_to_quat(R_des)                                        # (N, 4) 

def acc_to_quat(acc_ref: Tensor, g: float = 9.81) -> Tensor:
    """
    Computes desired quaternion from reference acceleration.
    Desired thrust direction = acc_ref + gravity_vector.

    Args:
        acc_ref : (N, 3)
    Returns:
        q_des   : (N, 4)  [w, x, y, z]
    """
    gravity = torch.tensor([0.0, 0.0, g], device=acc_ref.device)   # acceleration compensation
    thrust_dir = acc_ref + gravity                                     # (N, 3)
    thrust_dir = torch.nn.functional.normalize(thrust_dir, dim=-1)    # (N, 3)

    # Body z-axis in world frame should align with thrust_dir
    # Rotation from world z [0,0,1] to thrust_dir
    world_z = torch.zeros_like(thrust_dir)
    world_z[:, 2] = 1.0

    # Axis of rotation = cross(world_z, thrust_dir)
    axis = torch.linalg.cross(world_z, thrust_dir)                    # (N, 3)
    # Angle: cos(theta) = dot(world_z, thrust_dir)
    dot  = (world_z * thrust_dir).sum(dim=-1, keepdim=True)           # (N, 1)

    # Quaternion: w = cos(theta/2), xyz = sin(theta/2) * axis_normalized
    # Using half-angle: w = sqrt((1 + cos)/2), |xyz| = sqrt((1 - cos)/2)
    w   = torch.sqrt(torch.clamp((1.0 + dot) / 2.0, min=1e-6))       # (N, 1)
    xyz = torch.nn.functional.normalize(axis, dim=-1) * torch.sqrt(
        torch.clamp((1.0 - dot) / 2.0, min=0.0)
    )                                                                  # (N, 3)

    return torch.cat([w, xyz], dim=-1)                                 # (N, 4)

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
def rotmat_to_quat(R: Tensor) -> Tensor:
    """
    Rotation matrix to quaternion [w, x, y, z] — Shepperd method.
    Args:
        R : (N, 3, 3)
    Returns:
        q : (N, 4)
    """
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]                  # (N,)

    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2        # 4w
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s

    return F.normalize(q, dim=-1)

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