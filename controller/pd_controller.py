import torch
from torch import Tensor
from omegaconf import DictConfig

from utils.randomize import QuadrotorParams


class ProportionalDerivative:
    """
    Cascade PD controller for quadrotor position control.

    Outer loop (position → desired thrust vector):
        e_pos  = target_pos - pos
        e_vel  = -vel
        a_des  = Kp_pos * e_pos + Kd_pos * e_vel + [0, 0, g]   (ENU, z-up)

    Inner loop (attitude → motor thrusts):
        Compute desired z-body axis from a_des, derive roll/pitch error,
        then PD on attitude error → collective thrust + torques →
        distribute to 4 motors via allocation matrix.

    All tensors are (num_envs, ...) on the same device.

    State vector convention:
        state[0:3]   = pos   (ENU)
        state[3:6]   = vel   (ENU)
        state[6:10]  = quat  [w, x, y, z]
        state[10:13] = omega (body frame)
    """

    def __init__(
        self,
        cfg:    DictConfig,
        params: QuadrotorParams,
        device: torch.device,
        dtype:  torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype  = dtype
        self.params = params
        self.g      = cfg.gravity if hasattr(cfg, "gravity") else 9.81

        N = params.mass.shape[0]

        # ------------------------------------------------------------------
        # Gains — per environment, shape (N,)
        # ------------------------------------------------------------------
        pc = cfg.gains.pos
        ac = cfg.gains.att

        self.Kp_pos = torch.full((N,), pc.kp, device=device, dtype=dtype)
        self.Kd_pos = torch.full((N,), pc.kd, device=device, dtype=dtype)

        self.Kp_att = torch.full((N,), ac.kp, device=device, dtype=dtype)
        self.Kd_att = torch.full((N,), ac.kd, device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # Motor allocation matrix  (N, 4, 4)
        # Maps [F_total, tau_x, tau_y, tau_z] -> motor thrusts [f1..f4]
        # Inverse of the standard allocation matrix for an X-frame.
        # ------------------------------------------------------------------
        self._build_allocation(params)

    # ------------------------------------------------------------------
    # Allocation matrix
    # ------------------------------------------------------------------

    def _build_allocation(self, params: QuadrotorParams):
        """
        Build per-env inverse allocation matrix (N, 4, 4).

        Standard X-frame allocation (body frame):
            F_total = f1 + f2 + f3 + f4
            tau_x   = l*(-f1 + f2 + f3 - f4) * sin(alpha)
            tau_y   = l*(-f1 - f2 + f3 + f4) * cos(alpha)
            tau_z   = km*(f1 - f2 + f3 - f4)       [reaction torques]

        where l = arm_length, alpha = arm_angle, km = torque/thrust ratio.
        """
        N  = params.mass.shape[0]
        l  = params.arm_length          # (N,)
        a  = params.arm_angle           # (N,)
        km = params.km                  # (N,)

        s = torch.sin(a)
        c = torch.cos(a)

        # Allocation matrix A s.t.  A @ [f1,f2,f3,f4]^T = [F, tx, ty, tz]^T
        # Shape: (N, 4, 4)
        ones = torch.ones(N, device=self.device, dtype=self.dtype)
        A = torch.stack([
            torch.stack([ ones,        ones,        ones,        ones      ], dim=-1),
            torch.stack([ l*s,         l*s,         -l*s,        -l*s       ], dim=-1),
            torch.stack([-l*c,         l*c,         l*c,         -l*c       ], dim=-1),
            torch.stack([ km,         -km,           km,         -km        ], dim=-1),
        ], dim=1)   # (N, 4, 4)

        # Pseudo-inverse (exact inverse since A is square and full-rank)
        self._alloc_inv = torch.linalg.inv(A)   # (N, 4, 4)

    # ------------------------------------------------------------------
    # Quaternion helpers  (all batched, quaternion = [w, x, y, z])
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_rotmat(q: Tensor) -> Tensor:
        """q: (N, 4) [w,x,y,z] → R: (N, 3, 3)"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
                2*(x*y + w*z),  1 - 2*(x*x + z*z),  2*(y*z - w*x),
                2*(x*z - w*y),  2*(y*z + w*x),       1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)
        return R

    @staticmethod
    def _vee(M: Tensor) -> Tensor:
        """Extract axial vector from skew-symmetric matrix (N,3,3) → (N,3)"""
        return torch.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], dim=-1)

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def compute(self, state: Tensor, target_pos: Tensor) -> Tensor:
        """
        Args:
            state      : (N, 13)  current state
            target_pos : (N,  3)  desired position in ENU

        Returns:
            actions    : (N,  4)  motor thrusts [N], clamped to [0, inf)
        """
        pos   = state[:, 0:3]    # (N, 3)
        vel   = state[:, 3:6]    # (N, 3)
        quat  = state[:, 6:10]   # (N, 4)  [w,x,y,z]
        omega = state[:, 10:13]  # (N, 3)  body-frame angular velocity

        N = pos.shape[0]
        g_vec = torch.zeros(N, 3, device=self.device, dtype=self.dtype)
        g_vec[:, 2] = self.g     # ENU: z-up gravity compensation

        # ------------------------------------------------------------------
        # Outer loop — desired acceleration
        # ------------------------------------------------------------------
        e_pos = target_pos - pos                           # (N, 3)
        e_vel = -vel                                       # (N, 3)

        # Kp/Kd broadcast: (N,1) * (N,3)
        a_des = (self.Kp_pos.unsqueeze(1) * e_pos
               + self.Kd_pos.unsqueeze(1) * e_vel
               + g_vec)                                    # (N, 3)

        # Desired collective thrust magnitude
        mass = self.params.mass                            # (N,)
        F_des = mass * torch.norm(a_des, dim=-1)           # (N,)

        # ------------------------------------------------------------------
        # Inner loop — attitude error
        # ------------------------------------------------------------------
        R = self._quat_to_rotmat(quat)                     # (N, 3, 3)

        # Current body z-axis in world frame
        b3 = R[:, :, 2]                                    # (N, 3)

        # Desired body z-axis (normalised desired acceleration)
        a_des_norm = torch.norm(a_des, dim=-1, keepdim=True).clamp(min=1e-6)
        b3_des = a_des / a_des_norm                        # (N, 3)

        # Rotation error: e_R = 0.5 * vee(R_des^T R - R^T R_des)
        # Simplified for small angles: cross product of current and desired z
        e_R = torch.cross(b3, b3_des, dim=-1)              # (N, 3)

        # Angular velocity error (drive omega to zero)
        e_omega = -omega                                    # (N, 3)

        # Desired torques
        J = self.params.J                                  # (N, 3)
        tau_des = (self.Kp_att.unsqueeze(1) * e_R
                 + self.Kd_att.unsqueeze(1) * e_omega
                 ) * J                                     # (N, 3)

        # ------------------------------------------------------------------
        # Motor allocation  [F, tx, ty, tz] → [f1, f2, f3, f4]
        # ------------------------------------------------------------------
        wrench = torch.stack([F_des,
                               tau_des[:, 0],
                               tau_des[:, 1],
                               tau_des[:, 2]], dim=-1).unsqueeze(-1)   # (N, 4, 1)

        thrusts = (self._alloc_inv @ wrench).squeeze(-1)               # (N, 4)

        # Clamp to non-negative thrust
        thrusts = thrusts.clamp(min=0.0)

        return thrusts