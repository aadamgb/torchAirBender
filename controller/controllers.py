import torch
from torch.functional import F
from utils.math import quat_to_rotmat

class DirectAllocation:
    """Maps wrench to per-motor thrusts by pseudo-inverting the allocation matrix."""
    def __init__(self, alloc_matrix):
        self.alloc_inv = torch.linalg.pinv(alloc_matrix)

    def __call__(self, Fz, tau):
        wrench = torch.cat([Fz, tau], dim=-1)           # (N, 4)
        return torch.bmm(self.alloc_inv, wrench.unsqueeze(-1)).squeeze(-1)

    def update_params(self, alloc_matrix):
        self.alloc_inv = torch.linalg.pinv(alloc_matrix)

class SRT:
    def __init__(self, max_thrust=20.0, min_thrust=0.5):
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust

    def __call__(self, state, raw):
        return raw * (self.max_thrust - self.min_thrust) + self.min_thrust # (N, 4) rotor thrusts

    def update_params(self, alloc_matrix=None, J=None, max_thrust=None, min_thrust=None):
        if alloc_matrix is not None:
            self.alloc_inv = torch.linalg.pinv(alloc_matrix)
        if J is not None:
            self.J = J
        if max_thrust is not None:
            self.max_thrust = max_thrust.reshape(-1, 1) if torch.is_tensor(max_thrust) else max_thrust
        if min_thrust is not None:
            self.min_thrust = min_thrust.reshape(-1, 1) if torch.is_tensor(min_thrust) else min_thrust

class CTBR:
    def __init__(self, allocator, J, max_thrust=20.0, min_thrust=0.5, max_rate=10.0, kp_rate=1.0, dt=0.01):
        self.allocator  = allocator
        self.max_thrust = max_thrust * 4.0
        self.min_thrust = min_thrust * 4.0 # TODO: fix this xd 
        self.max_rate   = max_rate
        self.kp_rate    = kp_rate
        # self.alloc_inv  = torch.linalg.pinv(alloc_matrix)
        self.J          = J
        self.dt         = dt

    def __call__(self, state, raw):
        w = state[:, 10:13]
        Fz    =  raw[:, 0:1] * (self.max_thrust - self.min_thrust) + self.min_thrust
        w_des = (raw[:, 1:4] * 2.0 - 1.0) * self.max_rate
        tau = self.J * (self.kp_rate * (w_des - w) / self.dt)  # (N, 3)
        return self.allocator(Fz, tau)
    
    # def wrench_to_motors(self, Fz, tau):
    #     """
    #     Fz    : (N, 1)  physical thrust [N]
    #     tau   : (N, 3)  physical torques [Nm]
    #     """
    #     wrench = torch.cat([Fz, tau], dim=-1)                   # (N, 4)
    #     return torch.bmm(self.alloc_inv, wrench.unsqueeze(-1)).squeeze(-1)

    def update_params(self, alloc_matrix=None, J=None, max_thrust=None, min_thrust=None):
        if alloc_matrix is not None:
            self.allocator.update_params(alloc_matrix)
        if J is not None:
            self.J = J
        if max_thrust is not None:
            self.max_thrust = max_thrust.reshape(-1, 1) if torch.is_tensor(max_thrust) else max_thrust
        if min_thrust is not None:
            self.min_thrust = min_thrust.reshape(-1, 1) if torch.is_tensor(min_thrust) else min_thrust


class LVYR:
    """
    Linear Velocity + Yaw Rate controller.
    
    Action: (vx_des, vy_des, vz_des, yaw_rate_des) in [0, 1]
    Output: (Fz, wx_des, wy_des, wz_des) wrench for CTBR
    """
    def __init__(
        self,
        allocator,
        m, J, g, 
        kv=1.0,       # linear velocity P gain
        kR=1.0,       # attitude P gain (roll/pitch)
        kw=0.25,
        max_vel=10.0,
        max_yaw_rate=1.0,
        gain_scale = [5.0, 5.0, 2.0]
    ):
        self.allocator    = allocator   
        self.m            = m
        self.J            = J
        self.g            = g
        self.kv           = kv
        self.kR           = kR
        self.kw           = kw
        self.max_vel      = max_vel
        self.max_yaw_rate = max_yaw_rate
        self.gain_scale   = gain_scale

    def update_params(self, alloc_matrix=None, J=None, kv=None, kR=None, kw=None):
        if alloc_matrix is not None:
            self.allocator.update_params(alloc_matrix)
        if kv is not None:
            self.kv = kv
        if kR is not None:
            self.kR = kR
        if kw is not None:
            self.kw = kw

    def __call__(self, state, raw, gains=None):
        """
        state : (N, 13)
        raw   : (N, 4)  all in [0, 1] from policy
        returns (Fz, wx_des, wy_des, wz_des) as (N, 4) — feeds into CTBR
        """

        # Gain scheduling if cm is lvyr_g
        if gains is not None:
            # gains: (N, 3), softplus to keep positive
            gains = F.softplus(gains) * self.gain_scale  + 0.1  # +0.1 to prevent zero gain values
            kv = gains[:, 0:1]   # (N, 1)
            kR = gains[:, 1:2]   # (N, 1)
            kw = gains[:, 2:3]   # (N, 1)
        else:
            kv, kR, kw = self.kv, self.kR, self.kw

        # --- Unpack state ---
        vel  = state[:, 3:6]                                            # (N, 3) vx vy vz
        quat = state[:, 6:10]                                           # (N, 4) w x y z
        w    = state[:, 10:13]                                          # (N, 3) wx wy wz

        R = quat_to_rotmat(quat)                                        # (N, 3, 3)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        # --- Decode actions ---
        v_des       = (raw[:, 0:3] * 2.0 - 1.0) * self.max_vel          # (N, 3) [-max, max]
        yaw_rate_des = (raw[:, 3:4] * 2.0 - 1.0) * self.max_yaw_rate    # (N, 1)

        # --- Desired force vector in world frame ---
        # Proportional velocity error -> desired acceleration
        a_des = kv * (v_des - vel)               # (N, 3)
        # Add gravity compensation on z
        a_des[:, 2] += self.g
        f_des = self.m * a_des                        # (N, 3) world frame force

        b3_des = F.normalize(f_des, dim=-1, eps=1e-6)

        # --- Desired x-body-axis from current yaw heading ---
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw),
                               torch.zeros_like(yaw)], dim=-1)  # (N, 3)

        # b2_des = b3_des x heading, b1_des = b2_des x b3_des
        b2_des = F.normalize(torch.linalg.cross(b3_des, heading), dim=-1, eps=1e-6)
        b1_des = torch.linalg.cross(b2_des, b3_des)

        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)  # (N, 3, 3) column-major

        # --- SO(3) attitude error (vee map of skew-symmetric part) ---
        RdTR   = torch.bmm(R_des.transpose(-1, -2), R)
        skew   = RdTR - RdTR.transpose(-1, -2)                                                # (N, 3, 3) skew-sym
        eR = 0.5 * torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)         # (N, 3)

        RTRd   = torch.bmm(R.transpose(-1, -2), R_des)
        w_des = torch.zeros_like(w)
        w_des[:, 2:3] = yaw_rate_des
        eW = w - torch.bmm(RTRd, w_des.unsqueeze(-1)).squeeze(-1)

        gyro = torch.linalg.cross(w, self.J * w)

        # --- Total thrust magnitude projected onto body z ---
        b3_cur = R[:, :, 2]                                     # (N, 3) current z-axis
        Fz = (f_des * b3_cur).sum(dim=-1, keepdim=True)         # (N, 1)
        
        Fz = F.softplus(Fz)
        tau    = -kR * eR - kw * eW + gyro        # TODO: Think wether to feedforward term...

        return self.allocator(Fz, tau)