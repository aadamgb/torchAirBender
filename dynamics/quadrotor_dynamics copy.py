import torch
from torch import Tensor
from omegaconf import DictConfig
from utils.randomize import QuadrotorParams

from utils.math import *


#=====================================================
"""
    Motor layout (top view, Z-up ENU):
                    ^ b2
                    |
           (2) CW   |   (1) CCW
               \    |    /
                \   |   /
                 \  |  /
        ----------- + ----------> b1
                 /  |  \
                /   |   \
               /    |    \
           (3) CCW  |   (4) CW
                    |
"""
#=====================================================
class QuadrotorDynamics:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dt = cfg.dt
        # self.g = 9.81
        self.g = torch.tensor([0.0, 0.0, -9.81], device=self.device)

        self.m = torch.tensor(cfg.dynamics.mass.nominal, device=self.device)
        self.arm_length = torch.tensor(cfg.dynamics.arm_length.nominal, device=self.device)
        self.arm_angle = torch.tensor(cfg.dynamics.arm_angle.nominal, device=self.device) * torch.pi / 180   # Converting to radians
        self.J = torch.tensor(
            [
                cfg.dynamics.inertia.xx.nominal,
                cfg.dynamics.inertia.yy.nominal,
                cfg.dynamics.inertia.zz.nominal,
            ],
            device=self.device,
        )
        self.km = torch.tensor(cfg.dynamics.km.nominal, device=self.device)


    def _translational_deriv(
            self,
            v:      Tensor,
            q:      Tensor,
            Fz:     Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Point Mass Model
        Returns (p_dot, v_dot).

        Args:
            v    : (B, 3)
            q    : (B, 4)
            Fz   : (B,)
        """

        R = quat_to_rotmat(q)   # (B, 3, 3)

        # Thrust vector in world frame
        thrust_body  = torch.stack(
            [torch.zeros_like(Fz), torch.zeros_like(Fz), Fz], dim=-1
        ).unsqueeze(-1)                                     # (B, 3, 1)
        thrust_world = torch.bmm(R, thrust_body).squeeze(-1)  # (B, 3)

        v_dot = thrust_world / self.m.unsqueeze(-1) + self.g

        return v, v_dot   # p_dot = v
    
    def _rotational_deriv(
        self,
        q:   Tensor,
        w:   Tensor,
        tau: Tensor,
        J:   Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (q_dot, w_dot).

        Args:
            q   : (B, 4)
            w   : (B, 3)
            tau : (B, 3)
            J   : (B, 3)  diagonal inertia
        """
        J_inv = 1.0 / J                                     # (B, 3)
        w_dot = J_inv * (tau - torch.linalg.cross(w, J * w))  # (B, 3)
        q_dot = quat_derivative(q, w)                       # (B, 4)
        return q_dot, w_dot

    def step(self, 
            state: Tensor,
            action: Tensor):
        
        B = state.shape[0]

        p     = state[..., 0:3]
        v     = state[..., 3:6]
        q     = state[..., 6:10]
        w     = state[..., 10:13]
        
        # Compute sine and cosine of arm angle
        s = torch.sin(self.arm_angle)
        c = torch.cos(self.arm_angle)
        l = self.arm_length
        km = self.km

        # Build the allocation matrix for each env
        row0 = torch.ones(B, 4, device=self.device)
        row1 = torch.stack([ l*s,  l*s, -l*s, -l*s ], dim=1)
        row2 = torch.stack([-l*c,  l*c,  l*c, -l*c ], dim=1)
        row3 = torch.stack([ km,  -km,   km,  -km ], dim=1)

        A = torch.stack([row0, row1, row2, row3], dim=1)

        # Compute wrench aka CTBR
        W = torch.bmm(A, action.unsqueeze(-1)).squeeze(-1)  # (B, 4)
        Fz  = W[..., 0]
        tau = W[..., 1:4]

        p_dot, v_dot = self._translational_deriv(v, q, Fz)
        # print(f"p_dot: {p_dot},\n v_dot: {v_dot}")
        # print(self.m)
        q_dot, w_dot = self._rotational_deriv(q, w, tau, self.J)
        print(f"q_dot: {q_dot}")
        print(f"w_dot: {w_dot}")
        
        p_next, v_next, q_next, w_next = integrate_euler(self.dt,
            p, v, q, w, p_dot, v_dot, q_dot, w_dot
        )

        return torch.cat([p_next, v_next, q_next, w_next], dim=-1)



    def set_parameters(self, params: QuadrotorParams):
        """
        Apply randomized parameters to the dynamics.
        params: QuadrotorParams NamedTuple with shape (N, ...)
        """
        self.m = params.mass                    # (N,)
        self.arm_length = params.arm_length     # (N,)
        self.arm_angle = params.arm_angle       # (N,)
        self.J = params.J                       # (N, 3)
        self.km = params.km       # (N,)


    def get_parameters(self):
        """
        Get the current dynamics parameters.
        """
        return {
            'mass': self.m,
            'arm_length': self.arm_length,
            'arm_angle': self.arm_angle,
            'inertia': self.J,
        }