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

        self.g = torch.tensor([0.0, 0.0, -9.81], device=self.device)

        # FIX: All scalar parameters stored as shape (1,) tensors, not scalars.
        # This makes them consistent with the (N,) shape from set_parameters(),
        # and ensures .unsqueeze(-1) always produces (N, 1) or (1, 1) — never ().
        self.m = torch.tensor(
            [cfg.dynamics.mass.nominal], device=self.device
        )                                                           # (1,)
        self.arm_length = torch.tensor(
            [cfg.dynamics.arm_length.nominal], device=self.device
        )                                                           # (1,)
        self.arm_angle = (
            torch.tensor([cfg.dynamics.arm_angle.nominal], device=self.device)
            * torch.pi / 180
        )                                                           # (1,)

        # FIX: J stored as (1, 3) so it broadcasts correctly with batched (B, 3)
        # and is consistent with (N, 3) from set_parameters().
        self.J = torch.tensor(
            [[
                cfg.dynamics.inertia.xx.nominal,
                cfg.dynamics.inertia.yy.nominal,
                cfg.dynamics.inertia.zz.nominal,
            ]],
            device=self.device,
        )                                                           # (1, 3)
        self.km = torch.tensor(
            [cfg.dynamics.km.nominal], device=self.device
        )                                                           # (1,)

        # FIX: Cache allocation matrix — rebuilt only when parameters change.
        self._alloc_matrix: Tensor | None = None
        self._rebuild_alloc_matrix()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_alloc_matrix(self):
        """
        Build and cache the (N, 4, 4) wrench allocation matrix A.
        Maps motor thrusts f ∈ ℝ⁴ → wrench W = [Fz, τx, τy, τz].

        Called once at init and again after every set_parameters() call.
        """
        s  = torch.sin(self.arm_angle)   # (N,) or (1,)
        c  = torch.cos(self.arm_angle)
        l  = self.arm_length
        km = self.km

        row0 = torch.ones(len(self.m), 4, device=self.device)      # (N, 4)
        row1 = torch.stack([ l*s,  l*s, -l*s, -l*s], dim=-1)      # (N, 4)
        row2 = torch.stack([-l*c,  l*c,  l*c, -l*c], dim=-1)      # (N, 4)
        row3 = torch.stack([ km,  -km,   km,  -km ], dim=-1)      # (N, 4)

        self._alloc_matrix = torch.stack(
            [row0, row1, row2, row3], dim=1
        )                                                           # (N, 4, 4)

    # ------------------------------------------------------------------
    # Derivative computations
    # ------------------------------------------------------------------

    def _translational_deriv(
        self,
        v:  Tensor,
        q:  Tensor,
        Fz: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (p_dot, v_dot).

        Args:
            v  : (B, 3)
            q  : (B, 4)
            Fz : (B,)
        """
        R = quat_to_rotmat(q)                                       # (B, 3, 3)

        thrust_body = torch.stack(
            [torch.zeros_like(Fz), torch.zeros_like(Fz), Fz], dim=-1
        ).unsqueeze(-1)                                             # (B, 3, 1)
        thrust_world = torch.bmm(R, thrust_body).squeeze(-1)       # (B, 3)

        # FIX: self.m is now always (N,) or (1,), so unsqueeze(-1) → (N, 1)
        v_dot = thrust_world / self.m.unsqueeze(-1) + self.g       # (B, 3)

        return v, v_dot   # p_dot = v

    def _rotational_deriv(
        self,
        q:   Tensor,
        w:   Tensor,
        tau: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (q_dot, w_dot).

        Args:
            q   : (B, 4)
            w   : (B, 3)
            tau : (B, 3)

        Note: self.J is (N, 3) or (1, 3) — broadcasts correctly with (B, 3).
              J parameter removed from signature; always uses self.J.
        """
        J     = self.J                                              # (N, 3)
        J_inv = 1.0 / J
        w_dot = J_inv * (tau - torch.linalg.cross(w, J * w))      # (B, 3)
        q_dot = quat_derivative(q, w)                              # (B, 4)
        return q_dot, w_dot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Integrate one time-step forward.

        Args:
            state  : (B, 13)  [p(3), v(3), q(4), w(3)]
            action : (B,  4)  per-motor thrust commands

        Returns:
            next_state : (B, 13)
        """
        p = state[..., 0:3]
        v = state[..., 3:6]
        q = state[..., 6:10]
        w = state[..., 10:13]

        # FIX: Use cached allocation matrix instead of rebuilding every step.
        W   = torch.bmm(self._alloc_matrix, action.unsqueeze(-1)).squeeze(-1)
        Fz  = W[..., 0]      # (B,)
        tau = W[..., 1:4]    # (B, 3)

        p_dot, v_dot = self._translational_deriv(v, q, Fz)
        q_dot, w_dot = self._rotational_deriv(q, w, tau)  # FIX: J no longer passed as arg

        # FIX: Removed debug print statements (these severely hurt loop performance)
        p_next, v_next, q_next, w_next = integrate_euler(
            self.dt, p, v, q, w, p_dot, v_dot, q_dot, w_dot
        )

        return torch.cat([p_next, v_next, q_next, w_next], dim=-1)

    def set_parameters(self, params: QuadrotorParams):
        """
        Apply randomized parameters. Rebuilds the allocation matrix cache.

        Args:
            params : QuadrotorParams with per-env tensors of shape (N, ...)
        """
        self.m          = params.mass           # (N,)
        self.arm_length = params.arm_length     # (N,)
        self.arm_angle  = params.arm_angle      # (N,)
        self.J          = params.J              # (N, 3)
        self.km         = params.km             # (N,)

        # FIX: Rebuild cache after parameters change
        self._rebuild_alloc_matrix()

    def get_parameters(self) -> dict:
        """Return the current dynamics parameters."""
        return {
            'mass':       self.m,
            'arm_length': self.arm_length,
            'arm_angle':  self.arm_angle,
            'inertia':    self.J,
            'km':         self.km,   # FIX: km was missing from original
        }