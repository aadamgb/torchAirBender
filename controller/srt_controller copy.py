# controller/srt_controller.py
# SRT = Square Root Throttle, or whatever your PMM maps to —
# just a direct [0,1] sigmoid output scaled to [0, max_thrust]

import torch
from torch import Tensor
from omegaconf import DictConfig
from controller.base_controller import BaseController


class SRTController(BaseController):

    def __init__(
        self,
        hover_thrust: Tensor,
        hover_ratio: float = 2.0,
        min_ratio: float = 0.0,
    ):
        self._hover_ratio = hover_ratio
        self._min_ratio   = min_ratio

        self.update_hover(hover_thrust)

    def update_hover(self, hover_thrust: Tensor):
        """Call this after every re-randomization."""
        self._hover = hover_thrust
        self._t_max = hover_thrust * self._hover_ratio
        self._t_min = hover_thrust * self._min_ratio

    def __call__(self, raw: Tensor) -> Tensor:
        # raw: (N,4) in [0,1]

        hover = self._hover.unsqueeze(1)
        t_min = self._t_min.unsqueeze(1)
        t_max = self._t_max.unsqueeze(1)

        lower = t_min + (raw / 0.5) * (hover - t_min)
        upper = hover + ((raw - 0.5) / 0.5) * (t_max - hover)

        thrust = torch.where(raw <= 0.5, lower, upper)

        return thrust
    


class SRTController_old(BaseController):
    """
    Minimal controller for the point-mass model.
    Maps sigmoid policy output -> per-motor thrust [N].

    Future extensions:
        - Motor lag / first-order dynamics
        - Drag model
        - RPM -> thrust curve (nonlinear)
    """

    def __init__(self, cfg: DictConfig):
        self.max_thrust = cfg.dynamics.max_thrust   

    def __call__(self, raw: Tensor) -> Tensor:
        # raw: (N, 4) in range (0, 1) from Sigmoid
        return raw * self.max_thrust           # (N, 4) in (0, max_thrust)
    



class CTBRController(BaseController):
    """
    Collective Thrust and Body Rates (CTBR) controller.

    The policy outputs a 4-dimensional command:
        raw[:, 0]   : collective thrust  in [0, 1]  → scaled to [t_min, t_max]
        raw[:, 1:4] : body rates [wx, wy, wz] in [-1, 1] → scaled to [-w_max, w_max]

    The controller then solves for per-motor thrusts using the pseudo-inverse
    of the allocation matrix:

        [Fz, tau_x, tau_y, tau_z]^T = A @ [f1, f2, f3, f4]^T
        => motors = A_pinv @ [Fz, tau_x, tau_y, tau_z]^T

    where torques are produced by a P-controller on the body-rate error:
        tau = J * (w_des - w_current) / dt   (single-step rate control)

    Args:
        hover_thrust : (N,)  per-env hover thrust per motor [N]
        alloc_matrix : (N, 4, 4)  allocation matrix from QuadrotorDynamics
        J            : (N, 3)     diagonal inertia [kg·m²]
        dt           : float      simulation timestep [s]
        hover_ratio  : float      t_max = hover_thrust * hover_ratio
        min_ratio    : float      t_min = hover_thrust * min_ratio
        w_max        : float      max body rate [rad/s]
        kp_rate      : float      P-gain for body-rate controller
    """

    def __init__(
        self,
        hover_thrust: Tensor,        # (N,)
        alloc_matrix: Tensor,        # (N, 4, 4)
        J:            Tensor,        # (N, 3)
        dt:           float,
        hover_ratio:  float = 2.0,
        min_ratio:    float = 0.0,
        w_max:        float = 6.0,   # [rad/s]
        kp_rate:      float = 1.0,
    ):
        self._dt      = dt
        self._w_max   = w_max
        self._kp_rate = kp_rate

        self._hover_ratio = hover_ratio
        self._min_ratio   = min_ratio

        # Store inertia for rate-to-torque conversion
        self._J = J  # (N, 3)

        # Precompute pseudo-inverse of the allocation matrix: (N, 4, 4)
        self._alloc_pinv = torch.linalg.pinv(alloc_matrix)  # (N, 4, 4)

        self.update_hover(hover_thrust)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_hover(self, hover_thrust: Tensor):
        """Call this after every re-randomization to refresh thrust bounds."""
        self._hover  = hover_thrust                          # (N,)
        self._t_max  = hover_thrust * self._hover_ratio      # (N,)
        self._t_min  = hover_thrust * self._min_ratio        # (N,)
        # print(self._hover)

    def update_parameters(
        self,
        hover_thrust: Tensor,
        alloc_matrix: Tensor,
        J:            Tensor,
    ):
        """
        Full parameter refresh — call after set_parameters() on the dynamics.

        Args:
            hover_thrust : (N,)
            alloc_matrix : (N, 4, 4)
            J            : (N, 3)
        """
        self._J          = J
        self._alloc_pinv = torch.linalg.pinv(alloc_matrix)
        self.update_hover(hover_thrust)

    def __call__(self, raw: Tensor, w: Tensor) -> Tensor:
        """
        Map policy output + current body rates to per-motor thrust commands.

        Args:
            raw : (N, 4)  policy output in [0, 1] for thrust, [-1, 1] for rates
            w   : (N, 3)  current body rates [rad/s] from state[..., 10:13]

        Returns:
            motors : (N, 4)  per-motor thrust [N], clamped to [t_min, t_max]
        """
        N = raw.shape[0]

        # ── 1. Decode collective thrust ──────────────────────────────────
        t_min  = self._t_min.unsqueeze(1)   # (N, 1)
        t_max  = self._t_max.unsqueeze(1)   # (N, 1)
        hover  = self._hover.unsqueeze(1)   # (N, 1)

        c      = raw[:, 0:1]                # (N, 1) in [0, 1]
        lower  = t_min + (c / 0.5) * (hover - t_min)
        upper  = hover + ((c - 0.5) / 0.5) * (t_max - hover)
        Fz     = torch.where(c <= 0.5, lower, upper).squeeze(1)   # (N,)  [N]

        # ── 2. Decode desired body rates ─────────────────────────────────
        # w_des = raw[:, 1:4] * self._w_max   # (N, 3)  [rad/s]

        # need to remap this since the current output is sigmoid from 0 to 1 and body rates can be negative...
        w_des = (raw[:, 1:4] * 2.0 - 1.0) * self._w_max   # [0,1] → [-1,1] → [-w_max, w_max]

        # ── 3. Rate P-controller → desired torques ───────────────────────
        #   tau = J * kp * (w_des - w) / dt
        #   Using single-step Euler approximation of angular acceleration.
        w_err = w_des - w                                        # (N, 3)
        tau   = self._J * (self._kp_rate * w_err / self._dt)    # (N, 3)  [N·m]

        # ── 4. Pack wrench vector [Fz, tau_x, tau_y, tau_z] ─────────────
        wrench = torch.cat([Fz.unsqueeze(-1), tau], dim=-1)      # (N, 4)

        # ── 5. Invert allocation matrix → per-motor thrusts ─────────────
        #   rotors thrust = A_pinv @ wrench   (batched)
        srt = torch.bmm(
            self._alloc_pinv,
            wrench.unsqueeze(-1)
        ).squeeze(-1)                                             # (N, 4)


        return srt, wrench