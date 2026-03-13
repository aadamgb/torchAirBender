# controller/controllers.py

import torch
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from utils.math import quat_to_rotmat


# ===========================================================
# JIT-compiled helpers
# ===========================================================

@torch.jit.script
def _map_TTWR(
    c:     Tensor,   # (N, 1) in [0, 1]
    t_min: Tensor,   # (N, 1)
    hover: Tensor,   # (N, 1)
    t_max: Tensor,   # (N, 1)
) -> Tensor:
    """
    Piecewise-linear thrust mapping:
        c in [0.0, 0.5] → [t_min, hover]
        c in [0.5, 1.0] → [hover, t_max]

    Ensures sigmoid(0) = 0.5 maps exactly to hover thrust.

    Returns:
        Fz : (N,)
    """
    lower = t_min + (c / 0.5) * (hover - t_min)
    upper = hover + ((c - 0.5) / 0.5) * (t_max - hover)
    return torch.where(c <= 0.5, lower, upper).squeeze(1)   # (N,)


@torch.jit.script
def _vee(S: Tensor) -> Tensor:
    """
    Vee operator — extracts the 3-vector from a (B, 3, 3) skew-symmetric matrix.

        S = [  0   -v2   v1 ]
            [  v2   0   -v0 ]
            [ -v1   v0   0  ]
    """
    return torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)


@torch.jit.script
def _attitude_error(R: Tensor, R_des: Tensor) -> Tensor:
    """
    Geometric attitude error on SO(3) — Lee et al. (2010), eq. (10).

        e_R = 0.5 * vee( R_des^T R  -  R^T R_des )

    Args:
        R     : (B, 3, 3)  current rotation matrix
        R_des : (B, 3, 3)  desired rotation matrix

    Returns:
        e_R : (B, 3)
    """
    Rt_Rdes  = torch.bmm(R.transpose(-1, -2), R_des)
    Rdes_Rt  = torch.bmm(R_des.transpose(-1, -2), R)
    return _vee(Rdes_Rt - Rt_Rdes) * 0.5


@torch.jit.script
def _geometric_torques(
    R:     Tensor,   # (B, 3, 3)
    R_des: Tensor,   # (B, 3, 3)
    w:     Tensor,   # (B, 3)
    w_des: Tensor,   # (B, 3)
    J:     Tensor,   # (N, 3)
    kR:    float,
    kw:    float,
) -> Tensor:
    """
    SO(3) geometric PD torque law — Lee et al. (2010), eq. (15).

        tau = -kR * e_R  -  kw * e_w  +  w × (J w)

    The gyroscopic term w × Jw cancels the Coriolis effect so
    the PD gains don't need to fight it.
    """
    e_R  = _attitude_error(R, R_des)
    e_w  = w - w_des
    gyro = torch.linalg.cross(w, J * w)
    return -kR * e_R  -  kw * e_w  +  gyro


# ===========================================================
# Base controller
# ===========================================================

class BaseController(ABC):
    """
    Shared state and helpers for all controllers.

    All subclasses receive:
        - Piecewise-linear thrust decoding via _map_TTWR()
        - Allocation matrix pseudo-inverse for wrench → motors
        - update_hover() / update_parameters() lifecycle methods
    """

    def __init__(
        self,
        hover_thrust : Tensor,   # (N,)
        alloc_matrix : Tensor,   # (N, 4, 4)
        hover_ratio  : float = 2.0,
        min_ratio    : float = 0.0,
    ):
        self._hover_ratio = hover_ratio
        self._min_ratio   = min_ratio
        self._alloc_pinv  = torch.linalg.pinv(alloc_matrix)   # (N, 4, 4)
        self.update_hover(hover_thrust)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update_hover(self, hover_thrust: Tensor):
        """Refresh thrust bounds — call after every re-randomization."""
        self._hover = hover_thrust                          # (N,)
        self._t_max = hover_thrust * self._hover_ratio      # (N,)
        self._t_min = hover_thrust * self._min_ratio        # (N,)

    def update_parameters(self, hover_thrust: Tensor, alloc_matrix: Tensor):
        """Refresh alloc pinv + hover — call alongside quadrotor.set_parameters()."""
        self._alloc_pinv = torch.linalg.pinv(alloc_matrix)
        self.update_hover(hover_thrust)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _thrust_bounds(self) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (t_min, hover, t_max) each (N, 1) — ready for broadcasting."""
        return (
            self._t_min.unsqueeze(1),
            self._hover.unsqueeze(1),
            self._t_max.unsqueeze(1),
        )

    def _wrench_to_motors(self, wrench: Tensor) -> Tensor:
        """
        Inverts the allocation matrix and clamps to physical limits.

        Args:
            wrench : (N, 4)  [Fz, tau_x, tau_y, tau_z]

        Returns:
            motors : (N, 4)  per-motor thrust [N]
        """
        motors = torch.bmm(self._alloc_pinv, wrench.unsqueeze(-1)).squeeze(-1)
        return motors

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        """Returns (motors, wrench)."""
        ...


# ===========================================================
# SRT Controller
# ===========================================================

class SRTController(BaseController):
    """
    Square-Root Throttle controller.

    Maps sigmoid policy output (N, 4) directly to per-motor thrusts,
    with the piecewise-linear hover-centred scaling.

    Policy input : (N, 4) in [0, 1]  — one value per motor
    Output       : (N, 4) per-motor thrust [N]
    """

    def __call__(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        t_min, hover, t_max = self._thrust_bounds()
        motors = _map_TTWR(raw, t_min, hover, t_max)   # reuse scalar path
        # For SRT every channel is a motor directly — no wrench inversion needed.
        # We reconstruct a dummy wrench (all-zeros tau) for API consistency.
        lower  = t_min + (raw / 0.5) * (hover - t_min)
        upper  = hover + ((raw - 0.5) / 0.5) * (t_max - hover)
        motors = torch.where(raw <= 0.5, lower, upper)      # (N, 4)
        wrench = torch.zeros(raw.shape[0], 4, device=raw.device)
        return motors, wrench


# ===========================================================
# CTBR Controller
# ===========================================================

class CTBRController(BaseController):
    """
    Collective Thrust and Body Rates controller.

    Policy input:
        raw[:, 0]   : collective thrust in [0, 1]
        raw[:, 1:4] : body rates [wx, wy, wz] in [0, 1] → remapped to [-w_max, w_max]

    Args:
        hover_thrust : (N,)       total hover thrust [N]  (= mg)
        alloc_matrix : (N, 4, 4)
        J            : (N, 3)     diagonal inertia [kg·m²]
        dt           : float      simulation timestep [s]
        hover_ratio  : float
        min_ratio    : float
        w_max        : float      max body rate [rad/s]
        kp_rate      : float      P-gain on body-rate error
    """

    def __init__(
        self,
        hover_thrust : Tensor,
        alloc_matrix : Tensor,
        J            : Tensor,
        dt           : float,
        hover_ratio  : float = 2.0,
        min_ratio    : float = 0.0,
        w_max        : float = 6.0,
        kp_rate      : float = 1.0,
    ):
        super().__init__(hover_thrust, alloc_matrix, hover_ratio, min_ratio)
        self._J       = J
        self._dt      = dt
        self._w_max   = w_max
        self._kp_rate = kp_rate

    def update_parameters(self, hover_thrust: Tensor, alloc_matrix: Tensor, J: Tensor):
        super().update_parameters(hover_thrust, alloc_matrix)
        self._J = J

    def __call__(self, raw: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            raw : (N, 4)  policy output
            w   : (N, 3)  state[..., 10:13]

        Returns:
            motors : (N, 4)
            wrench : (N, 4)  [Fz, tau_x, tau_y, tau_z]
        """
        t_min, hover, t_max = self._thrust_bounds()

        Fz    = _map_TTWR(raw[:, 0:1], t_min, hover, t_max)          # (N,)
        w_des = (raw[:, 1:4] * 2.0 - 1.0) * self._w_max                  # (N, 3)
        tau   = self._J * (self._kp_rate * (w_des - w) / self._dt)        # (N, 3)

        wrench = torch.cat([Fz.unsqueeze(-1), tau], dim=-1)               # (N, 4)
        return self._wrench_to_motors(wrench), wrench


# ===========================================================
# Geometric Controller
# ===========================================================
class GeometricClassic:
    # ⚠️ Neglecting Gyroscopic term  Ω × JΩ and  Feed-forward: -J(Ω̂ R^T R_des Ω_des) ⚠️
    def __init__(self, alloc_matrix, m, g=9.81, kp=200.0, kv=20.0, kR=120.81, kw=3.5):
        self.alloc_inv = torch.linalg.pinv(alloc_matrix)
        self.m  = m
        self.g  = g
        self.kp = kp
        self.kv = kv
        self.kR = kR
        self.kw = kw

    def __call__(self, state, p_ref, v_ref, a_ref, b1d):
        p, v, q, w = state[:, 0:3], state[:, 3:6], state[:, 6:10], state[:, 10:13]
        R = quat_to_rotmat(q)

        e3      = torch.zeros_like(a_ref); e3[:, 2] = 1.0
        A       = -self.kp * (p - p_ref) - self.kv * (v - v_ref) + self.m * (self.g * e3 + a_ref)

        b3d     = F.normalize(A, dim=-1)
        b2d     = F.normalize(torch.linalg.cross(b3d, b1d), dim=-1)
        R_des   = torch.stack([torch.linalg.cross(b2d, b3d), b2d, b3d], dim=-1)

        RdTR    = torch.bmm(R_des.transpose(-1, -2), R)
        skew    = RdTR - RdTR.transpose(-1, -2)
        eR      = 0.5 * torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)
        eW      = w - torch.bmm(R.transpose(-1, -2), R_des).bmm(torch.zeros_like(w).unsqueeze(-1)).squeeze(-1)

        Fz      = (A * R[:, :, 2]).sum(dim=-1, keepdim=True)
        tau     = -self.kR * eR - self.kw * eW
        wrench  = torch.cat([Fz, tau], dim=-1)

        return torch.bmm(self.alloc_inv, wrench.unsqueeze(-1)).squeeze(-1)


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
        
