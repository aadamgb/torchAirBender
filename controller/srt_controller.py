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

@torch.jit.script
def _compute_desired_thrust(
    p     : Tensor,   # (N, 3)  current position
    v     : Tensor,   # (N, 3)  current velocity
    p_ref : Tensor,   # (N, 3)  reference position
    v_ref : Tensor,   # (N, 3)  reference velocity
    a_ref : Tensor,   # (N, 3)  reference acceleration
    R     : Tensor,   # (N, 3, 3)  current rotation matrix
    m     : Tensor,   # (N,)  mass [kg]
    g     : float,    # scalar gravity [m/s²]
    kp    : float,    # position gain
    kv    : float,    # velocity gain
    t_min : float,    # collective thrust lower bound [N]
    t_max : float,    # collective thrust upper bound [N]
) -> Tensor:
    """
    Computes collective thrust from position/velocity errors — Lee et al. SE(3), eq. (17).

        F_des = -kp * e_p  -  kv * e_v  +  m*g*e3  +  m*a_ref
        Fz    = dot(F_des, R[:, :, 2])     ← project onto current body z-axis

    Args:
        p, v       : (N, 3)    current state
        p_ref, v_ref, a_ref : (N, 3)  reference trajectory
        R          : (N, 3, 3) current rotation matrix
        m          : (N,)      per-env mass
        g          : float     gravity magnitude
        kp, kv     : float     position / velocity gains
        t_min/max  : float     scalar clamp bounds

    Returns:
        Fz : (N,)  collective thrust [N]
    """
    e3    = torch.zeros_like(p)
    e3[:, 2] = 1.0

    F_des = -kp * (p - p_ref) - kv * (v - v_ref) + m.unsqueeze(-1) * g * e3 + m.unsqueeze(-1) * a_ref  # (N, 3)
    body_z = R[:, :, 2]                                                       # (N, 3)  third column of R
    Fz     = (F_des * body_z).sum(dim=-1)                                     # (N,)  dot product

    return Fz.clamp(t_min, t_max)


class GeometricController(BaseController):
    """
    SO(3) Geometric Controller — Lee, Leok, McClamroch, CDC 2010.

    Thrust is computed analytically from position/velocity errors,
    so the policy only needs to output a desired attitude.

    Policy input:
        raw : (N, 4)  desired quaternion q_des [w, x, y, z]  (unbounded, will be normalised)

    Controller computes:
        Fz  — from position/velocity PD law projected onto body z-axis
        tau — from geometric attitude error on SO(3)

    Args:
        hover_thrust : (N,)       total hover thrust [N]  (= mg)
        alloc_matrix : (N, 4, 4)
        J            : (N, 3)     diagonal inertia [kg·m²]
        m            : (N,)       mass per env [kg]
        hover_ratio  : float
        min_ratio    : float
        kp           : float      position gain
        kv           : float      velocity gain
        kR           : float      attitude error gain
        kw           : float      angular rate error gain
    """

    def __init__(
        self,
        hover_thrust : Tensor,        # (N,)
        alloc_matrix : Tensor,        # (N, 4, 4)
        J            : Tensor,        # (N, 3)
        m            : Tensor,        # (N,)
        hover_ratio  : float = 2.0,
        min_ratio    : float = 0.0,
        kp           : float = 4.0,
        kv           : float = 2.0,
        kR           : float = 0.05,
        kw           : float = 0.05,
        g            : float = 9.81,
    ):
        super().__init__(hover_thrust, alloc_matrix, hover_ratio, min_ratio)
        self._J  = J
        self._m  = m
        self._kp = kp
        self._kv = kv
        self._kR = kR
        self._kw = kw
        self._g  = g

    def update_parameters(
        self,
        hover_thrust : Tensor,
        alloc_matrix : Tensor,
        J            : Tensor,
        m            : Tensor,
    ):
        super().update_parameters(hover_thrust, alloc_matrix)
        self._J = J
        self._m = m

    def __call__(
        self,
        raw   : Tensor,              # (N, 4)   policy output — desired quaternion
        state : Tensor,              # (N, 13)  full state [p, v, q, w]
        p_ref : Tensor,              # (N, 3)   reference position
        v_ref : Tensor,              # (N, 3)   reference velocity
        a_ref : Tensor,              # (N, 3)   reference acceleration
        w_des : Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            raw         : (N, 4)   policy output — desired quaternion (unbounded)
            state       : (N, 13)  full current state
            p_ref       : (N, 3)   reference position
            v_ref       : (N, 3)   reference velocity
            a_ref       : (N, 3)   reference acceleration
            w_des       : (N, 3)   desired body rates — None = zero (attitude hold)

        Returns:
            motors : (N, 4)  per-motor thrust [N]
            wrench : (N, 4)  [Fz, tau_x, tau_y, tau_z]  for logging
        """
        p = state[:, 0:3]
        v = state[:, 3:6]
        q = state[:, 6:10]
        w = state[:, 10:13]

        R     = quat_to_rotmat(q)                              # (N, 3, 3)
        R_des = quat_to_rotmat(F.normalize(raw, dim=-1))       # (N, 3, 3)

        # ── Thrust from position/velocity PD law ─────────────────────────
        Fz = _compute_desired_thrust(
            p, v, p_ref, v_ref, a_ref, R,
            self._m, self._g, self._kp, self._kv,
            float(self._t_min.min()), float(self._t_max.max()),
        )

        # ── Torques from geometric attitude error ─────────────────────────
        w_des_ = torch.zeros_like(w) if w_des is None else w_des
        tau    = _geometric_torques(R, R_des, w, w_des_, self._J, self._kR, self._kw)

        wrench = torch.cat([Fz.unsqueeze(-1), tau], dim=-1)    # (N, 4)
        return self._wrench_to_motors(wrench), wrench


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
        
