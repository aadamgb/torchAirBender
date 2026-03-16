import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from utils.randomize import QuadrotorParams
from utils.math import *


#=====================================================
# """
#     Motor layout (top view, Z-up ENU):
#                     ^ b2
#                     |
#            (2) CW   |   (1) CCW
#                \    |    /
#                 \   |   /
#                  \  |  /
#         ----------- + ----------> b1
#                  /  |  \
#                 /   |   \
#                /    |    \
#            (3) CCW  |   (4) CW
#                     |
# """
"""
    Motor layout (top view, Z-up ENU):
                    ^ b2
                    |
           (1) CW   |   (3) CCW
               \    |    /
                \   |   /
                 \  |  /
        ----------- + ----------> b1
                 /  |  \
                /   |   \
               /    |    \
           (4) CCW  |   (2) CW
                    |
"""
#=====================================================


# ===========================================================
# JIT-compiled pure functions — these are the hottest paths.
# torch.jit.script works perfectly on stateless functions
# that only use tensor ops and have no Python-class baggage.
# ===========================================================

@torch.jit.script
def _compute_wrench(alloc_matrix: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
    """
    Maps per-motor thrusts to collective thrust + body torques.

    Args:
        alloc_matrix : (N, 4, 4)
        action       : (B, 4)

    Returns:
        Fz  : (B,)
        tau : (B, 3)
    """
    W   = torch.bmm(alloc_matrix, action.unsqueeze(-1)).squeeze(-1)  # (B, 4)
    Fz  = W[..., 0]
    tau = W[..., 1:4]
    return Fz, tau


@torch.jit.script
def _translational_deriv(
    v:  Tensor,
    q:  Tensor,
    Fz: Tensor,
    m:  Tensor,
    g:  Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Returns (p_dot, v_dot).

    Args:
        v  : (B, 3)
        q  : (B, 4)
        Fz : (B,)
        m  : (N,) or (1,)
        g  : (3,)
    """
    R = quat_to_rotmat(q) 

    thrust_body = torch.stack(
        [torch.zeros_like(Fz), torch.zeros_like(Fz), Fz], dim=-1
    ).unsqueeze(-1)                                                   # (B, 3, 1)
    thrust_world = torch.bmm(R, thrust_body).squeeze(-1)              # (B, 3)

    v_dot = thrust_world / m.unsqueeze(-1) + g                        # (B, 3)
    return v, v_dot


@torch.jit.script
def _rotational_deriv(
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
        J   : (N, 3) or (1, 3)
    """
    J_inv = 1.0 / J
    w_dot = J_inv * (tau - torch.linalg.cross(w, J * w))          # (B, 3)
    q_dot = quat_derivative(q, w)                                  # (B, 4)
    return q_dot, w_dot


@torch.jit.script
def _step_pmm(
    state:         Tensor,
    action:        Tensor,
    alloc_matrix:  Tensor,
    m:             Tensor,
    J:             Tensor,
    g:             Tensor,
    dt:            float,
) -> Tensor:
    """
    Full dynamics step as a single JIT-compiled function.
    Keeping all tensor work in one scripted function maximises
    fusion opportunities for the compiler.

    Args:
        state        : (B, 13)
        action       : (B,  4)
        alloc_matrix : (N, 4, 4)
        m            : (N,) or (1,)
        J            : (N, 3) or (1, 3)
        g            : (3,)
        dt           : scalar float

    Returns:
        next_state : (B, 13)
    """
    p = state[..., 0:3]
    v = state[..., 3:6]
    q = state[..., 6:10]
    w = state[..., 10:13]

    Fz, tau = _compute_wrench(alloc_matrix, action)
    p_dot, v_dot = _translational_deriv(v, q, Fz, m, g)
    q_dot, w_dot = _rotational_deriv(q, w, tau, J)

    p_next, v_next, q_next, w_next = integrate_euler(
        dt, p, v, q, w, p_dot, v_dot, q_dot, w_dot
    )

    return torch.cat([p_next, v_next, q_next, w_next], dim=-1)


# ===========================================================
# Main class — unchanged public API.
# torch.compile wraps the hot step() call at instantiation.
# ===========================================================

class QuadrotorDynamics:
    def __init__(self, cfg: DictConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.dt     = cfg.dt

        self.G = torch.tensor([0.0, 0.0, -9.81], device=self.device)  # (3,)
        self.g = 9.81

        self.m = torch.tensor(
            [cfg.dynamics.mass.nominal], device=self.device
        )                                                               # (1,)
        self.arm_length = torch.tensor(
            [cfg.dynamics.arm_length.nominal], device=self.device
        )                                                               # (1,)
        self.arm_angle = (
            torch.tensor([cfg.dynamics.arm_angle.nominal], device=self.device)
            * torch.pi / 180
        )                                                               # (1,)
        self.J = torch.tensor(
            [[
                cfg.dynamics.inertia.xx.nominal,
                cfg.dynamics.inertia.yy.nominal,
                cfg.dynamics.inertia.zz.nominal,
            ]],
            device=self.device,
        )                                                               # (1, 3)
        self.km = torch.tensor(
            [cfg.dynamics.km.nominal], device=self.device
        )                                                               # (1,)

        self._alloc_matrix: Tensor | None = None
        self._rebuild_alloc_matrix()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_alloc_matrix(self):
        s  = torch.sin(self.arm_angle)
        c  = torch.cos(self.arm_angle)
        l  = self.arm_length
        km = self.km

        # row0 = torch.ones(len(self.m), 4, device=self.device)
        # row1 = torch.stack([ l*s,  l*s, -l*s, -l*s], dim=-1)
        # row2 = torch.stack([-l*c,  l*c,  l*c, -l*c], dim=-1)
        # row3 = torch.stack([ km,  -km,   km,  -km ], dim=-1)

        row0 = torch.ones(len(self.m), 4, device=self.device)
        row1 = torch.stack([ l*s,  -l*s, l*s, -l*s], dim=-1)
        row2 = torch.stack([-l*c,  l*c,  l*c, -l*c], dim=-1)
        row3 = torch.stack([ km,  km,   -km,  -km ], dim=-1)

        self._alloc_matrix = torch.stack([row0, row1, row2, row3], dim=1)  # (N, 4, 4)

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
        # Delegates entirely to the JIT-compiled core function.
        # All Python overhead is eliminated inside _step_pmm.
        return _step_pmm(
            state,
            action,
            self._alloc_matrix,
            self.m,
            self.J,
            self.G,
            self.dt,
        )

    def set_parameters(self, params: QuadrotorParams):
        """
        Apply randomized per-env parameters and rebuild allocation matrix.

        Args:
            params : QuadrotorParams with tensors of shape (N, ...)
        """
        self.m          = params.mass
        self.arm_length = params.arm_length
        self.arm_angle  = params.arm_angle
        self.J          = params.J
        self.km         = params.km
        self._rebuild_alloc_matrix()

    def get_parameters(self) -> dict:
        return {
            'mass':       self.m,
            'arm_length': self.arm_length,
            'arm_angle':  self.arm_angle,
            'km':         self.km,
            'inertia':    self.J,
        }
    

    def get_srt_hover(self):
        return self.m * self.g / 4.0
    
    def get_hover_thrust(self):
        return self.m * self.g 



# ===========================================================
# torch.compile usage — apply OUTSIDE the class definition.
#
# Option A (recommended): compile at the point of instantiation
#   dynamics = QuadrotorDynamics(cfg)
#   dynamics.step = torch.compile(dynamics.step, mode="reduce-overhead")
#
# Option B: compile the whole class step method (affects all instances)
#   QuadrotorDynamics.step = torch.compile(
#       QuadrotorDynamics.step, mode="reduce-overhead"
#   )
#
# Mode guide:
#   "default"         — safe, good general speedup
#   "reduce-overhead" — best for small batches / RL envs (reduces kernel launch cost)
#   "max-autotune"    — slowest to compile, fastest at runtime (large-scale training)
#
# Note: torch.compile requires PyTorch >= 2.0 and a CUDA or MPS device to
# show meaningful gains. On CPU it still helps but less dramatically.
# ===========================================================