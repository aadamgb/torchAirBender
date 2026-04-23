import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from utils.randomize import QuadrotorParams
from utils.math import quat_to_rotmat, quat_derivative


#=====================================================
"""
    Motor layout (top view, Z-up ENU):
                    ^ b2
                    |
           (2) CW   |    (3) CCW
               \    |    /
                \   |   /
                 \  |  /
        ----------- + ----------> b1
                 /  |  \
                /   |   \
               /    |    \
           (4) CCW  |     (1) CW
                    |
"""
#=====================================================

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
def integrate_euler(
    dt: float,
    p: Tensor, v: Tensor, q: Tensor, w: Tensor,
    p_dot: Tensor, v_dot: Tensor, q_dot: Tensor, w_dot: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    v_next = v + v_dot * dt
    p_next = p + v_next * dt
    q_next = nn.functional.normalize(q + q_dot * dt, dim=-1)               # renormalize quaternion
    w_next = w + w_dot * dt
    return p_next, v_next, q_next, w_next


@torch.jit.script
def _step_fwd(
    state:         Tensor,
    thrusts:       Tensor,
    alloc:         Tensor,
    m:             Tensor,
    J:             Tensor,
    km:            Tensor,
    a0:            Tensor,
    motor_tau:     Tensor,
    G:             Tensor,
    dt:            float,
) -> Tensor:
    """
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
    Omegas = state[..., 13:17]

    # print(thrusts)
    # print(m)
    # print(km.view(-1, 1))
    Omegas_cmd = torch.sqrt(torch.clamp(thrusts / a0.view(-1, 1), min=1e-3))
    # print(Omegas_cmd)

    Omegas_dot = (Omegas_cmd - Omegas) / motor_tau  # TODO: Improve this.... clamp rate limits
    # print(f"Omegas_dot: {Omegas_dot}")

    Omegas_next = Omegas + Omegas_dot * dt

    thrusts_next = a0.view(-1, 1) * Omegas_next**2
    # print(thrusts_next)

    # print(f"Omegas:\n {Omegas}")

    Fz, tau = _compute_wrench(alloc, thrusts_next)
    p_dot, v_dot = _translational_deriv(v, q, Fz, m, G)
    q_dot, w_dot = _rotational_deriv(q, w, tau, J)

    p_next, v_next, q_next, w_next = integrate_euler(
        dt, p, v, q, w, p_dot, v_dot, q_dot, w_dot
    )

    return torch.cat([p_next, v_next, q_next, w_next, Omegas_next], dim=-1)

@torch.jit.script
def _step_bck(
    state:         Tensor,
    action:        Tensor,
    alloc_matrix:  Tensor,
    m:             Tensor,
    J:             Tensor,
    g:             Tensor,
    dt:            float,
) -> Tensor:
    """
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
#                        Main class 
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
        self.arm_length = torch.full(
            (1, 4), cfg.dynamics.arm_length.nominal, device=self.device
        )                                                               # (1, 4)
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
        ) 

        self.a0 = torch.tensor(
            [cfg.dynamics.a0.nominal], device=self.device
        )

        self.motor_tau = torch.tensor(
            [cfg.dynamics.motor_tau.nominal], device=self.device
        )                                                               

        self._alloc_matrix: Tensor | None = None

        self.motor_eta = torch.ones((1, 4), device=self.device)
        
        self._rebuild_alloc_matrix()

        self.max_TWR  = torch.tensor(
            [cfg.dynamics.max_TWR], device=self.device
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_alloc_matrix(self):
        s  = torch.sin(self.arm_angle)     # Careful with the units!!!!
        c  = torch.cos(self.arm_angle)
        l  = self.arm_length
        km = self.km

        N = l.shape[0]
        row0 = torch.ones(N, 4, device=l.device)
        row1 = torch.stack([-l[:,0]*s,  l[:,1]*s,  l[:,2]*s, -l[:,3]*s], dim=-1)
        row2 = torch.stack([-l[:,0]*c,  l[:,1]*c, -l[:,2]*c,  l[:,3]*c], dim=-1)
        row3 = torch.stack([ -km,  -km,   km,  km ], dim=-1)

        A = torch.stack([row0, row1, row2, row3], dim=1)   # (N, 4, 4)

        self._alloc_matrix = A * self.motor_eta.unsqueeze(1)  # (N, 4, 4) * (N, 1, 4)

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
        return _step_fwd(
            state=state,
            thrusts=action,
            alloc=self._alloc_matrix,
            m=self.m,
            J=self.J,
            km=self.km,
            a0=self.a0,
            motor_tau=self.motor_tau,
            G=self.G,
            dt=self.dt,
        )
        # return _step_bck(
        #     state=state,
        #     action=action,
        #     alloc=self._alloc_matrix,
        #     m=self.m,
        #     J=self.J,
        #     G=self.G,
        #     dt=self.dt,
        # )

    def set_parameters(self, params: QuadrotorParams):
        """
        Apply randomized per-env parameters and rebuild allocation matrix.

        Args:
            params : QuadrotorParams with tensors of shape (N, ...)
        """
        self.m          = params.mass
        self.arm_length = params.arm_length
        self.arm_angle  = params.arm_angle * torch.pi / 180
        self.J          = params.J
        self.km         = params.km
        self.max_TWR    = params.max_TWR
        self.motor_eta  = params.motor_eta
        self._rebuild_alloc_matrix()

    def get_parameters(self) -> dict:
        return {
            'mass':       self.m,
            'arm_length': self.arm_length,
            'arm_angle':  self.arm_angle,
            'km':         self.km,
            'inertia':    self.J,
            'max_TWR':    self.max_TWR
        }
    

    def get_srt_hover_thurst(self):
        return self.m * self.g / 4.0
    
    def get_srt_hover_speed(self):
        return torch.sqrt(self.m * self.g / (4.0 * self.motor_tau))
    
    def get_total_hover_thrust(self):
        return self.m * self.g 

