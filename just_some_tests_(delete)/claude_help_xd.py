import torch
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from omegaconf import DictConfig


@dataclass
class QuadrotorParams:
    """Per-environment physical parameters."""
    mass:       Tensor  # (B,)
    kf:         Tensor  # (B,)
    km:         Tensor  # (B,)
    arm_length: Tensor  # (B,)
    J:          Tensor  # (B, 3)
    C_D:        Tensor  # (B,) or (B, 3)
    motor_tau:  Tensor  # (B,)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        q : (B, 4)  [w, x, y, z]

    Returns:
        R : (B, 3, 3)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    B = q.shape[0]

    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(B, 3, 3)

    return R


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


def compute_wrench(
    Omega:      Tensor,
    kf:         Tensor,
    km:         Tensor,
    arm_length: Tensor,
    arm_angle:  float,
) -> Tensor:
    """
    Compute wrench [Thrust, Tau_x, Tau_y, Tau_z] from motor speeds.

    Motor layout (top view, Z-up ENU):
                    ^ b2
                    |
           (2) CW   |   (1) CCW
               \    |    /
                 \  |  /
        ----------- + ----------> b1
                 /  |  \
               /    |    \
           (3) CCW  |   (4) CW

    Args:
        Omega      : (B, 4)  motor speeds [rad/s]
        kf         : (B,)    thrust coefficient
        km         : (B,)    drag-torque coefficient
        arm_length : (B,)    arm length [m]
        arm_angle  : float   arm angle offset from b1 axis [rad]

    Returns:
        W : (B, 4)  [Thrust, Tau_x, Tau_y, Tau_z]
    """
    s = torch.sin(torch.tensor(arm_angle, dtype=Omega.dtype, device=Omega.device))
    c = torch.cos(torch.tensor(arm_angle, dtype=Omega.dtype, device=Omega.device))
    l  = arm_length   # (B,)
    kf = kf           # (B,)
    km = km           # (B,)

    # Build allocation matrix A: (B, 4, 4)
    # Matches JAX reference exactly, column-per-motor, row-per-wrench-component
    z  = torch.zeros_like(l)
    A = torch.stack([
        torch.stack([ kf,       kf,       kf,      kf      ], dim=-1),  # Thrust row
        torch.stack([ l*kf*s,   l*kf*c,  -l*kf*s, -l*kf*c ], dim=-1),  # Tau_x row
        torch.stack([-l*kf*c,   l*kf*s,   l*kf*c, -l*kf*s ], dim=-1),  # Tau_y row
        torch.stack([ km,      -km,        km,     -km      ], dim=-1),  # Tau_z row
    ], dim=1)  # (B, 4, 4)

    # W = A @ Omega^2  — equivalent to JAX: (Omega**2) @ A.T
    Omega_sq = (Omega ** 2).unsqueeze(-1)  # (B, 4, 1)
    W = torch.bmm(A, Omega_sq).squeeze(-1)  # (B, 4)

    return W


def motor_dynamics_step(
    Omega:     Tensor,
    Omega_cmd: Tensor,
    motor_tau: Tensor,
    dt:        float,
    dot_min:   float,
    dot_max:   float,
    o_min:     float,
    o_max:     float,
) -> Tensor:
    """
    First-order motor dynamics with rate and speed limits.

    Args:
        Omega     : (B, 4)  current motor speeds
        Omega_cmd : (B, 4)  commanded motor speeds
        motor_tau : (B,)    motor time constant
        dt        : float   timestep

    Returns:
        Omega_next : (B, 4)
    """
    tau = motor_tau.unsqueeze(-1)                          # (B, 1) — broadcast over motors
    Omega_dot = (Omega_cmd - Omega) / tau
    Omega_dot = Omega_dot.clamp(dot_min, dot_max)
    Omega_next = Omega + Omega_dot * dt
    Omega_next = Omega_next.clamp(o_min, o_max)
    return Omega_next


class QuadrotorDynamics:
    """
    Batched quadrotor rigid-body dynamics in the Z-up ENU frame.

    Gravity points in the -z direction: g = [0, 0, -9.81] m/s^2.

    Physical parameters are NOT stored in the class — they are passed in
    as a QuadrotorParams dataclass at every step call, so each environment
    can have its own randomized parameters.

    State layout:
        Full      : [p(3), v(3), q(4), w(3), Omega(4)]  -> 17 elements
        Simplified: [p(3), v(3), q(4), w(3)]             -> 13 elements

    Usage:
        dynamics = QuadrotorDynamics(cfg)
        params   = randomize_parameters(cfg.dynamics, num_envs, device)
        states   = dynamics.step_full(states, cmds, params)
    """

    def __init__(self, cfg: DictConfig):
        dyn = cfg.dynamics

        self.device    = torch.device(cfg.get("device", "cpu"))
        self.dt        = float(dyn.dt)
        self.arm_angle = float(dyn.arm_angle)
        self.g         = torch.tensor([0.0, 0.0, -float(dyn.g)], device=self.device)

        self.dot_min = float(dyn.motor_dot_min)
        self.dot_max = float(dyn.motor_dot_max)
        self.o_min   = float(dyn.motor_omega_min)
        self.o_max   = float(dyn.motor_omega_max)

    # ------------------------------------------------------------------
    # Continuous-time derivatives
    # ------------------------------------------------------------------

    def _translational_accel(
        self,
        v:      Tensor,
        q:      Tensor,
        Fz:     Tensor,
        mass:   Tensor,
        C_D:    Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (p_dot, v_dot).

        Includes linear aerodynamic drag: F_drag = -C_D * v (world frame).

        Args:
            v    : (B, 3)
            q    : (B, 4)
            Fz   : (B,)
            mass : (B,)
            C_D  : (B,) or (B, 3)
        """
        R = quat_to_rotmat(q)                               # (B, 3, 3)

        # Thrust vector in world frame
        thrust_body  = torch.stack(
            [torch.zeros_like(Fz), torch.zeros_like(Fz), Fz], dim=-1
        ).unsqueeze(-1)                                     # (B, 3, 1)
        thrust_world = torch.bmm(R, thrust_body).squeeze(-1)  # (B, 3)

        drag  = -C_D.unsqueeze(-1) * v                      # (B, 3)
        v_dot = (thrust_world + drag) / mass.unsqueeze(-1) + self.g

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

    # ------------------------------------------------------------------
    # Euler integrator
    # ------------------------------------------------------------------

    def _integrate_euler(
        self,
        p: Tensor, v: Tensor, q: Tensor, w: Tensor,
        p_dot: Tensor, v_dot: Tensor, q_dot: Tensor, w_dot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        p_next = p + p_dot * self.dt
        v_next = v + v_dot * self.dt
        q_next = q + q_dot * self.dt
        q_next = F.normalize(q_next, dim=-1)               # renormalize quaternion
        w_next = w + w_dot * self.dt
        return p_next, v_next, q_next, w_next

    # ------------------------------------------------------------------
    # Batched step functions
    # ------------------------------------------------------------------

    def step_full(
        self,
        state:     Tensor,
        Omega_cmd: Tensor,
        params:    QuadrotorParams,
    ) -> Tensor:
        """
        One step with motor dynamics.

        Args:
            state     : (B, 17)  [p, v, q, w, Omega]
            Omega_cmd : (B, 4)   commanded motor speeds [rad/s]
            params    : QuadrotorParams

        Returns:
            state_next : (B, 17)
        """
        p     = state[:, 0:3]
        v     = state[:, 3:6]
        q     = state[:, 6:10]
        w     = state[:, 10:13]
        Omega = state[:, 13:17]

        # Motor dynamics
        Omega_next = motor_dynamics_step(
            Omega, Omega_cmd, params.motor_tau, self.dt,
            self.dot_min, self.dot_max, self.o_min, self.o_max,
        )

        # Wrench
        W   = compute_wrench(
            Omega_next, params.kf, params.km, params.arm_length, self.arm_angle
        )
        Fz  = W[:, 0]
        tau = W[:, 1:4]

        # Derivatives
        p_dot, v_dot = self._translational_accel(v, q, Fz, params.mass, params.C_D)
        q_dot, w_dot = self._rotational_deriv(q, w, tau, params.J)

        # Integrate
        p_n, v_n, q_n, w_n = self._integrate_euler(
            p, v, q, w, p_dot, v_dot, q_dot, w_dot
        )

        return torch.cat([p_n, v_n, q_n, w_n, Omega_next], dim=-1)

    def step_simple(
        self,
        state:  Tensor,
        W:      Tensor,
        params: QuadrotorParams,
    ) -> Tensor:
        """
        One step without motor dynamics — wrench [Fz, Tx, Ty, Tz] is passed directly.

        Args:
            state  : (B, 13)  [p, v, q, w]
            W      : (B, 4)   [Thrust, Tau_x, Tau_y, Tau_z]
            params : QuadrotorParams

        Returns:
            state_next : (B, 13)
        """
        p  = state[:, 0:3]
        v  = state[:, 3:6]
        q  = state[:, 6:10]
        w  = state[:, 10:13]
        Fz = W[:, 0]
        tau = W[:, 1:4]

        p_dot, v_dot = self._translational_accel(v, q, Fz, params.mass, params.C_D)
        q_dot, w_dot = self._rotational_deriv(q, w, tau, params.J)

        p_n, v_n, q_n, w_n = self._integrate_euler(
            p, v, q, w, p_dot, v_dot, q_dot, w_dot
        )

        return torch.cat([p_n, v_n, q_n, w_n], dim=-1)