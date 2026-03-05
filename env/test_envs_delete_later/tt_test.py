import torch
from torch import Tensor
from omegaconf import DictConfig

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from utils.randomize import randomize_parameters
from utils.replay import TrajectoryTrackingRenderer


def generate_trajectory_params(
    num_envs: int,
    device,
    num_harmonics: int = 5,
    z_offset: float = 2.0,
) -> dict:
    amps   = torch.rand((num_envs, 3, num_harmonics), device=device) * 1.0 + 0.5
    freqs  = torch.rand((num_envs, 3, num_harmonics), device=device) * 2.0 + 0.2
    phases = torch.rand((num_envs, 3, num_harmonics), device=device) * 2 * torch.pi
    return {"amps": amps, "freqs": freqs, "phases": phases, "z_offset": z_offset}


def get_target(t: float, params: dict) -> tuple[Tensor, Tensor, Tensor]:
    amps, freqs, phases = params["amps"], params["freqs"], params["phases"]
    t_tensor = torch.full((amps.shape[0], 1, 1), t, device=amps.device, dtype=amps.dtype)
    angle = freqs * t_tensor + phases
    pos = torch.sum(amps * torch.sin(angle),             dim=2)
    vel = torch.sum(amps * freqs * torch.cos(angle),     dim=2)
    acc = torch.sum(-amps * freqs**2 * torch.sin(angle), dim=2)
    pos[:, 2] = pos[:, 2] + params["z_offset"]
    return pos, vel, acc


def train(cfg: DictConfig):
    dt       = cfg.dt
    device   = cfg.device
    steps    = cfg.steps

    # ── single env for now ───────────────────────────────────────────────
    num_envs = 1

    # ── init dynamics ────────────────────────────────────────────────────
    quadrotor = QuadrotorDynamics(cfg)
    randomized_params = randomize_parameters(cfg.dynamics, num_envs, device)
    quadrotor.set_parameters(randomized_params)

    # ── init state: hover at z=2 ─────────────────────────────────────────
    states = torch.zeros((num_envs, 13), device=device)
    states[:, 2] = 2.0   # start at z_offset height
    states[:, 6] = 1.0   # quaternion w = 1

    # ── hover thrust ─────────────────────────────────────────────────────
    hover_thrust = quadrotor.get_srt_hover()                        # (N,)
    actions = hover_thrust.unsqueeze(1).expand(-1, 4)               # (N, 4)

    # ── generate reference trajectory (env 0) ────────────────────────────
    traj_params = generate_trajectory_params(num_envs, device)

    # Pre-evaluate full reference path for rendering
    ref_positions = torch.stack(
        [get_target(t * dt, traj_params)[0][0] for t in range(steps)]
    )                                                                # (T, 3)

    # ── rollout buffers ───────────────────────────────────────────────────
    traj_env0 = torch.empty((steps, 17), device=device)             # 13 state + 4 actions

    # ── rollout ───────────────────────────────────────────────────────────
    for t in range(steps):
        states  = quadrotor.step(state=states, action=actions)
        traj_env0[t] = torch.cat([states[0].detach(), actions[0].detach()], dim=0)

    # ── render ────────────────────────────────────────────────────────────
    renderer = TrajectoryTrackingRenderer(
        ref_trajectory=ref_positions.cpu().numpy(),                  # (T, 3) ENU
        trajectory=traj_env0.cpu().numpy(),                          # (T, 17)
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()