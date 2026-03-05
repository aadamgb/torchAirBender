import os
import time
import torch
from torch import Tensor
from torch import nn
from omegaconf import DictConfig, OmegaConf

from utils.nn import MLP 
from utils.randomize import QuadrotorParams, randomize_parameters
from utils.replay import TrajectoryTrackingRenderer

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.srt_controller import SRTController


def generate_trajectory_params(
    num_envs: int,
    device,
    cfg: DictConfig
) -> dict:
    """
    Generates random harmonic trajectory coefficients for each environment.

    Returns a dict with tensors of shape (num_envs, 3, num_harmonics):
        amps     : amplitudes
        freqs    : frequencies
        phases   : phases
        z_offset : scalar float added to z position at eval time
    """
    amps   = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.A + 0.5
    freqs  = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.w + 0.2
    phases = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.phi * torch.pi

    return {"amps": amps, "freqs": freqs, "phases": phases, "z_offset": cfg.z_offset}


def get_target(
    t: float,
    params: dict,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Evaluates the trajectory at time t for all environments.

    Args:
        t      : current time (scalar float)
        params : dict from generate_trajectory_params

    Returns:
        pos : (num_envs, 3)
        vel : (num_envs, 3)
        acc : (num_envs, 3)
    """
    amps, freqs, phases = params["amps"], params["freqs"], params["phases"]

    t_tensor = torch.full(
        (amps.shape[0], 1, 1), t, device=amps.device, dtype=amps.dtype
    )

    angle = freqs * t_tensor + phases                               # (N, 3, H)

    pos = torch.sum(amps * torch.sin(angle),              dim=2)    # (N, 3)
    vel = torch.sum(amps * freqs * torch.cos(angle),      dim=2)    # (N, 3)
    acc = torch.sum(-amps * freqs**2 * torch.sin(angle),  dim=2)    # (N, 3)

    # Shift z up so the trajectory stays airborne
    pos[:, 2] = pos[:, 2] + params["z_offset"]
    # vel and acc are unaffected (derivative of a constant is 0)

    return pos, vel, acc


def reset(
        cfg: DictConfig,
        traj_params: dict,
) -> tuple[Tensor, QuadrotorParams]:
    """Reset all envs to the trajectory's t=0 position and velocity."""
    num_envs = cfg.num_envs
    device   = cfg.device

    pos0, vel0, _ = get_target(0.0, traj_params)                   # (N, 3) each

    states = torch.zeros((num_envs, 13), device=device)
    states[:, 0:3]  = pos0.detach()
    states[:, 3:6]  = vel0.detach()                                # match target velocity
    states[:, 6]    = 1.0                                          # quaternion w = 1

    params = randomize_parameters(cfg.dynamics, num_envs, device)

    return states, params


def reset_terminated(
        states: Tensor,
        terminated: Tensor,     # (N,) bool
        pos_ref: Tensor,        # (N, 3) current reference position
        vel_ref: Tensor,        # (N, 3) current reference velocity
) -> Tensor:
    """Snap terminated envs back to current reference position and velocity."""
    if not terminated.any():
        return states

    idx = terminated.nonzero(as_tuple=True)[0]

    states[idx, 0:3]  = pos_ref[idx].detach()
    states[idx, 3:6]  = vel_ref[idx].detach()                     # match target velocity
    states[idx, 6:13] = 0.0
    states[idx, 6]    = 1.0                                        # quaternion w = 1

    return states


def get_observation(
        states: Tensor,
        pos_ref: Tensor,   # (N, 3)
        vel_ref: Tensor,   # (N, 3)
        acc_ref: Tensor,   # (N, 3)
) -> Tensor:
    p = states[:, 0:3]
    v = states[:, 3:6]
    q = states[:, 6:10]
    w = states[:, 10:13]

    p_error = pos_ref - p   # (N, 3)
    v_error = vel_ref - v   # (N, 3)

    return torch.cat([p_error, v_error, acc_ref, q, w], dim=-1)  # (N, 16)

def compute_loss(
    states: Tensor,
    pos_ref: Tensor,
    vel_ref: Tensor,
    weights: DictConfig,
    mask: Tensor | None = None,
) -> Tensor:
    if mask is not None and not mask.any():
        return states.sum() * 0.0

    s = states[mask] if mask is not None else states
    p = pos_ref[mask] if mask is not None else pos_ref
    v = vel_ref[mask] if mask is not None else vel_ref

    pos_loss  = torch.linalg.norm(p - s[:, 0:3], dim=-1).mean()
    vel_loss  = torch.linalg.norm(v - s[:, 3:6], dim=-1).mean()  
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    return weights.pos * pos_loss + weights.vel * vel_loss + weights.body_rates * rate_loss



def train(cfg: DictConfig):
    start = time.time()
    output_dir = "/home/adame/torchAirBender/outputs/policies/TT"
    os.makedirs(output_dir, exist_ok=True)

    dt         = cfg.dt
    device     = cfg.device
    num_envs   = cfg.num_envs
    episodes   = cfg.episodes
    steps      = cfg.steps
    truncation = cfg.truncation

    print(f"\n{'='*75}")
    print(f"  Trajectory Tracking Training")
    print(f"  Envs: {num_envs}  |  Episodes: {episodes}  |  Steps: {steps}  |  Horizon: {truncation}")
    print(f"{'='*75}\n")

    states = torch.zeros((num_envs, 13), device=device)
    truncation_losses = torch.empty(truncation, device=device)
    traj_env0 = torch.empty((steps, 20), device=device)               # 13 state + 4 actions + 3 ref pos

    quadrotor = QuadrotorDynamics(cfg)
    policy = MLP(
        layer_sizes=cfg.env.nn,                                        # must be [16, ..., 4]
        activation=nn.ReLU,
        output_activation=nn.Sigmoid(),
        output_bias_init=0.0,
    ).to(device)
    controller = SRTController(cfg)
    optimizer  = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)

    best_loss = float("inf")
    for ep in range(episodes):

        # New trajectory + drone params each episode
        traj_params       = generate_trajectory_params(num_envs, device, cfg.env.traj)

        # --- episode reset ---
        states, randomized_params = reset(cfg, traj_params)
        quadrotor.set_parameters(randomized_params)

        ep_loss    = 0.0
        num_updates = 0

        for t in range(steps):

            if t % truncation == 0:
                states       = states.detach()
                window_start = t
                window_loss_sum = torch.zeros(1, device=device)

            # --- reference at current time ---
            pos_ref, vel_ref, acc_ref = get_target(t * dt, traj_params)  # (N, 3) each

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            raw     = policy(obs)
            actions = controller(raw)
            states  = quadrotor.step(state=states, action=actions)

            # --- termination ---
            dist = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)  # (N,)
            too_far = dist > cfg.env.max_dist_to_target
            alive = ~too_far

            step_loss = compute_loss(states, pos_ref, vel_ref, cfg.env.loss_weights, mask=alive)
            truncation_losses[t % truncation] = step_loss.detach()
            window_loss_sum = window_loss_sum + step_loss

            if ep == episodes - 1:
                traj_env0[t] = torch.cat([states[0].detach(), actions[0].detach(), pos_ref[0].detach(),], dim=0)

            if (t + 1) % truncation == 0 or (t + 1) == steps:
                window_len = (t + 1) - window_start
                loss = window_loss_sum / window_len
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss     += truncation_losses[:window_len].mean().item()
                num_updates += 1

            # --- reset terminated envs ---
            states = reset_terminated(states, too_far, pos_ref, vel_ref)


        avg_ep_loss = ep_loss / num_updates
        print(f"  Episode {ep+1:>4}/{episodes}  |  Avg Loss: {avg_ep_loss:.4f}")

        if avg_ep_loss < best_loss:
            best_loss = avg_ep_loss
            torch.save(policy.state_dict(), os.path.join(output_dir, "trajectory_tracking_best.pt"))

    print(f"Total training time: {time.time() - start:.2f}s")
    torch.save(policy.state_dict(), os.path.join(output_dir, "trajectory_tracking_final.pt"))

    renderer = TrajectoryTrackingRenderer(
        ref_trajectory=traj_env0[:, 17:20].cpu().numpy(),
        trajectory=traj_env0.cpu().numpy(),
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()



#==============================================================
# Load policy and test one env one episode
#==============================================================

def test(cfg: DictConfig):
    # policy_path = "/home/adame/torchAirBender/outputs/policies/TT_BEST.pt"
    # policy_path = "/home/adame/torchAirBender/outputs/policies/trajectory_tracking_best.pt"
    policy_path = "/home/adame/torchAirBender/outputs/policies/TT/trajectory_tracking_best.pt"
    dt      = cfg.dt
    device  = cfg.device
    steps   = cfg.steps

    # ── single env ───────────────────────────────────────────────────────
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["num_envs"] = 1
    cfg = OmegaConf.create(cfg_dict)

    # ── load policy ──────────────────────────────────────────────────────
    policy = MLP(
        layer_sizes=cfg.env.nn,
        activation=nn.ReLU,
        output_activation=nn.Sigmoid(),
        output_bias_init=0.0,
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    print(f"Loaded policy from: {policy_path}")

    # ── init ─────────────────────────────────────────────────────────────
    controller        = SRTController(cfg)
    traj_params       = generate_trajectory_params(1, device, cfg.env.traj)
    states, randomized_params = reset(cfg, traj_params)
    quadrotor         = QuadrotorDynamics(cfg)
    quadrotor.set_parameters(randomized_params)

    traj = torch.empty((steps, 20), device=device)  # 13 state + 4 actions + 3 ref pos

    print(f"\n{'='*60}")
    print(" Testing Trajectory Tracking ")
    print(f"{'='*60}\n")

    # ── rollout ──────────────────────────────────────────────────────────
    total_loss = 0.0
    with torch.no_grad():
        for t in range(steps):

            pos_ref, vel_ref, acc_ref = get_target(t * dt, traj_params)

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            raw     = policy(obs)
            actions = controller(raw)
            states  = quadrotor.step(state=states, action=actions)

            dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target

            step_loss   = compute_loss(states, pos_ref, vel_ref, cfg.env.loss_weights)
            total_loss += step_loss.item()

            traj[t] = torch.cat([states[0], actions[0], pos_ref[0]], dim=0)

            if too_far[0]:
                print(f"  !! Terminated at step {t+1} — dist: {dist[0]:.3f}")
                states = reset_terminated(states, too_far, pos_ref, vel_ref)

    print(f"\n  Avg Loss:   {total_loss / steps:.4f}")

    # ── replay ───────────────────────────────────────────────────────────
    renderer = TrajectoryTrackingRenderer(
        ref_trajectory=traj[:, 17:20].cpu().numpy(),
        trajectory=traj[:, :17].cpu().numpy(),
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()