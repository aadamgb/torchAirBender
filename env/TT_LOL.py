import os
import time
import torch
from torch import Tensor
from torch import nn

from omegaconf import DictConfig, OmegaConf
import pandas as pd

from utils.nn import MLP
from utils.randomize import QuadrotorParams, randomize_parameters
from utils.replay import TrajectoryTrackingRenderer
from utils.math import quat_to_rotmat

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.srt_controller import SRTController

from miscellaneous.loader import load_LOL


# ==============================================================
# Helpers
# ==============================================================

def acc_to_quat(acc_ref: Tensor, g: float = 9.81) -> Tensor:
    gravity    = torch.tensor([0.0, 0.0, g], device=acc_ref.device)
    thrust_dir = torch.nn.functional.normalize(acc_ref + gravity, dim=-1)
    world_z       = torch.zeros_like(thrust_dir)
    world_z[:, 2] = 1.0
    axis = torch.linalg.cross(world_z, thrust_dir)
    dot  = (world_z * thrust_dir).sum(dim=-1, keepdim=True)
    w    = torch.sqrt(torch.clamp((1.0 + dot) / 2.0, min=1e-6))
    xyz  = torch.nn.functional.normalize(axis, dim=-1) * torch.sqrt(
        torch.clamp((1.0 - dot) / 2.0, min=0.0)
    )
    return torch.cat([w, xyz], dim=-1)


def reset_terminated(
    states:     Tensor,
    terminated: Tensor,   # (N,) bool
    pos_ref:    Tensor,   # (N, 3)
    vel_ref:    Tensor,   # (N, 3)
    acc_ref:    Tensor,   # (N, 3)
) -> Tensor:
    """Snap terminated envs back to current reference position and velocity."""
    if not terminated.any():
        return states

    idx = terminated.nonzero(as_tuple=True)[0]
    states[idx, 0:3]   = pos_ref[idx].detach()
    # states[idx, 3:6]   = vel_ref[idx].detach()
    # states[idx, 6:10]  = acc_to_quat(acc_ref[idx].detach())
    # states[idx, 10:13] = 0.0
    return states


def get_observation(
    states:  Tensor,
    pos_ref: Tensor,   # (N, 3)
    vel_ref: Tensor,   # (N, 3)
    acc_ref: Tensor,   # (N, 3)
) -> Tensor:
    p_error = pos_ref - states[:, 0:3]
    v_error = vel_ref - states[:, 3:6]
    return torch.cat([p_error, v_error, acc_ref, states[:, 6:10], states[:, 10:13]], dim=-1)  # (N, 16)


def compute_loss(
    states:  Tensor,
    pos_ref: Tensor,
    vel_ref: Tensor,
    acc_ref: Tensor,
    weights: DictConfig,
    mask:    Tensor | None = None,
) -> Tensor:
    if mask is not None and not mask.any():
        return states.sum() * 0.0

    s = states[mask] if mask is not None else states
    p = pos_ref[mask] if mask is not None else pos_ref
    v = vel_ref[mask] if mask is not None else vel_ref
    a = acc_ref[mask] if mask is not None else acc_ref

    pos_loss  = torch.linalg.norm(p - s[:, 0:3], dim=-1).mean()
    vel_loss  = torch.linalg.norm(v - s[:, 3:6], dim=-1).mean()
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    gravity    = torch.tensor([0.0, 0.0, -9.81], device=s.device)
    thrust_dir = torch.nn.functional.normalize(a + gravity, dim=-1)
    R          = quat_to_rotmat(s[:, 6:10])
    body_z     = R[:, :, 2]
    cos_angle  = (body_z * thrust_dir).sum(dim=-1).clamp(-1, 1)
    att_loss   = (1.0 - cos_angle).mean()

    return (
        weights.pos        * pos_loss  +
        weights.vel        * vel_loss  +
        weights.att        * att_loss  +
        weights.body_rates * rate_loss
    )


def _make_get_ref(loaded: dict, num_envs: int, speed_scale=1.0):
    """Wraps a loaded trajectory dict into a get_ref(t) callable."""
    def get_ref(t: int):
        scaled_t = min(int(t * speed_scale), len(loaded["pos"]) - 1)
        pos = loaded["pos"][scaled_t].unsqueeze(0).expand(num_envs, -1)
        vel = loaded["vel"][scaled_t].unsqueeze(0).expand(num_envs, -1)
        acc = loaded["acc"][scaled_t].unsqueeze(0).expand(num_envs, -1)
        return pos, vel, acc
    return get_ref


# ==============================================================
# Train
# ==============================================================

def train(cfg: DictConfig):
    start      = time.time()
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

    truncation_losses = torch.empty(truncation, device=device)
    traj_env0         = torch.empty((steps, 20), device=device)   # 13 state + 4 actions + 3 ref pos

    quadrotor  = QuadrotorDynamics(cfg)
    policy     = MLP(
        layer_sizes=cfg.env.nn,
        activation=nn.ReLU,
        output_activation=nn.Sigmoid(),
        output_bias_init=0.0,
    ).to(device)
    # policy.load_state_dict(torch.load(
    #     "/home/adame/torchAirBender/outputs/las_mejores/trajectory_tracking_w_2.75.pt",
    #     map_location=device,
    # ))

    controller = SRTController(cfg)
    optimizer  = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)

    # load trajectory once — it doesn't change between episodes
    loaded = load_LOL(cfg.env.traj.path, steps, device, dt=cfg.dt)

    speed_scale = 0.1
    best_loss = float("inf")

    for ep in range(episodes):

        # --- episode reset ---
        get_ref = _make_get_ref(loaded, num_envs, speed_scale)
        pos0, vel0, acc0  = get_ref(0)
        states            = torch.zeros((num_envs, 13), device=device)
        states[:, 0:3]    = pos0.detach()
        states[:, 3:6]    = vel0.detach()
        states[:, 6:10]   = acc_to_quat(acc0.detach())
        randomized_params = randomize_parameters(cfg.dynamics, num_envs, device)
        quadrotor.set_parameters(randomized_params)

        ep_loss     = 0.0
        num_updates = 0

        for t in range(steps):

            if t % truncation == 0:
                states          = states.detach()
                window_start    = t
                window_loss_sum = torch.zeros(1, device=device)

            pos_ref, vel_ref, acc_ref = get_ref(t)

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            raw     = policy(obs)
            actions = controller(raw)
            states  = quadrotor.step(state=states, action=actions)

            dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target
            alive   = ~too_far

            step_loss = compute_loss(states, pos_ref, vel_ref, acc_ref, cfg.env.loss_weights, mask=alive)
            truncation_losses[t % truncation] = step_loss.detach()
            window_loss_sum = window_loss_sum + step_loss

            if ep == episodes - 1:
                traj_env0[t] = torch.cat(
                    [states[0].detach(), actions[0].detach(), pos_ref[0].detach()],
                    dim=0,
                )

            if (t + 1) % truncation == 0 or (t + 1) == steps:
                window_len = (t + 1) - window_start
                loss       = window_loss_sum / window_len
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss     += truncation_losses[:window_len].mean().item()
                num_updates += 1

            states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)

        avg_ep_loss = ep_loss / num_updates
        if avg_ep_loss < 1.5:
            print(f"Increasing speed scale 🔥")
            torch.save(policy.state_dict(), os.path.join(output_dir, f"TT_at_{speed_scale:.2f}.pt"))
            speed_scale += 0.05

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


# ==============================================================
# Test
# ==============================================================

def test(cfg: DictConfig):
    policy_path = "/home/adame/torchAirBender/outputs/policies/TT/trajectory_tracking_best.pt"
    dt     = cfg.dt
    device = cfg.device
    steps  = cfg.steps

    # single env
    cfg_dict             = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["num_envs"] = 1
    cfg                  = OmegaConf.create(cfg_dict)

    # load policy
    policy = MLP(
        layer_sizes=cfg.env.nn,
        activation=nn.ReLU,
        output_activation=nn.Sigmoid(),
        output_bias_init=0.0,
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    print(f"Loaded policy from: {policy_path}")

    # load trajectory
    loaded  = load_LOL(cfg.env.traj.path, steps, device, dt)
    get_ref = _make_get_ref(loaded, num_envs=1)

    # init
    controller        = SRTController(cfg)
    pos0, vel0, acc0  = get_ref(0)
    states            = torch.zeros((1, 13), device=device)
    states[:, 0:3]    = pos0.detach()
    states[:, 3:6]    = vel0.detach()
    states[:, 6:10]   = acc_to_quat(acc0.detach())

    randomized_params = randomize_parameters(cfg.dynamics, 1, device)
    quadrotor         = QuadrotorDynamics(cfg)
    quadrotor.set_parameters(randomized_params)

    traj = torch.empty((steps, 20), device=device)

    print(f"\n{'='*60}")
    print(" Testing Trajectory Tracking ")
    print(f"{'='*60}\n")

    total_loss   = 0.0
    sq_error_sum = 0.0

    with torch.inference_mode():
        for t in range(steps):

            pos_ref, vel_ref, acc_ref = get_ref(t)

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            raw     = policy(obs)
            actions = controller(raw)
            states  = quadrotor.step(state=states, action=actions)

            dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target

            sq_error_sum += (dist ** 2).item()

            step_loss   = compute_loss(states, pos_ref, vel_ref, acc_ref, cfg.env.loss_weights)
            total_loss += step_loss.item()

            traj[t] = torch.cat([states[0], actions[0], pos_ref[0]], dim=0)

            if too_far[0]:
                print(f"  !! Terminated at step {t+1} — dist: {dist[0]:.3f}")
                states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)

    print(f"\n  Avg Loss : {total_loss / steps:.4f}")
    print(f"  RMSE Pos : {(sq_error_sum / steps) ** 0.5:.4f} m")

    renderer = TrajectoryTrackingRenderer(
        ref_trajectory=traj[:, 17:20].cpu().numpy(),
        trajectory=traj[:, :17].cpu().numpy(),
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()