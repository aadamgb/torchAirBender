import os
import time
import torch
from torch import Tensor, nn
from omegaconf import DictConfig

from utils.nn import MLP
from utils.randomize import QuadrotorParams, randomize_parameters
from utils.replay import TrajectoryTrackingRenderer
from utils.math import quat_to_rotmat
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import CTBRController

def acc_to_quat(acc_ref: Tensor, g: float = 9.81) -> Tensor:
    """
    Computes desired quaternion from reference acceleration.
    Desired thrust direction = acc_ref + gravity_vector.

    Args:
        acc_ref : (N, 3)
    Returns:
        q_des   : (N, 4)  [w, x, y, z]
    """
    gravity = torch.tensor([0.0, 0.0, g], device=acc_ref.device)   # acceleration compensation
    thrust_dir = acc_ref + gravity                                     # (N, 3)
    thrust_dir = torch.nn.functional.normalize(thrust_dir, dim=-1)    # (N, 3)

    # Body z-axis in world frame should align with thrust_dir
    # Rotation from world z [0,0,1] to thrust_dir
    world_z = torch.zeros_like(thrust_dir)
    world_z[:, 2] = 1.0

    # Axis of rotation = cross(world_z, thrust_dir)
    axis = torch.linalg.cross(world_z, thrust_dir)                    # (N, 3)
    # Angle: cos(theta) = dot(world_z, thrust_dir)
    dot  = (world_z * thrust_dir).sum(dim=-1, keepdim=True)           # (N, 1)

    # Quaternion: w = cos(theta/2), xyz = sin(theta/2) * axis_normalized
    # Using half-angle: w = sqrt((1 + cos)/2), |xyz| = sqrt((1 - cos)/2)
    w   = torch.sqrt(torch.clamp((1.0 + dot) / 2.0, min=1e-6))       # (N, 1)
    xyz = torch.nn.functional.normalize(axis, dim=-1) * torch.sqrt(
        torch.clamp((1.0 - dot) / 2.0, min=0.0)
    )                                                                  # (N, 3)

    return torch.cat([w, xyz], dim=-1)                                 # (N, 4)

# ── Trajectory ────────────────────────────────────────────────────────────────

def generate_traj_params(num_envs, device, cfg):
    return {
        "amps":     torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.A + 0.5,
        "freqs":    torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.w + 0.2,
        "phases":   torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.phi * torch.pi,
        "z_offset": cfg.z_offset,
    }

def get_reference(t, dt, params):
    amps, freqs, phases = params["amps"], params["freqs"], params["phases"]
    t_tensor = torch.full((amps.shape[0], 1, 1), t * dt, device=amps.device)
    angle    = freqs * t_tensor + phases
    pos      = torch.sum(amps * torch.sin(angle),              dim=2)
    vel      = torch.sum(amps * freqs * torch.cos(angle),      dim=2)
    acc      = torch.sum(-amps * freqs**2 * torch.sin(angle),  dim=2)
    pos[:, 2] += params["z_offset"]
    return pos, vel, acc


# ── Environment ───────────────────────────────────────────────────────────────

def get_observation(states, pos_ref, vel_ref, acc_ref):
    return torch.cat([
        pos_ref - states[:, 0:3],
        vel_ref - states[:, 3:6],
        acc_ref,
        states[:, 6:10],
        states[:, 10:13],
    ], dim=-1)                          # (N, 16)

def compute_loss(states, pos_ref, vel_ref, acc_ref, weights, mask=None):
    if mask is not None and not mask.any():
        return states.sum() * 0.0

    s, p, v, a = (x[mask] if mask is not None else x
                  for x in (states, pos_ref, vel_ref, acc_ref))

    gravity    = torch.tensor([0., 0., -9.81], device=s.device)
    thrust_dir = torch.nn.functional.normalize(a + gravity, dim=-1)
    body_z     = quat_to_rotmat(s[:, 6:10])[:, :, 2]
    att_loss   = (1.0 - (body_z * thrust_dir).sum(dim=-1).clamp(-1, 1)).mean()

    return (
        weights.pos        * torch.linalg.norm(p - s[:, 0:3], dim=-1).mean() +
        weights.vel        * torch.linalg.norm(v - s[:, 3:6], dim=-1).mean() +
        weights.att        * att_loss +
        weights.body_rates * (s[:, 10:13] ** 2).sum(dim=-1).mean()
    )

def reset_states(cfg, traj_params):
    pos0, vel0, acc0 = get_reference(0, cfg.dt, traj_params)
    states = torch.zeros((cfg.num_envs, 13), device=cfg.device)
    states[:, 0:3]  = pos0.detach()
    states[:, 3:6]  = vel0.detach()
    states[:, 6:10] = acc_to_quat(acc0.detach())
    return states

def reset_terminated(states, terminated, pos_ref, vel_ref, acc_ref):
    if not terminated.any():
        return states
    idx = terminated.nonzero(as_tuple=True)[0]
    states[idx, 0:3]   = pos_ref[idx].detach()
    states[idx, 3:6]   = vel_ref[idx].detach()
    states[idx, 6:10]  = acc_to_quat(acc_ref[idx].detach())
    states[idx, 10:13] = 0.0
    return states


# ── Train ─────────────────────────────────────────────────────────────────────

def train(cfg: DictConfig):
    start     = time.time()
    out_dir   = "/home/adame/torchAirBender/outputs/policies/CTBR"
    os.makedirs(out_dir, exist_ok=True)

    dt, device, num_envs = cfg.dt, cfg.device, cfg.num_envs

    print(f"\n{'='*75}")
    print(f"  CTBR  |  Envs: {num_envs}  |  Episodes: {cfg.episodes}  |  Steps: {cfg.steps}  |  Horizon: {cfg.truncation}")
    print(f"{'='*75}\n")

    quadrotor  = QuadrotorDynamics(cfg)
    controller = CTBRController(
        hover_thrust = quadrotor.get_hover_thrust(),
        alloc_matrix = quadrotor._alloc_matrix,
        J            = quadrotor.J,
        dt           = dt,
        hover_ratio  = cfg.env.max_mass_norm_thrust,
        w_max        = cfg.env.w_max,
        kp_rate      = cfg.env.kp_rate,
    )
    policy    = MLP(cfg.env.policy, nn.ReLU, nn.Sigmoid(), output_bias_init=0.0).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)

    best_loss = float("inf")
    traj_env0 = torch.empty((cfg.steps, 20), device=device)

    for ep in range(cfg.episodes):
        traj_params                = generate_traj_params(num_envs, device, cfg.env.traj)
        states                     = reset_states(cfg, traj_params)
        randomized_params          = randomize_parameters(cfg.dynamics, num_envs, device)
        quadrotor.set_parameters(randomized_params)
        controller.update_parameters(
            hover_thrust = quadrotor.get_hover_thrust(),
            alloc_matrix = quadrotor._alloc_matrix,
            J            = quadrotor.J,
        )

        ep_loss, num_updates = 0.0, 0
        sq_error_sum         = torch.zeros(1, device=device)
        num_samples          = 0
        window_loss          = torch.zeros(1, device=device)
        window_start         = 0

        for t in range(cfg.steps):
            if t % cfg.truncation == 0:
                states       = states.detach()
                window_start = t
                window_loss  = torch.zeros(1, device=device)

            pos_ref, vel_ref, acc_ref = get_reference(t, dt, traj_params)

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            actions = controller(policy(obs), states[:, 10:13])
            states  = quadrotor.step(states, actions)

            dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target

            sq_error_sum += (dist ** 2).sum()
            num_samples  += dist.numel()
            window_loss  += compute_loss(
                states, pos_ref, vel_ref, acc_ref, cfg.env.loss_weights, mask=~too_far
            )

            if ep == cfg.episodes - 1:
                traj_env0[t] = torch.cat(
                    [states[0].detach(), actions[0].detach(), pos_ref[0].detach()], dim=0
                )

            if (t + 1) % cfg.truncation == 0 or (t + 1) == cfg.steps:
                loss = window_loss / ((t + 1) - window_start)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss     += loss.item()
                num_updates += 1

            states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)

        avg_loss = ep_loss / max(num_updates, 1)
        rmse     = torch.sqrt(sq_error_sum / num_samples).item()
        print(f"  Episode {ep+1:>4}/{cfg.episodes}  |  Loss: {avg_loss:.4f}  |  RMSE: {rmse:.3f} m")

        if rmse < cfg.env.rmse_threshold:
            print(f"  >> rsme < {cfg.env.rmse_threshold} w: {cfg.env.traj.w:.2f} → {cfg.env.traj.w + 0.25:.2f} 🔥")
            torch.save(policy.state_dict(), os.path.join(out_dir, f"{cfg.env.name}_w{cfg.env.traj.w:.2f}.pt"))
            cfg.env.traj.w += 0.25

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), os.path.join(out_dir, f"{cfg.env.name}_best.pt"))

    torch.save(policy.state_dict(), os.path.join(out_dir, f"{cfg.env.name}_final.pt"))
    print(f"\nTotal training time: {time.time() - start:.1f}s")

    TrajectoryTrackingRenderer(
        ref_trajectory = traj_env0[:, 17:20].cpu().numpy(),
        trajectory     = traj_env0.cpu().numpy(),
        arm_length     = float(randomized_params.arm_length[0].cpu()),
        arm_angle      = float(randomized_params.arm_angle[0].cpu()),
        mass           = float(randomized_params.mass[0].cpu()),
        dt             = dt,
    ).run()