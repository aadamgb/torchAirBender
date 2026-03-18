import os
import time
import torch
from torch import Tensor
from torch import nn

from omegaconf import DictConfig, OmegaConf             # params loading
import pandas as pd                                     # for trajectory loading

from utils.nn import MLP 
from utils.randomize import QuadrotorParams, randomize_parameters
from utils.replay import TrajectoryTrackingRenderer
from utils.math import quat_to_rotmat

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers_old import CTBRController#, SRTController


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

def reset(
        cfg: DictConfig,
        traj_params: dict,
) -> tuple[Tensor, QuadrotorParams]:
    """Reset all envs to the trajectory's t=0 position and velocity."""
    num_envs = cfg.num_envs
    device   = cfg.device

    pos0, vel0, acc0 = get_target(0.0, traj_params)                   # (N, 3) each

    states = torch.zeros((num_envs, 13), device=device)
    states[:, 0:3]  = pos0.detach()
    states[:, 3:6]  = vel0.detach()                                # match target velocity
    states[:, 6:10] = acc_to_quat(acc0.detach())   
    # states[:, 6]    = 1.0                                          # quaternion w = 1

    params = randomize_parameters(cfg.dynamics, num_envs, device)

    return states, params


def reset_terminated(
        states: Tensor,
        terminated: Tensor,     # (N,) bool
        pos_ref: Tensor,        # (N, 3) current reference position
        vel_ref: Tensor,        # (N, 3) current reference velocity
        acc_ref: Tensor,        # (N, 3) current reference velocity
) -> Tensor:
    """Snap terminated envs back to current reference position and velocity."""
    if not terminated.any():
        return states

    idx = terminated.nonzero(as_tuple=True)[0]

    states[idx, 0:3]  = pos_ref[idx].detach()
    states[idx, 3:6]  = vel_ref[idx].detach()                     # match target velocity
    states[idx, 6:10] = acc_to_quat(acc_ref[idx].detach())
    states[idx, 10:13] = 0.0
    # states[idx, 6]    = 1.0                                        # quaternion w = 1

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

    return torch.cat([p_error, v_error, acc_ref, q, w], dim=-1)  # (N, 13)

def compute_loss(
    states: Tensor,
    pos_ref: Tensor,
    vel_ref: Tensor,
    acc_ref: Tensor,
    weights: DictConfig,
    mask: Tensor | None = None,
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

    # Orientation loss: angle between actual body-z and desired thrust direction
    gravity     = torch.tensor([0.0, 0.0, -9.81], device=s.device)
    thrust_dir  = torch.nn.functional.normalize(a + gravity, dim=-1)  # (N, 3)
    R           = quat_to_rotmat(s[:, 6:10])                          # (N, 3, 3)
    body_z      = R[:, :, 2]                                          # (N, 3) — third column
    cos_angle   = (body_z * thrust_dir).sum(dim=-1).clamp(-1, 1)      # (N,)
    att_loss    = (1.0 - cos_angle).mean()                            # 0 when aligned, 2 when opposite


    return (weights.pos * pos_loss + weights.vel * vel_loss +  
            weights.att * att_loss + weights.body_rates * rate_loss)



def train(cfg: DictConfig):
    start = time.time()
    output_dir = "/home/adame/torchAirBender/outputs/policies/CTBR"
    os.makedirs(output_dir, exist_ok=True)

    dt         = cfg.dt
    device     = cfg.device
    num_envs   = cfg.num_envs
    episodes   = cfg.episodes
    steps      = cfg.steps
    truncation = cfg.truncation

    print(f"\n{'='*75}")
    print(f"  CTBR no encoder")
    print(f"  Envs: {num_envs}  |  Episodes: {episodes}  |  Steps: {steps}  |  Horizon: {truncation}")
    print(f"{'='*75}\n")

    states = torch.zeros((num_envs, 13), device=device)
    truncation_losses = torch.empty(truncation, device=device)
    traj_env0 = torch.empty((steps, 20), device=device)               # 13 state + 4 actions + 3 ref pos

    quadrotor = QuadrotorDynamics(cfg)

    controller   = CTBRController(
        hover_thrust = quadrotor.get_hover_thrust(),
        alloc_matrix = quadrotor._alloc_matrix,
        J            = quadrotor.J,
        dt           = dt,
        hover_ratio  = cfg.env.max_mass_norm_thrust,
        w_max        = cfg.env.w_max,       
        kp_rate      = cfg.env.kp_rate,     
    )
    policy = MLP(layer_sizes=cfg.env.policy, activation=nn.ReLU, 
                 output_activation=nn.Sigmoid(), output_bias_init=0.0).to(device)

    optimizer  = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)

    best_loss = float("inf")
    for ep in range(episodes):

        traj_params       = generate_trajectory_params(num_envs, device, cfg.env.traj)

        # --- episode reset ---
        states, randomized_params = reset(cfg, traj_params)
        quadrotor.set_parameters(randomized_params)

        controller.update_parameters(
            hover_thrust = quadrotor.get_hover_thrust(),
            alloc_matrix = quadrotor._alloc_matrix,
            J            = quadrotor.J,
        )

        ep_loss    = 0.0
        num_updates = 0

        # To print the RSME
        sq_error_sum = torch.zeros(1, device=device)
        num_samples = 0

        # lambda_recon = 0.05 

        for t in range(steps):

            if t % truncation == 0:
                states       = states.detach()
                window_start = t
                window_loss_sum = torch.zeros(1, device=device)


            # --- reference at current time ---
            pos_ref, vel_ref, acc_ref = get_target(t * dt, traj_params)  # (N, 3) each

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            raw     = policy(obs)
            actions = controller(raw, states[:, 10:13])
            states  = quadrotor.step(states, actions)

            # --- termination ---
            dist = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)  # (N,)
            too_far = dist > cfg.env.max_dist_to_target
            alive = ~too_far

            sq_error_sum += (dist**2).sum()
            num_samples += dist.numel()

            step_loss = compute_loss(states, pos_ref, vel_ref, acc_ref, cfg.env.loss_weights, mask=alive)
            truncation_losses[t % truncation] = step_loss.detach()
            window_loss_sum = window_loss_sum + step_loss

            # Save the last trajectory for rendering
            if ep == episodes - 1:
                traj_env0[t] = torch.cat([states[0].detach(), actions[0].detach(), pos_ref[0].detach(),], dim=0)

            if (t + 1) % truncation == 0 or (t + 1) == steps:
                window_len = (t + 1) - window_start
                loss = window_loss_sum / window_len #+ lambda_recon * recon_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss     += truncation_losses[:window_len].mean().item()
                num_updates += 1

            # --- reset terminated envs ---
            states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)


        avg_ep_loss = ep_loss / num_updates
        rmse = torch.sqrt(sq_error_sum / num_samples).item()
        print(f"  Episode {ep+1:>4}/{episodes}  |  Avg Loss: {avg_ep_loss:.4f}  |  RMSE: {rmse:.3f} m")

        rmse_threshold = 0.15
        if rmse < rmse_threshold:
            print(f"RSME below {rmse_threshold} m increasing w = {cfg.env.traj.w} by 0.25 🔥")
            torch.save(policy.state_dict(), os.path.join(output_dir, f"{cfg.env.name}_w_{cfg.env.traj.w}.pt"))
            cfg.env.traj.w += 0.25


        if avg_ep_loss < best_loss:
            best_loss = avg_ep_loss
            torch.save(policy.state_dict(), os.path.join(output_dir, f"{cfg.env.name}_best.pt"))

    print(f"Total training time: {time.time() - start:.2f}s")
    torch.save(policy.state_dict(), os.path.join(output_dir, f"{cfg.env.name}_final.pt"))

    renderer = TrajectoryTrackingRenderer(
        ref_trajectory=traj_env0[:, 17:20].cpu().numpy(),
        trajectory=traj_env0.cpu().numpy(),
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()


    ### Plotting some stuff for analysis (delte later on)

    import matplotlib.pyplot as plt
    import numpy as np

    # convert to cpu numpy
    traj_np = traj_env0.cpu().numpy()

    actions = traj_np[:, 13:17]  # 4 motors
    time_plt = np.arange(steps) * dt

    plt.figure(figsize=(10,5))

    plt.plot(time_plt, actions[:,0], label="motor 1")
    plt.plot(time_plt, actions[:,1], label="motor 2")
    plt.plot(time_plt, actions[:,2], label="motor 3")
    plt.plot(time_plt, actions[:,3], label="motor 4")

    plt.xlabel("Time [s]")
    plt.ylabel("Motor command")
    plt.title("Quadrotor Actions Over Rollout")
    plt.legend()
    plt.grid(True)

    plt.show()