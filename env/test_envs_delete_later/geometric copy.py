import os
import time
import torch
from torch import nn
from torch.functional import F, Tensor
from omegaconf import DictConfig

from utils.replay import RacingRenderer
from utils.math import quat_to_rotmat

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import AttitudeGeometricController, DFGC
from miscellaneous.loader import load_gates_from_yaml, load_TOGT

from utils.nn import MLP 
from utils.randomize import QuadrotorParams, randomize_parameters


def acc_to_quat(acc_ref: Tensor, g: float = 9.81) -> Tensor:
    """
    Computes desired quaternion from reference acceleration.
    Desired thrust direction = acc_ref + gravity_vector.

    Args:
        acc_ref : (N, 3)
    Returns:
        q_des   : (N, 4)  [w, x, y, z]
    """
    gravity    = torch.tensor([0.0, 0.0, g], device=acc_ref.device)
    thrust_dir = acc_ref + gravity
    thrust_dir = torch.nn.functional.normalize(thrust_dir, dim=-1)

    world_z       = torch.zeros_like(thrust_dir)
    world_z[:, 2] = 1.0

    axis = torch.linalg.cross(world_z, thrust_dir)                   # (N, 3)
    dot  = (world_z * thrust_dir).sum(dim=-1, keepdim=True)          # (N, 1)

    w   = torch.sqrt(torch.clamp((1.0 + dot) / 2.0, min=1e-6))
    xyz = torch.nn.functional.normalize(axis, dim=-1) * torch.sqrt(
        torch.clamp((1.0 - dot) / 2.0, min=0.0)
    )

    return torch.cat([w, xyz], dim=-1)   

def reset_terminated(
    states:     Tensor,
    terminated: Tensor,    # (N,) bool
    pos_ref:    Tensor,    # (N, 3)
    vel_ref:    Tensor,    # (N, 3)
    acc_ref:    Tensor,    # (N, 3)
) -> Tensor:
    """Snap terminated envs back to current reference position and velocity."""
    if not terminated.any():
        return states

    idx = terminated.nonzero(as_tuple=True)[0]

    states[idx, 0:3]   = pos_ref[idx].detach()
    states[idx, 3:6]   = vel_ref[idx].detach()
    states[idx, 6:10]  = acc_to_quat(acc_ref[idx].detach())
    states[idx, 10:13] = 0.0

    return states

def get_observation(
    states:  Tensor,
    p_ref:   Tensor,   # (N, 3)
    v_ref:   Tensor,   # (N, 3)
    a_ref:   Tensor,   # (N, 3)
    b1d_ref: Tensor,   # (N, 3)  
) -> Tensor:
    p = states[:, 0:3]
    v = states[:, 3:6]
    q = states[:, 6:10]
    w = states[:, 10:13]
    R_flat = quat_to_rotmat(q).reshape(-1, 9)   # (N, 9)

    p_error = p_ref - p          # (N, 3)
    v_error = v_ref - v          # (N, 3)

    return torch.cat([p_error, v_error, a_ref, R_flat, w, b1d_ref], dim=-1)  # (N, 24)

def compute_loss(
    states:   Tensor,        # (N, 13)
    p_ref:  Tensor,        # (N, 3)
    v_ref:  Tensor,        # (N, 3)
    # Fz:       Tensor,        # (N, 1)  current thrust command
    # Fz_prev:  Tensor,        # (N, 1)  previous thrust command
    # R_des:    Tensor,        # (N, 3, 3)
    # R_des_prev: Tensor,      # (N, 3, 3)
    weights:  DictConfig,
    dt:       float,
    mask:       Tensor | None = None,
) -> Tensor:

    # early exit if all envs are crashed
    if mask is not None and not mask.any():
        # print("All environments terminated ⚠️")
        return states.sum() * 0.0

    # apply mask
    s     = states[mask] if mask is not None else states
    p     = p_ref[mask]  if mask is not None else p_ref
    v     = v_ref[mask]  if mask is not None else v_ref
    # R_d   = R_des[mask]  if mask is not None else R_des
    # R_dp  = R_des_prev[mask] if mask is not None else R_des_prev
    # fz    = Fz[mask]     if mask is not None else Fz
    # fz_p  = Fz_prev[mask] if mask is not None else Fz_prev

    # --- Primary tracking: L1 norm like the other env, smoother near zero ---
    pos_loss  = torch.linalg.norm(p - s[:, 0:3], dim=-1).mean()
    vel_loss  = torch.linalg.norm(v - s[:, 3:6], dim=-1).mean()

    # --- Body rate penalty (from other env) ---
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    # # --- Smoothness: Fz jerk ---
    # fz_jerk   = F.mse_loss(fz, fz_p) / dt

    # # --- Smoothness: R_des geodesic jerk ---
    # RdTRd_prev = torch.bmm(R_d.transpose(-1, -2), R_dp)
    # skew       = RdTRd_prev - RdTRd_prev.transpose(-1, -2)
    # eR_dot     = 0.5 * torch.stack(
    #     [skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1
    # )
    # att_jerk   = (eR_dot.norm(dim=-1) / dt).mean()

    total = (
        weights.pos             * pos_loss  +
        weights.vel             * vel_loss  +
        weights.body_rates      * rate_loss #+
        # weights.jerk_Fz         * fz_jerk   +
        # weights.jerk_att        * att_jerk
    )

    return total

# ==============================================================
# Train
# ==============================================================
def train(cfg: DictConfig):
    start = time.time()
    output_dir = "/home/adame/torchAirBender/outputs/policies/GEO"
    os.makedirs(output_dir, exist_ok=True)
    track_path = cfg.env.track_path
    trajectory_path = cfg.env.trajectory_path

    dt         = cfg.dt
    device     = cfg.device
    num_envs   = cfg.num_envs
    episodes   = cfg.episodes
    steps      = cfg.steps
    truncation = cfg.truncation

    print(f"\n{'='*75}")
    print(f" Training the Geometric controller")
    print(f"  Envs: {num_envs}  |  Episodes: {episodes}  |  Steps: {steps}  |  Horizon: {truncation}")
    print(f"{'='*75}\n")


    last_ep_env0_trajectory = torch.empty((steps, 20), device=device)  # 13 state + 4 actions + 3 ref pos

    quadrotor = QuadrotorDynamics(cfg)
    quadrotor.set_parameters(randomize_parameters(cfg.dynamics, num_envs, device))

    controller = AttitudeGeometricController(
        allocation_matrix=quadrotor._alloc_matrix,
        J=quadrotor.J
    )

    policy = MLP(layer_sizes=cfg.env.policy, activation=nn.ReLU).to(device)
    
    optimizer  = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)


    # ------- load the racing trajectory ------
    loaded_trajectory = load_TOGT(trajectory_path,  device=device)

    def get_reference(t: int, speed_scale=0.2):
        # map t -> scaled index into the trajectory
        scaled_t = min(int(t * speed_scale), len(loaded_trajectory["pos"]) - 1)
        pos = loaded_trajectory["pos"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1)
        vel = loaded_trajectory["vel"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1) * speed_scale
        acc = loaded_trajectory["acc"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1) * speed_scale ** 2

        # Compute b1d as in main loop
        v_heading    = loaded_trajectory["vel"][scaled_t].clone()
        v_heading[2] = 0.0
        b1d          = F.normalize(v_heading, dim=-1).unsqueeze(0).expand(num_envs, -1)  # (N, 3)

        return pos, vel, acc, b1d

    best_loss = float("inf")
    for ep in range(episodes):
        # Randomize batched drones each new episode
        quadrotor.set_parameters(randomize_parameters(cfg.dynamics, num_envs, device))
        # print(quadrotor.get_parameters())
        controller.update_params(quadrotor._alloc_matrix, quadrotor.J, kR=5.0, kw=1.0)

        # ── initial conditions ────────────────────────────────────────────
        pos0, _, _, _  = get_reference(0)
        states         = torch.zeros((num_envs, 13), device=device)
        states[:, 0:3] = pos0.detach()
        states[:, 6]   = 1.0

        ep_loss      = 0.0
        num_updates  = 0
        window_loss  = torch.zeros(1, device=device)
        window_start = 0
        Fz_prev      = None                                
        R_des_prev   = None                               
        # actions = quadrotor.get_srt_hover().unsqueeze(1).expand(num_envs, 4).clone()  # hover thrust for testing
        for t in range(steps):
            # ── forward pass ─────────────────────────────────────────────
            p_ref, v_ref, a_ref, b1d_ref = get_reference(t)
            # Override reference to fixed values for all environments
            # p_ref = torch.tensor([[0.0, 0.0, 1.5]], device=device).expand(num_envs, -1)
            # v_ref = torch.zeros((num_envs, 3), device=device)
            # a_ref = torch.zeros((num_envs, 3), device=device)
            # b1d_ref = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(num_envs, -1)

            obs        = get_observation(states, p_ref, v_ref, a_ref, b1d_ref)
            raw        = policy(obs)                                # (N, 7): 4 quat + 1 Fz

            
            # q_des      = F.normalize(raw[:, :4], dim=-1)           # (N, 4) normalize quaternion output so it lies on unit sphere
            # # q_des[:]   = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_des.device)  # set to identity quaternion for all envs
            # Fz         = F.softplus(raw[:, 4]).unsqueeze(-1)    * 80.0   # (N, 1)  always positive
            # # print(Fz[0])
            # print(q_des[0])
            # R_des      = quat_to_rotmat(q_des)                     # (N, 3, 3)

            # # --- decode network output ---
            # A          = raw[:, 0:3]                                        # (N, 3) free, no constraint
            # cs         = F.normalize(raw[:, 3:5], dim=-1)                  # (N, 2) unit circle yaw
            # b1d        = torch.stack(
            #     [cs[:, 0], cs[:, 1], torch.zeros(num_envs, device=device)],
            #     dim=-1
            # )                                                               # (N, 3)

            # # --- compute R_des and Fz exactly like geometric controller ---
            # Fz         = A.norm(dim=-1, keepdim=True)                      # (N, 1) always positive
            # b3d        = F.normalize(A, dim=-1)                            # (N, 3)
            # b2d        = F.normalize(torch.linalg.cross(b3d, b1d), dim=-1)# (N, 3)
            # R_des      = torch.stack(
            #     [torch.linalg.cross(b2d, b3d), b2d, b3d], dim=-1
            # )                                                               # (N, 3, 3)

            # actions    = controller(states, R_des, Fz)
            actions = raw
            states     = quadrotor.step(states, actions)


            # ── terminated envs mask ─────────────────────────────────────────────
            dist    = torch.linalg.norm(p_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target
            alive   = ~too_far

             # ── loss ─────────────────────────────────────────────────────
            step_loss = compute_loss(
                states     = states,
                p_ref      = p_ref,
                v_ref      = v_ref,
                # Fz         = Fz,
                # Fz_prev    = Fz_prev if Fz_prev is not None else Fz.detach(),
                # R_des      = R_des,
                # R_des_prev = R_des_prev if R_des_prev is not None else R_des.detach(),
                weights    = cfg.env.loss_weights,
                dt         = dt,
                mask       = alive
            )
            window_loss = window_loss + step_loss
            # Fz_prev     = Fz.detach()
            # R_des_prev  = R_des.detach()

            # ── TBPTT: update at end of each window and at the end of the ep ──────────────────────
            if (t % truncation == 0 and t > 0) or t + 1 == steps:
                window_len = (t + 1) - window_start
                loss       = window_loss / window_len

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                ep_loss     += loss.item()
                num_updates += 1
                window_loss  = torch.zeros(1, device=device)
                window_start = t + 1
                Fz_prev      = None
                R_des_prev   = None
                states       = states.detach()   # ← cut graph after update

            states = reset_terminated(states, too_far, p_ref, v_ref, a_ref)

            if ep == episodes - 1:
                last_ep_env0_trajectory[t] = torch.cat(
                    [states[0].detach(), actions[0].detach(), p_ref[0].detach()],
                    dim=0,
                )


        avg_loss = ep_loss / max(num_updates, 1)
        print(f"  Episode {ep+1:4d}/{episodes} | Loss: {avg_loss:.4f} ")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), os.path.join(output_dir, "policy_best.pt"))
            # print(f"  >> New best: {best_loss:.4f}")

    # ==== Replay the reajectory ====
    gates_position, gates_rpy = load_gates_from_yaml(track_path)
    renderer = RacingRenderer(
          gates_position=gates_position,
          gates_rpy=gates_rpy,
          ref_trajectory=last_ep_env0_trajectory[:, 17:20].cpu().numpy(),
          trajectory=last_ep_env0_trajectory[:, :17].cpu().numpy(),
          arm_length=float(quadrotor.arm_length[0].cpu().numpy()),
          arm_angle=float(quadrotor.arm_angle[0].cpu().numpy()),
          mass=float(quadrotor.m[0].cpu().numpy()),
          dt=dt,
    )
    renderer.run()
# ==============================================================
# Test
# ==============================================================
def test(cfg: DictConfig):
    dt = cfg.dt
    device = cfg.device
    track_path = cfg.env.track_path
    trajectory_path = cfg.env.trajectory_path
    
    loaded_trajectory = load_TOGT(trajectory_path,  device=device)

    steps = loaded_trajectory["pos"].shape[0]

    def get_reference(t: int, speed_scale=1.0):
        # map t -> scaled index into the trajectory
        scaled_t = min(int(t * speed_scale), len(loaded_trajectory["pos"]) - 1)
        pos = loaded_trajectory["pos"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1)
        vel = loaded_trajectory["vel"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1) * speed_scale
        acc = loaded_trajectory["acc"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1) * speed_scale ** 2

        # Compute b1d as in main loop
        v_ref_heading = vel[0].clone()
        v_ref_heading[2] = 0.0
        v_ref_heading = v_ref_heading / (torch.norm(v_ref_heading) + 1e-6)
        b1d = v_ref_heading.unsqueeze(0)

        return pos, vel, acc, b1d

    # create quadrotor object and controller
    quadrotor = QuadrotorDynamics(cfg)

    controller = DFGC(
        alloc_matrix=quadrotor._alloc_matrix,
        J=quadrotor.J,
        m=quadrotor.m,
        g=quadrotor.g,
    )

    state = torch.zeros((1, 13), device=device); state[0, 2] = 1.5; state[:, 6]   = 1.0
    actions = torch.zeros((1, 4), device=device)

    full_trajectory = torch.empty((steps, 20), device=device)  # 13 state + 4 actions + 3 ref pos

    #=========================
    # Simulate
    #=========================
    sq_error_sum = 0.0
    with torch.inference_mode():
          for t in range(steps):
            # Reference trajectory
            p_ref, v_ref, a_ref, b1d = get_reference(t)

            actions = controller(state, p_ref, v_ref, a_ref, b1d)
            state  = quadrotor.step(state=state, action=actions)

            dist    = torch.linalg.norm(p_ref - state[:, 0:3], dim=-1)
            sq_error_sum += (dist**2).item()  # accumulate mse for logging

            full_trajectory[t] = torch.cat([state[0], actions[0], p_ref[0]], dim=0)

    
    print(f"RMSE Pos: {(sq_error_sum / steps) ** 0.5:.4f} m")

    # ==== Replay the reajectory ====
    gates_position, gates_rpy = load_gates_from_yaml(track_path)
    renderer = RacingRenderer(
          gates_position=gates_position,
          gates_rpy=gates_rpy,
          ref_trajectory=full_trajectory[:, 17:20].cpu().numpy(),
          trajectory=full_trajectory[:, :17].cpu().numpy(),
          arm_length=float(quadrotor.arm_length.cpu().numpy()),
          arm_angle=float(quadrotor.arm_angle.cpu().numpy()),
          mass=float(quadrotor.m.cpu().numpy()),
          dt=dt,
    )
    renderer.run()


    # ---------- Plotting some stuff for analysis (delte later on) ------------
    # import matplotlib.pyplot as plt
    # import numpy as np

    # # convert to cpu numpy
    # traj_np = full_trajectory.cpu().numpy()

    # actions = traj_np[:, 13:17]  # 4 motors
    # time = np.arange(steps) * dt

    # plt.figure(figsize=(10,5))

    # plt.plot(time, actions[:,0], label="motor 1")
    # plt.plot(time, actions[:,1], label="motor 2")
    # plt.plot(time, actions[:,2], label="motor 3")
    # plt.plot(time, actions[:,3], label="motor 4")

    # plt.xlabel("Time [s]")
    # plt.ylabel("Motor command")
    # plt.title("Quadrotor Actions Over Rollout")
    # plt.legend()
    # plt.grid(True)

    # plt.show()