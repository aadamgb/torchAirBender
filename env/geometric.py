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

def get_observation(
    states:  Tensor,
    p_ref: Tensor,   # (N, 3)
    v_ref: Tensor,   # (N, 3)
    a_ref: Tensor,   # (N, 3)
) -> Tensor:
    p = states[:, 0:3]
    v = states[:, 3:6]
    q = states[:, 6:10]
    w = states[:, 10:13]
    R = quat_to_rotmat(q)          # (N, 3, 3)
    R_flat = R.reshape(-1, 9)      # (N, 9)  — better than quaternion for the network? (q and -q represent the same attitude)

    p_error = p_ref - p          # (N, 3)
    v_error = v_ref - v          # (N, 3)

    return torch.cat([p_error, v_error, a_ref, R_flat, w], dim=-1)  # (N, 21)

def compute_loss(
    states:   Tensor,        # (N, 13)
    p_ref:  Tensor,        # (N, 3)
    v_ref:  Tensor,        # (N, 3)
    Fz:       Tensor,        # (N, 1)  current thrust command
    Fz_prev:  Tensor,        # (N, 1)  previous thrust command
    R_des:    Tensor,        # (N, 3, 3)
    R_des_prev: Tensor,      # (N, 3, 3)
    weights:  DictConfig,
    dt:       float,
) -> Tensor:                 # scalar

    p = states[:, 0:3]
    v = states[:, 3:6]

    # --- Primary tracking errors ---
    pos_loss = F.mse_loss(p, p_ref)
    vel_loss = F.mse_loss(v, v_ref)

    # --- Smoothness: rate of change of Fz and R_des ---
    fz_jerk  = F.mse_loss(Fz, Fz_prev) / dt

    # geodesic distance between consecutive desired attitudes
    RdTRd_prev = torch.bmm(R_des.transpose(-1, -2), R_des_prev)
    skew       = RdTRd_prev - RdTRd_prev.transpose(-1, -2)
    eR_dot     = 0.5 * torch.stack(
        [skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1
    )                                                        # (N, 3)
    
    att_jerk   = (eR_dot.norm(dim=-1) / dt).mean()

    total = (
        weights.pos       * pos_loss   +
        weights.vel       * vel_loss   +
        weights.jerk      * fz_jerk    +
        weights.jerk      * att_jerk   
    )

    # return individual terms too for logging
    return total, {
        "pos":      pos_loss.item(),
        "vel":      vel_loss.item(),
        "fz_jerk":  fz_jerk.item(),
        "att_jerk": att_jerk.item()
    }

# ==============================================================
# Train
# ==============================================================
def train(cfg: DictConfig):
    start = time.time()
    output_dir = "/home/adame/torchAirBender/outputs/policies/PP"
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


    last_ep_trajectory = torch.empty((steps, 20), device=device)  # 13 state + 4 actions + 3 ref pos

    quadrotor = QuadrotorDynamics(cfg)
    quadrotor.set_parameters(randomize_parameters(cfg.dynamics, num_envs, device))
    controller = AttitudeGeometricController(
        allocation_matrix=quadrotor._alloc_matrix,
        J=quadrotor.J
    )

    policy = MLP(layer_sizes=cfg.env.policy, activation=nn.ReLU, 
                output_activation=nn.Sigmoid(), output_bias_init=0.0).to(device)
    
    optimizer  = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)


    # ------- load the racing trajectory ------
    loaded_trajectory = load_TOGT(trajectory_path,  device=device)

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

    best_loss = float("inf")
    for ep in range(episodes):
        # Randomize batched drones each new episode
        quadrotor.set_parameters(randomize_parameters(cfg.dynamics, num_envs, device))
        controller.update_params(quadrotor._alloc_matrix, quadrotor.J)

        # Initial conditions
        states = torch.zeros((num_envs, 13), device=device) 
        states[0, 2] = 1.5; states[:, 6]   = 1.0             # initialize at z=1.5 and pointing upwards
        actions = torch.zeros((num_envs, 4), device=device)

        episode_loss   = 0.0
        num_updates    = 0
        # actions = quadrotor.get_srt_hover().unsqueeze(1).expand(num_envs, 4).clone()  # hover thrust for testing
        for t in range(steps):

            if t % truncation == 0:
                states       = states.detach()

            p_ref, v_ref, a_ref, _ = get_reference(t)

            obs = get_observation(states, p_ref, v_ref, a_ref)
            raw_actions = policy(obs)

            R_des = quat_to_rotmat(raw_actions[:, :4])
            Fz = raw_actions[:, 4].unsqueeze(-1)
            actions = controller(states, R_des, Fz)
            
            states = quadrotor.step(states, actions)

            break

        
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