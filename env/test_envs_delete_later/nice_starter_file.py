import os
import time
import torch
from torch.functional import F
from omegaconf import DictConfig

from utils.replay import RacingRenderer
from utils.math import quat_to_rotmat

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers_old import AttitudeGeometricController, DFGeometricController
from miscellaneous.loader import load_gates_from_yaml, load_TOGT



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
    controller = DFGeometricController(
        alloc_matrix=quadrotor._alloc_matrix,
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