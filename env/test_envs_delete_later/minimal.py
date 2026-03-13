import os
import time
import torch
from torch import Tensor, nn

from omegaconf import DictConfig

from utils.replay import RacingRenderer
from utils.math import quat_to_rotmat

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from miscellaneous.loader import load_gates_from_yaml, load_TOGT

import matplotlib.pyplot as plt
import numpy as np



def test(cfg: DictConfig):
    dt = cfg.dt
    device = cfg.device
    track_path = cfg.env.track_path
    trajectory_path = cfg.env.trajectory_path
    

    loaded_trajectory = load_TOGT(trajectory_path,  device=device)

    steps = loaded_trajectory["pos"].shape[0]

    def get_reference(t: int, speed_scale=1.1):
            # map t -> scaled index into the trajectory
            scaled_t = min(int(t * speed_scale), len(loaded_trajectory["pos"]) - 1)
            pos = loaded_trajectory["pos"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1)
            vel = loaded_trajectory["vel"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1) * speed_scale
            acc = loaded_trajectory["acc"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1) * speed_scale ** 2
            return pos, vel, acc

    quadrotor = QuadrotorDynamics(cfg)

    states = torch.zeros((1, 13), device=device); states[0, 2] = 1.5
    actions = torch.zeros((1, 4), device=device)

    full_trajectory = torch.empty((steps, 20), device=device)  # 13 state + 4 actions + 3 ref pos

    #=========================
    # Simulate
    #=========================
    with torch.inference_mode():
          print(actions)
          for t in range(steps):
            pos_ref, vel_ref, acc_ref = get_reference(t)
            
            actions[0, :] = quadrotor.get_srt_hover() # just hover for now
            states  = quadrotor.step(state=states, action=actions)

            full_trajectory[t] = torch.cat([states[0], actions[0], pos_ref[0]], dim=0)





    # ==== Replay the reajectory ====
    gates_position, gates_rpy = load_gates_from_yaml(track_path)
    renderer = RacingRenderer(
          gates_position=gates_position,
          gates_rpy=gates_rpy,
          gate_mesh_path="/home/adame/torchAirBender/miscellaneous/gate.obj",
          ref_trajectory=full_trajectory[:, 17:20].cpu().numpy(),
          trajectory=full_trajectory[:, :17].cpu().numpy(),
          arm_length=float(quadrotor.arm_length.cpu().numpy()),
          arm_angle=float(quadrotor.arm_angle.cpu().numpy()),
          mass=float(quadrotor.m.cpu().numpy()),
          dt=dt,
    )
    renderer.run()
    ### Plotting some stuff for analysis (delte later on)



    # convert to cpu numpy
    # traj_np = traj.cpu().numpy()

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