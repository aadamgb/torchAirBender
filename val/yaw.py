import torch
import numpy as np
from omegaconf import DictConfig

from utils.randomize import randomize_parameters
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, SRT, CTBR_TEST
from utils.replay_old_w_fpv import BaseRenderer

def validate(cfg: DictConfig):
    dt = cfg.dt
    device = cfg.device
    num_envs = cfg.num_envs
    steps = cfg.steps

    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    randomized_params = randomize_parameters(
        cfg=cfg.dynamics,
        num_envs=num_envs,
        device=device,
        generator=generator,
    )

    quadrotor = QuadrotorDynamics(cfg)
    controller = SRT(
        mass=quadrotor.m,
        max_TWR=quadrotor.max_TWR,        
    )
    quadrotor.set_parameters(randomized_params)

    state = torch.zeros((num_envs, 17), device=device)
    state[:, 2] = 0.5
    state[:, 6] = 1.0  # seting quaternion w to 1

    srt_hover = quadrotor.get_srt_hover_thurst()
    raw = torch.zeros((num_envs, 4), device=device)
    action = torch.full((num_envs, 4), srt_hover[0], device=device)

    alloc = DirectAllocation(quadrotor._alloc_matrix)
    controller = CTBR_TEST(
            allocator=alloc,
            mass=quadrotor.m,
            max_TWR=quadrotor.max_TWR,
            J=quadrotor.J,
            dt=cfg.dt,
            kp_rate=0.2
        )
    # action = controller(state, raw)

    # Store the trajectory
    n = state.size(1) + action.size(1)
    traj_env0 = torch.empty((steps, n+2), device=device)

    t_vec = torch.linspace(0, steps * dt, steps, device=device)
    f0, f1, T = 0.1, 2.0, steps * dt
    phase = 2 * np.pi * (f0 * t_vec + (f1 - f0) / (2 * T) * t_vec**2)
    chirp_signal = 1 * torch.sin(phase)
    setpoint = torch.zeros(num_envs, device=device)

    for t in range(steps):
        # setpoint = chirp_signal[t].expand(num_envs)
        if t == 100:
            setpoint = torch.full_like(setpoint, 200.0)
        # setpoint = 6.0 * torch.sin(t_vec[t]).expand(num_envs)
        # setpoint = (0.0 + (6.0 * t * dt/T)) * torch.sin(2 * t_vec[t]).expand(num_envs)

        cmd = torch.stack([
            # torch.full_like(setpoint, 4 * srt_hover[0]),      # Hover Thrust
            setpoint,      # Hover Thrust
            torch.zeros_like(setpoint),                       # Pitch Rate
            torch.zeros_like(setpoint),                       # Roll Rate
            torch.zeros_like(setpoint),                                           # Yaw Rate
        ], dim=1)
        action = controller(state, cmd)
        state = quadrotor.step(state, action[:, 0:4])
        # print(action[:, 0:4])
        # print(state[:, :])
        traj_env0[t] = torch.cat([
            state[0], 
            action[0, 0:5], 
            setpoint[0].view(-1)
            ], dim=0)   
        
    traj_np = traj_env0.detach().cpu().numpy()
    # renderer = BaseRenderer(
    #     trajectory=traj_env0.detach().cpu().numpy(), 
    # ).run()


    import matplotlib.pyplot as plt
    # s = setpoint[0].cpu().numpy()
    # print(s)
    # print(traj_np[:-1])
    plt.plot(traj_np[:, -1], label='ref', c='k', ls='--')
    # convert from rpm to rad/s: multiply by 2*pi/60
    plt.plot(traj_np[:, 16] * (2 * np.pi / 60.0), label='omega_z (rad/s)', c='tab:blue')
    # plt.plot(traj_np[:, 2])
    plt.xlim([0, steps])
    plt.xlim([0, 300])
    # plt.ylim([-6, 6 + 0.2])
    # plt.ylim([0, 1.5])
    plt.grid()
    plt.legend()
    plt.show()


    import os
    save_path = "/home/adame/torchAirBender/val/data/airbndr/th_step.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, traj_np)