import torch
import numpy as np
from omegaconf import DictConfig

from utils.randomize import randomize_parameters
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, SRT, CTBR
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
    controller = CTBR(
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
    traj_env0 = torch.empty((steps, n+1), device=device)

    setpoint = torch.zeros(num_envs, device=device)
    # print(setpoint)

    for t in range(steps):
        if t == 200:
            setpoint.fill_(1.0)

        cmd = torch.stack([
            torch.full_like(setpoint, 4 * srt_hover[0]),  # Hover Thrust
            setpoint,                                         # Pitch Rate
            torch.zeros_like(setpoint),                       # Roll Rate
            torch.zeros_like(setpoint),                       # Yaw Rate
        ], dim=1)
        action = controller(state, cmd)
        state = quadrotor.step(state, action[:, 0:4])
        # print(action[:, 0:4])
        # print(state[:, :])
        traj_env0[t] = torch.cat([
            state[0], 
            action[0, 0:4], 
            setpoint[0].view(-1)
            ], dim=0)   
        
    traj_np = traj_env0.detach().cpu().numpy()
    # renderer = BaseRenderer(
    #     trajectory=traj_env0.detach().cpu().numpy(), 
    # ).run()


    import matplotlib.pyplot as plt
    # print(traj_np[:-1])
    plt.plot(traj_np[:, -1])
    plt.plot(traj_np[:, 10])
    # plt.plot(traj_np[:, 2])
    plt.xlim([0, steps])
    plt.ylim([0, 1.1])
    plt.show()


    import os
    save_path = "/home/adame/torchAirBender/val/data/airbndr/traj_data.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, traj_np)