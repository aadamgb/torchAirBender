import torch
from omegaconf import DictConfig

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from utils.randomize import randomize_parameters
from utils.replay import BaseRenderer

def train(cfg: DictConfig):
    print(f"\n{'='*25}")
    print("Trainng for hover")
    print(f"{'='*25}")

    dt = cfg.dt
    device = cfg.device
    num_envs = cfg.num_envs
    steps = cfg.steps

    print(f"Num envs: {num_envs}")
    print(f"Steps {num_envs}")
    print(f"Steps {dt}")
    
    quadrotor = QuadrotorDynamics(cfg)

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    randomized_params = randomize_parameters(
        cfg=cfg.dynamics,
        num_envs=num_envs,
        device=device,
        generator=generator,
    )


    quadrotor.set_parameters(randomized_params)
    # print(quadrotor.get_parameters())
          

    states = torch.zeros((num_envs, 13), device=device)
    states[:, 6] = 1.0  # set w=1 for valid identity quaternion 
    actions = torch.ones((num_envs, 4), device=device) * 2


    # Store the trajectory
    traj_env0 = torch.empty((steps, 13), device=device)
    for t in range(steps):
        # actions = torch.ones((num_envs, 4), device=device) * 2
        # actions = torch.tensor([1.5, 2.0, 2.0, 1.5], device=device).unsqueeze(0).expand(num_envs, -1)
        actions = torch.tensor([2.0, 2.0, 2.0, 2.0], device=device).unsqueeze(0).expand(num_envs, -1)
        states = quadrotor.step(state=states, action=actions)
        # next_states = torch.compile(quadrotor.step, mode="reduce-overhead")
        traj_env0[t] = states[0]   # store only first env

    print("Final state")
    # print(traj_env0)



    # after your sim loop...
    renderer = BaseRenderer(
        trajectory=traj_env0.detach().cpu().numpy(),  # (T, 13)
        arm_length=float(randomized_params.arm_length[0].cpu()),
        dt=dt,
    )
    renderer.run()