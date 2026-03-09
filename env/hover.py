import torch
from omegaconf import DictConfig

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.srt_controller import SRTController
from utils.randomize import randomize_parameters
from utils.replay import BaseRenderer

def train(cfg: DictConfig):
    dt = cfg.dt
    device = cfg.device
    num_envs = cfg.num_envs
    steps = cfg.steps
    
    print(f"\n{'='*25}")
    print("Trainng for hover")
    print(f"Num envs: {num_envs}")
    print(f"Steps: {steps}")
    print(f"dt: {dt}")
    print(f"{'='*25}")

    quadrotor = QuadrotorDynamics(cfg)

    # Generting key for reproducability
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    randomized_params = randomize_parameters(
        cfg=cfg.dynamics,
        num_envs=num_envs,
        device=device,
        generator=generator,
    )

    quadrotor.set_parameters(randomized_params)
    print(randomized_params)

    states = torch.zeros((num_envs, 13), device=device)
    states[:, 2] = 0.5
    states[:, 6] = 1.0  # seting quaternion w to 1 
    
    # Get hover thrust
    srt_hover = quadrotor.get_srt_hover()
    controller   = SRTController(srt_hover, hover_ratio=cfg.env.max_mass_norm_thrust)
    actions_raw = torch.full((num_envs, 4), 0.45, device=device)

    print(actions_raw)

    actions = controller(actions_raw)

    print(actions)

    # Store the trajectory
    traj_env0 = torch.empty((steps, 17), device=device)

    #-------------------------------------------
    # Main Simulation Loop
    #------------------------------------------
    for t in range(steps):
        states = quadrotor.step(state=states, action=actions)
        traj_env0[t] = torch.cat([states[0], actions[0]], dim=0)   # store only first env


    # Replay the trajectory
    renderer = BaseRenderer(
        trajectory=traj_env0.detach().cpu().numpy(),  # (T, 13)
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()