import torch
from omegaconf import DictConfig

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from utils.randomize import randomize_parameters
from utils.replay import PositionControlRenderer

# from controller.pd_controller import ProportionalDerivative


def train(cfg: DictConfig):
    dt = cfg.dt
    device = cfg.device
    num_envs = cfg.num_envs
    steps = cfg.steps
    
    print(f"\n{'='*25}")
    print("Position Control")
    print(f"Num envs: {num_envs}")
    print(f"Steps: {steps}")
    print(f"dt: {dt}")
    print(f"{'='*25}")

    quadrotor = QuadrotorDynamics(cfg)
    # controller = ProportionalDerivative

    # Generting key for reproducability
    # generator = torch.Generator(device=device)
    # generator.manual_seed(42)
    randomized_params = randomize_parameters(
        cfg=cfg.dynamics,
        num_envs=num_envs,
        device=device,
        # generator=generator,
    )
    quadrotor.set_parameters(randomized_params)

    # Generate random target position
    target = torch.rand((num_envs, 3), device=device)


    states = torch.zeros((num_envs, 13), device=device)
    states[:, 2] = 0.5
    states[:, 6] = 1.0  # seting quaternion w to 1 
    
    # Get hover thrust
    srt_hover = (quadrotor.get_srt_hover())
    print(srt_hover)
    print(srt_hover[0])
    actions = srt_hover.unsqueeze(1).expand(-1, 4)

    # Store the trajectory (states + actions)
    traj_env0 = torch.empty((steps, 17), device=device)  # 13 state + 4 actions

    #-------------------------------------------
    # Main Simulation Loop
    #------------------------------------------
    for t in range(steps):
        states = quadrotor.step(state=states, action=actions)
        actions = torch.tensor([srt_hover[0] *1.05, srt_hover[0] * 0.95, srt_hover[0]*1.05, srt_hover[0] * 0.95], device=device).expand(num_envs, -1)
        # Concatenate first env state and actions
        traj_env0[t] = torch.cat([states[0], actions[0]], dim=0)


    # Replay the trajectory
    print(randomized_params)
    renderer = PositionControlRenderer(
        target_pos=target[0].detach().cpu().numpy(),
        trajectory=traj_env0.detach().cpu().numpy(),  # (T, 13)
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()