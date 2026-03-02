import torch
from omegaconf import DictConfig

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from utils.randomize import randomize_parameters

def train(cfg: DictConfig):
    print(f"\n{'='*25}")
    print("Trainng for hover")
    print(f"{'='*25}")

    num_envs = cfg.num_envs
    device = cfg.device


    quadrotor = QuadrotorDynamics(cfg)

    randomized_params = randomize_parameters(
        cfg=cfg.dynamics,
        num_envs=num_envs,
        device=device
    )

    quadrotor.set_parameters(randomized_params)

    # print(quadrotor.get_parameters())


    states = torch.zeros((num_envs, 13), device=device)
    actions = torch.ones((num_envs, 4), device=device) * 2

    print(states)

    for t in range(500):
        # print(f"Time step {t}")
        actions = torch.ones((num_envs, 4), device=device) * 2
        # print(states)
        next_states = quadrotor.step(state=states, action=actions)
        states = next_states

    print("Final state")
    print(states)