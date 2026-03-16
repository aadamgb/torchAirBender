from omegaconf import DictConfig

from utils.randomize import randomize_parameters

def train(cfg: DictConfig):
    params = randomize_parameters(
        cfg=cfg.dynamics,
        num_envs=cfg.num_envs,
        device=cfg.device
    )


    params_cpu = {
        "mass": params.mass.cpu(),
        "J": params.J.cpu(),
        "arm_length": params.arm_length.cpu(),
        "arm_angle": params.arm_angle.cpu(),
        "km": params.km.cpu()
    }


    print("\nRandomized parameters per environment:\n")

    for i in range(cfg.num_envs):
        print(f"--- Environment {i} ---")
        print(f"Mass:        {params_cpu['mass'][i].item():.6f}")
        print(f"Arm length:  {params_cpu['arm_length'][i].item():.6f}")
        print(f"Arm angle:  {params_cpu['arm_angle'][i].item():.6f}")
        print(f"Torque constant km:  {params_cpu['km'][i].item():.6f}")
        print(f"Inertia J:   {params_cpu['J'][i].tolist()}")
