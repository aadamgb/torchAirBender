import hydra
import numpy as np
from omegaconf import DictConfig
from env_gym.trajectory_tracking import TrajTrckEnv


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    env = TrajTrckEnv(cfg, render_mode="human")
    env.reset()
    
    for t in range(cfg.steps):
        # Sample random actions for all N envs at once
        action = np.random.uniform(0.0, 1.0, size=(cfg.num_envs, 4)).astype(np.float32)
        env.step(action)
    
    env.render()


if __name__ == "__main__":
    main()