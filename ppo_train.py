import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO

from env_gym.trajectory_tracking import TrajTrckEnv
from env_gym.wrappers import TorchVecEnv
from env_gym.callbacks import TrajTrackingCallback


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    env     = TrajTrckEnv(cfg, render_mode="human")
    vec_env = TorchVecEnv(env)

    model = PPO(
        policy          = "MlpPolicy",
        policy_kwargs = dict(
            net_arch  = cfg.env.hidden_layers,  
            activation_fn = torch.nn.ReLU),
        env             = vec_env,
        n_steps         = cfg.steps,
        batch_size      = cfg.num_envs,
        learning_rate   = cfg.env.lr,
        gamma           = cfg.env.ppo.gamma,
        verbose         = 1,
        tensorboard_log = "/home/adame/torchAirBender/outputs/policies/PPO/tb_logs",
    )

    callback = TrajTrackingCallback() # For loggin custom metrics
    
    try:
        model.learn(
            total_timesteps = cfg.env.ppo.total_timesteps,
            tb_log_name     = "traj_tracking_run",
            callback        = callback,
        )
        model.save("outputs/policies/PPO/tt_ctbr")

    finally:
        # Always runs, even if training is interrupted
        print("\nRunning deterministic rollout for rendering...")
        env.enable_recording()
        obs, _ = env.reset()
        for t in range(cfg.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main()