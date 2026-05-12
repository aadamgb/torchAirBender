import hydra
import importlib
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from env_gym.wrappers import TorchVecEnv
from env_gym.callbacks import TrajTrackingCallback


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    try:
        environment = importlib.import_module(f"env_gym.{cfg.env.name}")
    except ImportError:
        print(f"Error: Env '{cfg.env.name}' not found in env_gym/ folder.")
        return
    
    env     = environment.QuadrotorEnv(cfg, render_mode="human")
    vec_env = VecNormalize(TorchVecEnv(env), norm_obs=True, norm_reward=True)    

    model = PPO(
        policy          = "MlpPolicy",
        policy_kwargs = dict(
            net_arch  = cfg.env.hidden_layers,  
            activation_fn = torch.nn.ReLU),
        env             = vec_env,
        # n_steps         = cfg.steps,
        n_steps         = 64,
        batch_size      = 1024,
        learning_rate   = cfg.env.lr,
        n_epochs=4,
        gamma=0.999,
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