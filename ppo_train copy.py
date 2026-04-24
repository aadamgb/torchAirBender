import hydra
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO

from env_gym.trajectory_tracking import TrajTrckEnv
from env_gym.wrappers import TorchVecEnv


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    env     = TrajTrckEnv(cfg, render_mode="juman")
    env.enable_recording()                # Record data for rendering
    vec_env = TorchVecEnv(env)

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        n_steps         = cfg.steps,      # steps per update per env
        batch_size      = cfg.num_envs,
        learning_rate   = cfg.env.lr,
        gamma           = cfg.env.ppo.gamma,
        verbose         = 1,
        tensorboard_log ="/home/adame/torchAirBender/outputs/policies/PPO/tb_logs",
    )
    model.learn(
        total_timesteps=cfg.env.ppo.total_timesteps,
        tb_log_name="traj_tracking_run"
    )
    model.save("outputs/policies/PPO/tt_ctbr")

    # Force a clean rollout for rendering after training
    obs, _ = env.reset()
    with torch.no_grad():
        for t in range(cfg.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

    env.render()  # now traj_data is cleanly filled


if __name__ == "__main__":
    main()