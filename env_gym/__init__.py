from gymnasium.envs.registration import register

register(
    id="TrajTrckEnv-v0",
    entry_point="env_gym.trajectory_tracking:TrajTrckEnv",
)