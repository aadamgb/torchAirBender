import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrajTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env.env  # access the underlying QuadTrajEnv
        self.logger.record("analysis/reward_mean", env.reward_mean)
        self.logger.record("analysis/pos_mse", env.pos_mse_mean)
        self.logger.record("analysis/vel_mse", env.vel_mse_mean)
        return True