# import torch
# from torch import nn

class StateLogger:
    def __init__(self, env_idx: int = 0):
        """Logs a single environment for plotting."""
        self.idx = env_idx
        self.p_true,  self.p_noisy  = [], []
        self.v_true,  self.v_noisy  = [], []
        self.w_true,  self.w_noisy  = [], []

    def log(self, states, noisy_states):
        i = self.idx
        self.p_true.append(states[i, 0:3].cpu().numpy())
        self.p_noisy.append(noisy_states[i, 0:3].cpu().numpy())
        self.v_true.append(states[i, 3:6].cpu().numpy())
        self.v_noisy.append(noisy_states[i, 3:6].cpu().numpy())
        self.w_true.append(states[i, 10:13].cpu().numpy())
        self.w_noisy.append(noisy_states[i, 10:13].cpu().numpy())

    def plot(self, dt: float):
        import matplotlib.pyplot as plt
        import numpy as np

        p  = np.array(self.p_true);  pn = np.array(self.p_noisy)
        v  = np.array(self.v_true);  vn = np.array(self.v_noisy)
        w  = np.array(self.w_true);  wn = np.array(self.w_noisy)
        t  = np.arange(len(p)) * dt

        labels = [("x", "y", "z")]
        specs  = [
            (p,  pn, "Position",     ["x [m]",     "y [m]",     "z [m]"    ]),
            (v,  vn, "Velocity",     ["vx [m/s]",  "vy [m/s]",  "vz [m/s]" ]),
            (w,  wn, "Body rates",   ["p [rad/s]", "q [rad/s]", "r [rad/s]"]),
        ]

        for true, noisy, title, ylabels in specs:
            fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            fig.suptitle(title)
            for i, ax in enumerate(axes):
                ax.plot(t, true[:, i],  label="true",  linewidth=1.5)
                ax.plot(t, noisy[:, i], label="noisy", linewidth=1.0, alpha=0.7)
                ax.set_ylabel(ylabels[i])
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("time [s]")
            fig.tight_layout()

        plt.show()