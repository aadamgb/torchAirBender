import torch

class SRT:
    def __init__(self, max_thrust=20.0, min_thrust=0.5):
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust

    def __call__(self, state, raw):
        return raw * (self.max_thrust - self.min_thrust) + self.min_thrust

    def update_params(self, alloc_matrix=None, J=None, max_thrust=None, min_thrust=None):
        if alloc_matrix is not None:
            self.alloc_inv = torch.linalg.pinv(alloc_matrix)
        if J is not None:
            self.J = J
        if max_thrust is not None:
            self.max_thrust = max_thrust.reshape(-1, 1) if torch.is_tensor(max_thrust) else max_thrust
        if min_thrust is not None:
            self.min_thrust = min_thrust.reshape(-1, 1) if torch.is_tensor(min_thrust) else min_thrust

class CTBR:
    def __init__(self, alloc_matrix, J, max_thrust=20.0, min_thrust=0.5, max_rate=10.0, kp_rate=1.0, dt=0.01):
        self.max_thrust = max_thrust * 4.0
        self.min_thrust = min_thrust * 4.0 # TODO: fix this xd 
        self.max_rate   = max_rate
        self.kp_rate    = kp_rate
        self.alloc_inv  = torch.linalg.pinv(alloc_matrix)
        self.J          = J
        self.dt         = dt

    def __call__(self, state, raw):
        w = state[:, 10:13]

        Fz    =  raw[:, 0:1] * (self.max_thrust - self.min_thrust) + self.min_thrust                        # (N, 1)  [0,1] -> [0, max_thrust]
        w_des = (raw[:, 1:4] * 2.0 - 1.0) * self.max_rate            # (N, 3)  [0,1] -> [-max_rate, max_rate]

        tau    = self.J * (self.kp_rate * (w_des - w) / self.dt)     # (N, 3)
        wrench = torch.cat([Fz, tau], dim=-1)                        # (N, 4)

        return torch.bmm(self.alloc_inv, wrench.unsqueeze(-1)).squeeze(-1)  # (N, 4) rotor thrusts

    def update_params(self, alloc_matrix=None, J=None, max_thrust=None, min_thrust=None):
        if alloc_matrix is not None:
            self.alloc_inv = torch.linalg.pinv(alloc_matrix)
        if J is not None:
            self.J = J
        if max_thrust is not None:
            self.max_thrust = max_thrust.reshape(-1, 1) if torch.is_tensor(max_thrust) else max_thrust
        if min_thrust is not None:
            self.min_thrust = min_thrust.reshape(-1, 1) if torch.is_tensor(min_thrust) else min_thrust