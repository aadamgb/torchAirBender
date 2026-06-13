import torch
from torch.functional import F
from utils.math import quat_to_rotmat

class DirectAllocation:
    def __init__(self, alloc_matrix):
        self.alloc_inv = torch.linalg.pinv(alloc_matrix)

    def __call__(self, Fz, tau):
        wrench = torch.cat([Fz, tau], dim=-1)           
        return torch.bmm(self.alloc_inv, wrench.unsqueeze(-1)).squeeze(-1)

    def update_params(self, alloc_matrix):
        self.alloc_inv = torch.linalg.pinv(alloc_matrix)


# -----------------------------------------------
# Single Rotor Thrust Controller
# -----------------------------------------------
class SRT:
    def __init__(self, mass, max_TWR, min_thrust=0.5):
        self.mass             = mass
        self.max_TWR          = max_TWR
        self.max_total_thrust = (max_TWR * mass * 9.81).unsqueeze(-1)
        self.max_rotor_thrust = (max_TWR * mass * 9.81 / 4.0).unsqueeze(-1)
        self.min_thrust       = min_thrust

    def __call__(self, state, raw):
        return raw * (self.max_rotor_thrust - self.min_thrust) + self.min_thrust  

    def update_params(self, alloc_matrix=None, J=None, mass=None, max_TWR=None, min_thrust=None):
        if mass is not None or max_TWR is not None:
            self.mass             = mass    if mass    is not None else self.mass
            self.max_TWR          = max_TWR if max_TWR is not None else self.max_TWR
            self.max_total_thrust = (self.max_TWR * self.mass * 9.81).unsqueeze(-1)
            self.max_rotor_thrust = (self.max_TWR * self.mass * 9.81 / 4.0).unsqueeze(-1)
        if min_thrust is not None:
            self.min_thrust = min_thrust

# -----------------------------------------------
# Collective Thrust and Body Rates Controller
# -----------------------------------------------
class CTBR:
    def __init__(self, allocator, J, mass, max_TWR, min_thrust=0.5, max_rate=10.0, kp_rate=1.0, dt=0.01):
        self.allocator  = allocator
        self.mass       = mass
        self.max_TWR    = max_TWR
        self.max_total_thrust = (max_TWR * mass * 9.81).unsqueeze(-1)
        self.min_thrust = min_thrust * 4.0 # TODO: fix this xd 
        self.max_rate   = max_rate
        self.kp_rate    = kp_rate
        self.J          = J
        self.dt         = dt

    def __call__(self, state, raw):
        w = state[:, 10:13]
        Fz    =  raw[:, 0:1] * (self.max_total_thrust - self.min_thrust) + self.min_thrust
        w_des = (raw[:, 1:4] * 2.0 - 1.0) * self.max_rate
        tau = self.J * (self.kp_rate * (w_des - w) / self.dt)  
        wrench = torch.cat([Fz, tau], dim=-1)
        return torch.cat([self.allocator(Fz, tau), wrench], dim=-1)
    

    def update_params(self, alloc_matrix=None, mass=None, J=None, max_TWR=None, min_thrust=None):
        if alloc_matrix is not None:
            self.allocator.update_params(alloc_matrix)
        if J is not None:
            self.J = J
        if mass is not None or max_TWR is not None:
            self.mass    = mass    if mass    is not None else self.mass
            self.max_TWR = max_TWR if max_TWR is not None else self.max_TWR
            self.max_total_thrust = (self.max_TWR * self.mass * 9.81).unsqueeze(-1)
        if min_thrust is not None:
            self.min_thrust = min_thrust * 4.0  

# -----------------------------------------------
# Linear Velocities and Headung Rate Controller
# -----------------------------------------------
class LVHR:
    def __init__(
        self,
        allocator,
        m, J, g, 
        kv=0.5, kR=0.15, kw=0.05,   # TODO: Hardcoded for now...
        max_vel=20.0,
        max_yaw_rate=4.0,
        # gain_scale = [0.5, 0.15, 0.05]   # Only for gain scheduling TODO: Hardcoded for now...
        gain_scale = [0.4, 0.3, 0.05]   # Only for gain scheduling TODO: Hardcoded for now...
    ):
        self.allocator    = allocator   
        self.m            = m
        self.J            = J
        self.g            = g
        self.kv           = kv
        self.kR           = kR
        self.kw           = kw
        self.max_vel      = max_vel
        self.max_yaw_rate = max_yaw_rate
        self.gain_scale   = torch.tensor(gain_scale, dtype=torch.float32, device="cuda")  #TODO: Ugly fix...

    def update_params(self, alloc_matrix=None, mass=None, max_TWR=None, J=None, kv=None, kR=None, kw=None):
        if alloc_matrix is not None:
            self.allocator.update_params(alloc_matrix)
        if kv is not None:
            self.kv = kv
        if kR is not None:
            self.kR = kR
        if kw is not None:
            self.kw = kw

    def __call__(self, state, raw, gains=None):
        # Gain scheduling if cm is lvyr_g
        if gains is not None:
            gains = F.softplus(gains) * self.gain_scale    
            kv = gains[:, 0]   
            kR = gains[:, 1]   
            kw = gains[:, 2]   
        else:
            kv, kR, kw = self.kv, self.kR, self.kw

        vel  = state[:, 3:6]                                           
        quat = state[:, 6:10]                                           
        w    = state[:, 10:13]                                          

        R = quat_to_rotmat(quat)                                        
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        # --- Decode actions ---
        v_des       = (raw[:, 0:3] * 2.0 - 1.0) * self.max_vel          
        yaw_rate_des = (raw[:, 3:4] * 2.0 - 1.0) * self.max_yaw_rate   

        # --- Desired force vector in world frame ---
        a_des = kv * (v_des - vel)                                      

        a_des[:, 2] += self.g                                           
        f_des = self.m * a_des                                          

        b3_des = F.normalize(f_des, dim=-1, eps=1e-6)

        # --- Desired x-body-axis from current yaw heading ---
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw),
                               torch.zeros_like(yaw)], dim=-1)  

        ## b2_des = b3_des x heading, b1_des = b2_des x b3_des
        b2_des = F.normalize(torch.linalg.cross(b3_des, heading), dim=-1, eps=1e-6)
        b1_des = torch.linalg.cross(b2_des, b3_des)

        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)  

        # --- SO(3) attitude error (vee map of skew-symmetric part) ---
        RdTR   = torch.bmm(R_des.transpose(-1, -2), R)
        skew   = RdTR - RdTR.transpose(-1, -2)                                                
        eR = 0.5 * torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)         

        RTRd   = torch.bmm(R.transpose(-1, -2), R_des)
        w_des = torch.zeros_like(w)
        w_des[:, 2:3] = yaw_rate_des
        eW = w - torch.bmm(RTRd, w_des.unsqueeze(-1)).squeeze(-1)

        gyro = torch.linalg.cross(w, self.J * w)

        # --- Total thrust magnitude projected onto body z ---
        b3_cur = R[:, :, 2]                                     
        Fz = (f_des * b3_cur).sum(dim=-1, keepdim=True)         
        
        Fz = F.softplus(Fz)
        tau    = -kR * eR - kw * eW + gyro        # TODO: Think wether to add the feedforward term...
        wrench = torch.cat([Fz, tau], dim=-1)

        if gains is not None:
            out = torch.cat([self.allocator(Fz, tau), wrench, v_des, yaw_rate_des, kv.unsqueeze(-1), kR.unsqueeze(-1), kw.unsqueeze(-1)], dim=-1)
        else:
            out = torch.cat([self.allocator(Fz, tau), wrench, v_des, yaw_rate_des], dim=-1)

        return out