import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from omegaconf import DictConfig

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, CTBR
from utils.randomize import randomize_parameters
from utils.trajectory import TrajectoryManager
from utils.math import acc_to_quat, quat_to_rotmat

from utils.replay_multi import MultiDroneRenderer


class QuadrotorEnv(gym.Env):
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: DictConfig, render_mode="human"):
        super().__init__()
        self.cfg       = cfg
        self.num_envs  = cfg.num_envs
        self.device    = cfg.device
        self.dt        = cfg.dt
        self.max_steps = cfg.steps

        self.render_mode = render_mode
        self.traj_data = torch.empty((self.max_steps, 34), device=self.device)  
        self._record   = True 

        # --- Observation and Action spaces ---
        obs_dim = 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # --- Build the Controller (using CTBR only) ---
        self.quadrotor  = QuadrotorDynamics(cfg)
        alloc           = DirectAllocation(self.quadrotor._alloc_matrix)
        self.controller = CTBR(
            allocator = alloc,
            mass      = self.quadrotor.m,
            max_TWR   = self.quadrotor.max_TWR,
            J         = self.quadrotor.J,
            max_rate  = cfg.env.w_max,
            kp_rate   = cfg.env.kp_rate,
            dt        = cfg.dt,
        ) 

        # --- Build the Trajectory Manager ---
        self.traj = TrajectoryManager.from_harmonics(
            cfg.env.traj, self.num_envs, self.device
        )

        # -- Init Some Params ---
        self.states   = None   # (N, 17)
        self.t        = 0      # current timestep within episode


    # ============================================================
    # Reset Method
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Randomize and Update ---
        self.traj.randomize()
        params = randomize_parameters(self.cfg.dynamics, self.num_envs, self.device)
        self.quadrotor.set_parameters(params)
        self.controller.update_params(
            alloc_matrix = self.quadrotor._alloc_matrix,
            mass         = self.quadrotor.m,
            max_TWR      = self.quadrotor.max_TWR,
            J            = self.quadrotor.J,
        )

        # --- Initialize UAV at Reference Traj ---
        pos0, vel0, acc0, _ = self.traj.get_reference(0)
        self.states = torch.zeros((self.num_envs, 17), device=self.device)
        self.states[:, 0:3]  = pos0.detach()
        self.states[:, 3:6]  = vel0.detach()
        self.states[:, 6:10] = acc_to_quat(acc0.detach())
        self.t = 0

        obs  = self._get_obs()
        info = {}
        return obs, info

    # ============================================================
    # Step Method
    # ============================================================
    def step(self, action: np.ndarray):

        action_t = torch.tensor(action, dtype=torch.float32, device=self.device) 
        pos_ref, vel_ref, acc_ref, _ = self.traj.get_reference(self.t)


        # TODO: Add bool to not use customgrad when trianing whit PPo
        actions_full = self.controller(self.states, action_t)
        self.states  = self.quadrotor.step(self.states, actions_full[:, 0:4])
        self.t      += 1

        body_z     = quat_to_rotmat(self.states[:, 6:10])[:, :, 2]
        gravity    = torch.tensor([0., 0., -9.81], device=self.device)
        thrust_dir = torch.nn.functional.normalize(acc_ref + gravity, dim=-1)

        # --- Reward --- TODO: Match it with BPTT loss function ❗❗⚠️️⚠️❗❗❗⚠️⚠️
        pos_sq = torch.sum((pos_ref - self.states[:, 0:3])**2, dim=-1)
        vel_sq = torch.sum((vel_ref - self.states[:, 3:6])**2, dim=-1)
        
        pos_loss = torch.linalg.norm(pos_ref - self.states[:, 0:3], dim=-1)
        vel_loss = torch.linalg.norm(vel_ref - self.states[:, 3:6], dim=-1)
        rates = (self.states[:, 10:13]**2).sum(dim=-1)
        att   = (1.0 - (body_z * thrust_dir).sum(dim=-1).clamp(-1, 1))
        # survival = torch.ones(self.num_envs, device=self.device) * 0.5

        reward_t = -( self.cfg.env.loss_weights.pos   * pos_loss + 
                      self.cfg.env.loss_weights.vel   * vel_loss +
                      self.cfg.env.loss_weights.att   *    att +
                      self.cfg.env.loss_weights.rates * rates) #+ survival
        
        reward = reward_t.cpu().numpy().astype(np.float32)
    
        # --- Termination ---
        terminated = (pos_sq > self.cfg.env.max_dist_to_target**2).cpu().numpy()    

        # --- Reset Terminated  ---
        if terminated.any():
            idx = torch.from_numpy(np.where(terminated)[0]).to(self.device)
            self.states[idx, 0:3]  = pos_ref[idx].detach()
            self.states[idx, 3:6]  = vel_ref[idx].detach()
            self.states[idx, 6:10] = acc_to_quat(acc_ref[idx].detach())
            self.states[idx, 10:]  = 0.0
            
            # termination penalty
            # reward[terminated]    -= 5.0  
            
        # --- Truncation ---
        truncated = np.full(self.num_envs, self.t >= self.max_steps, dtype=bool)

        # --- Save States for Rendering ---
        if self._record and self.t < self.max_steps:
            pos_ref, vel_ref, acc_ref, _ = self.traj.get_reference(self.t)
            self.traj_data[self.t] = torch.cat([
                self.states[0].detach(),    # 0:17  — state  
                pos_ref[0].detach(),        # 17:20 — p_ref
                vel_ref[0].detach(),        # 20:23 — v_ref
                acc_ref[0].detach(),        # 23:26 — a_ref
                actions_full[0].detach(),   # 26:34 — motor thrusts + wrench
            ], dim=0)

        obs  = self._get_obs()

        # self.pos_mse_mean = pos_sq.mean().item()
        # self.vel_mse_mean = vel_sq.mean().item()
        self.pos_mse_mean = pos_loss.mean().item()
        self.vel_mse_mean = vel_loss.mean().item()
        self.reward_mean   = reward_t.mean().item()

        info = {} 

        return obs, reward, terminated, truncated, info
    
    # ============================================================
    # Get Observation
    # ============================================================
    def _get_obs(self) -> np.ndarray:
        pos_ref, vel_ref, acc_ref, _ = self.traj.get_reference(self.t)
        obs = torch.cat([
            pos_ref - self.states[:, 0:3],
            vel_ref - self.states[:, 3:6],
            acc_ref,
            self.states[:, 6:10],
            self.states[:, 10:13],
        ], dim=-1)  # (N, 16)
        return obs.cpu().numpy().astype(np.float32)

    # ============================================================
    # Rendering
    # ============================================================
    def enable_recording(self):
        self._record = True
        self.traj_data = torch.empty((self.max_steps, 34), device=self.device)

    def render(self):
        if self.render_mode == "human":
            traj_np = self.traj_data.cpu().numpy()
            print(f"traj_data shape: {traj_np.shape}")
            MultiDroneRenderer(
                trajectory     = traj_np,
                ref_trajectory = traj_np[:, 17:20],
            ).run()

    