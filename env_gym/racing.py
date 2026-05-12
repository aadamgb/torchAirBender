import torch
from torch import Tensor, nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from omegaconf import DictConfig

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, CTBR
from utils.randomize import randomize_parameters
from utils.trajectory import TrajectoryManager, LemniscataTrajectory
from utils.math import acc_to_quat, quat_to_rotmat
from miscellaneous.loader import load_gates_from_yaml
from utils.replay_multi import RacingRenderer

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
        obs_dim = 31
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

        # --- Load the Track ---
        gates_position, gates_rpy = load_gates_from_yaml(cfg.env.track)
        self.gates_position = gates_position   # keep numpy for renderer
        self.gates_rpy      = gates_rpy
        self.gate_pos, self.gate_R = self._build_gate_tensors(gates_position, gates_rpy, self.device)
        self.num_gates = self.gate_pos.shape[0]

        # --- Build the Trajectory Manager  (for reference only) ---
        self.traj = LemniscataTrajectory(cfg.num_envs, cfg.device, scale=3, speed=4)

        # -- Init Some Params ---
        self.states   = None   # (N, 17)
        self.t        = 0      # current timestep within episode


    # ============================================================
    # Reset Method
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Randomize and Update ---
        params = randomize_parameters(self.cfg.dynamics, self.num_envs, self.device)
        self.quadrotor.set_parameters(params)
        self.controller.update_params(
            alloc_matrix = self.quadrotor._alloc_matrix,
            mass         = self.quadrotor.m,
            max_TWR      = self.quadrotor.max_TWR,
            J            = self.quadrotor.J,
        )

        # --- Spawn near gate 2 ---
        s_idx = torch.randint(0, self.gate_pos.shape[0], (1,), device=self.device).item()
        s_idx = 0
        first_R   = self.gate_R[s_idx]      # (3, 3)
        first_pos = self.gate_pos[s_idx]    # (3,)

        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 1] = -1.0
        offset[:, 0] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.4
        offset[:, 2] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.4
        spawn_world = (first_R @ offset.T).T + first_pos    # (N, 3)

        self.states = torch.zeros((self.num_envs, 17), device=self.device)
        self.states[:, 0:3] = spawn_world
        self.states[:, 6]   = 1.0      # unit quaternion w = 1

        # --- Gate tracking state ---
        self.gate_idx = torch.full((self.num_envs,), s_idx, dtype=torch.long, device=self.device)

        with torch.no_grad():
            rel_pos, _, _, _ = self._gate_relative_state(
                self.states[:, 0:3], self.states[:, 3:6],
                self.gate_idx, self.gate_pos, self.gate_R
            )
        self.prev_rel_y = rel_pos[:, 1].clone()
        self.prev_dist  = torch.linalg.norm(rel_pos, dim=-1).clone()

        self.t = 0

        obs  = self._get_obs()
        info = {}
        return obs, info

    # ============================================================
    # Step Method
    # ============================================================
    def step(self, action: np.ndarray):

        action_t = torch.tensor(action, dtype=torch.float32, device=self.device) 
        actions_full = self.controller(self.states, action_t)
        self.states  = self.quadrotor.step(self.states, actions_full[:, 0:4])
        self.t      += 1

        with torch.no_grad():
            rel_pos, rel_vel, gate_fwd, _ = self._gate_relative_state(
                self.states[:, 0:3], self.states[:, 3:6],
                self.gate_idx, self.gate_pos, self.gate_R
            )

        # dist_to_gate = torch.linalg.norm(
        #     self.states[:, 0:3] - self.gate_pos[self.gate_idx], dim=-1
        # )
        
        dist_to_gate = torch.linalg.norm(rel_pos, dim=-1)
        # print(dist_to_gate[0])    
        rate_penalty = (self.states[:, 10:13] ** 2).sum(dim=-1)

        progress = rel_vel[:, 1].clamp(min=0)      # velocity toward gate (gate-frame y-axis)
        approach = self.prev_dist - dist_to_gate   # positive = getting closer this step

        reward_t = (- self.cfg.env.pos_progress  * approach
                    - self.cfg.env.w_rate        * rate_penalty
                    + self.cfg.env.vel_progress  * progress)
        
        
        # crossing detection
        crossed = (self.prev_rel_y < 0.0) & (rel_pos[:, 1] >= 0.0)
        in_x    = rel_pos[:, 0].abs() < (1.2 / 2.0) # gate width
        in_z    = rel_pos[:, 2].abs() < (1.2 / 2.0) # gate heigth
        success = crossed & in_x & in_z
        miss    = crossed & ~(in_x & in_z)

        reward_t = reward_t + success.float() * self.cfg.env.r_success
        reward_t = reward_t - miss.float()    * self.cfg.env.r_miss

        # advance gate index
        self.gate_idx = torch.where(
            success,
            (self.gate_idx + 1) % self.num_gates,
            self.gate_idx,
        )
        self.prev_rel_y = rel_pos[:, 1].clone()

        # Recompute after gate_idx has been advanced so we check distance to the current target gate
        dist_to_gate = torch.linalg.norm(
            self.states[:, 0:3] - self.gate_pos[self.gate_idx], dim=-1
        )

        # Termination and partial reset
        too_far    = (dist_to_gate > self.cfg.env.max_dist_to_gate).cpu().numpy()
        terminated = too_far
        truncated  = np.full(self.num_envs, self.t >= self.max_steps, dtype=bool)

        # terminated_t = (dist_to_gate > self.cfg.env.max_dist_to_gate)
        # reward_t = reward_t - terminated_t.float() * self.cfg.env.r_terminate 
        # reward_t = reward_t + (~terminated_t).float() * self.cfg.env.r_survival

        reward = reward_t.detach().cpu().numpy().astype(np.float32)

        if terminated.any():
            terminated_t = torch.from_numpy(terminated).to(self.device)
            curr_R   = self.gate_R[self.gate_idx]
            curr_pos = self.gate_pos[self.gate_idx]

            offset = torch.zeros((self.num_envs, 3), device=self.device)
            offset[:, 1] = -1.0
            offset[:, 0] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.4
            offset[:, 2] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.4
            spawn = torch.bmm(curr_R, offset.unsqueeze(-1)).squeeze(-1) + curr_pos

            reset_states        = torch.zeros((self.num_envs, 17), device=self.device)
            reset_states[:, 0:3] = spawn
            reset_states[:, 6]   = 1.0

            self.states = torch.where(
                terminated_t.unsqueeze(-1).expand_as(self.states),
                reset_states,
                self.states,
            )
            # reset prev_rel_y for respawned envs so crossing detector starts clean
            self.prev_rel_y = torch.where(
                terminated_t,
                torch.full_like(self.prev_rel_y, -1.5),
                self.prev_rel_y,
            )

        # Refresh prev_dist after any respawns so the next step diffs against the right position
        self.prev_dist = torch.linalg.norm(
            self.states[:, 0:3] - self.gate_pos[self.gate_idx], dim=-1
        )

        # record for rendering
        if self._record and self.t < self.max_steps:
            pos_ref, vel_ref, acc_ref, _ = self.traj.get_reference(self.t)
            self.traj_data[self.t] = torch.cat([
                self.states[0].detach(),
                pos_ref[0].detach(),
                vel_ref[0].detach(),
                acc_ref[0].detach(),
                actions_full[0].detach(),
            ], dim=0)

        self.pos_mse_mean  = dist_to_gate.mean().item()
        self.reward_mean   = reward_t.mean().item()

        # return observation
        obs  = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info
    
    # ============================================================
    # Private functions
    # ============================================================
    # def _get_obs(self) -> np.ndarray:
    #     with torch.no_grad():
    #         rel_pos, rel_vel, gate_fwd = self._gate_relative_state(
    #             self.states[:, 0:3], self.states[:, 3:6],
    #             self.gate_idx, self.gate_pos, self.gate_R
    #         )
    #     obs = torch.cat([
    #         rel_pos,                # (N, 3)  position in gate frame
    #         rel_vel,                # (N, 3)  velocity in gate frame
    #         gate_fwd,               # (N, 3)  gate facing direction (world)
    #         self.states[:, 6:10],   # (N, 4)  quaternion
    #         self.states[:, 10:13],  # (N, 3)  body rates
    #     ], dim=-1)                  # (N, 16)
    #     return obs.cpu().numpy().astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        with torch.no_grad():
            rel_pos, rel_vel, gate_fwd, next_gate_fwd = self._gate_relative_state(
                self.states[:, 0:3], self.states[:, 3:6],
                self.gate_idx, self.gate_pos, self.gate_R
            )
            R = quat_to_rotmat(self.states[:, 6:10])   # (N, 3, 3)
            R_flat = R.reshape(self.num_envs, 9)        # (N, 9)

        obs = torch.cat([
            self.gate_idx.float().unsqueeze(-1),
            self.states[:, 0:3],                # (N, 3)  position in gate frame
            self.states[:, 3:6],                # (N, 3)  position in gate frame
            rel_pos,                # (N, 3)  position in gate frame
            rel_vel,                # (N, 3)  velocity in gate frame
            gate_fwd,               # (N, 3)  gate facing direction (world)
            next_gate_fwd,               # (N, 3)  gate facing direction (world)
            R_flat,                 # (N, 9)  rotation matrix (flattened)
            self.states[:, 10:13],  # (N, 3)  body rates
        ], dim=-1)                  # (N, 21)
        return obs.cpu().numpy().astype(np.float32)
    

    def _build_gate_tensors(
        self,    
        gates_position: "np.ndarray",
        gates_rpy: "np.ndarray",
        device: str,
    ):
        """Convert numpy gate arrays to torch tensors on device."""
        gate_pos = torch.tensor(gates_position, dtype=torch.float32, device=device)   # (G, 3)
        gate_rpy = torch.tensor(gates_rpy,      dtype=torch.float32, device=device)   # (G, 3)
        gate_R   = self._rpy_deg_to_rotmat(gate_rpy)                                       # (G, 3, 3)
        return gate_pos, gate_R 
    
    def _rpy_deg_to_rotmat(self, rpy_deg: Tensor) -> Tensor:
        """(..., 3) degrees -> (..., 3, 3) rotation matrices (Rz @ Ry @ Rx)."""
        rpy = torch.deg2rad(rpy_deg)
        r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]

        cr, sr = torch.cos(r), torch.sin(r)
        cp, sp = torch.cos(p), torch.sin(p)
        cy, sy = torch.cos(y), torch.sin(y)

        zero = torch.zeros_like(r)
        one  = torch.ones_like(r)

        Rx = torch.stack([
            torch.stack([one,  zero, zero], dim=-1),
            torch.stack([zero,  cr,  -sr], dim=-1),
            torch.stack([zero,  sr,   cr], dim=-1),
        ], dim=-2)
        Ry = torch.stack([
            torch.stack([ cp,  zero,  sp], dim=-1),
            torch.stack([zero,  one, zero], dim=-1),
            torch.stack([-sp,  zero,  cp], dim=-1),
        ], dim=-2)
        Rz = torch.stack([
            torch.stack([cy,  -sy,  zero], dim=-1),
            torch.stack([sy,   cy,  zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ], dim=-2)

        return Rz @ Ry @ Rx   # (..., 3, 3)
    

    def _gate_relative_state(
        self,
        drone_pos: Tensor,            # (N, 3) world frame
        drone_vel: Tensor,            # (N, 3) world frame
        gate_idx:  Tensor,            # (N,)   long
        gate_pos:  Tensor,            # (G, 3)
        gate_R:    Tensor,            # (G, 3, 3)  R_world_from_gate
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        rel_pos  : (N, 3)  drone position in gate frame
        rel_vel  : (N, 3)  drone velocity in gate frame
        gate_fwd : (N, 3)  gate facing direction (b2 column) in world frame
        """
        curr_pos = gate_pos[gate_idx]       # (N, 3)
        curr_R   = gate_R[gate_idx]         # (N, 3, 3)
        next_idx   = (gate_idx + 1) % self.num_gates
        next_currR = gate_R[next_idx]
        # print(curr_R[0, :, 1])

        # R^T maps world -> gate frame
        delta   = drone_pos - curr_pos                                      # (N, 3)
        rel_pos = torch.bmm(curr_R.transpose(1, 2), delta.unsqueeze(-1)).squeeze(-1)   # (N, 3)
        rel_vel = torch.bmm(curr_R.transpose(1, 2), drone_vel.unsqueeze(-1)).squeeze(-1)

        gate_fwd = curr_R[:, :, 1]   # b2 column = gate facing direction    # (N, 3)
        next_gate_fwd = next_currR[:, :, 1]   # b2 column = gate facing direction    # (N, 3)
        return rel_pos, rel_vel, gate_fwd, next_gate_fwd









    # ============================================================
    # Rendering
    # ============================================================
    def enable_recording(self):
        self._record = True
        self.traj_data = torch.empty((self.max_steps, 34), device=self.device)

    def render(self):
        if self.render_mode == "human":
            traj_np = self.traj_data.cpu().numpy()
            RacingRenderer(
                trajectory     = traj_np,
                ref_trajectory = traj_np[:, 17:20],
                gates_position = self.gates_position,
                gates_rpy      = self.gates_rpy,
            ).run()

    