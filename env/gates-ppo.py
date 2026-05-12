"""
racing.py  —  Gate-progress racing environment for BPTT training.

Reward structure (replaces trajectory-tracking loss):
  - Proximity to current gate centroid        (dense, per-step)
  - Successful gate crossing bonus            (sparse)
  - Crash / miss penalty on crossing          (sparse)
  - Body-rate penalty                         (dense, regularisation)

Observation (16-dim, same size as trajectory-tracking so policy arch is unchanged):
  [0:3]   drone position in current gate frame   (relative pos)
  [3:6]   drone velocity in current gate frame   (body-frame vel projection)
  [6:9]   gate facing direction in world frame   (gate b2 column of R_wg)
  [9:13]  quaternion  [w, x, y, z]
  [13:16] body rates  [p, q, r]
"""

import os
import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from omegaconf import DictConfig, OmegaConf
import numpy as np

from utils.nn import MLP
from utils.randomize import randomize_parameters
from utils.replay_multi import MultiDroneRenderer, RacingRenderer
from utils.math import acc_to_quat, quat_to_rotmat
from utils.plotter import plot_rollout
from utils.trajectory import LemniscataTrajectory
from miscellaneous.loader import load_gates_from_yaml

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, CTBR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACT_DIMS = {"ctbr": 4}
CM_COLS  = {"ctbr": 34}   # state(17) + p_ref(3) + v_ref(3) + a_ref(3) + actions(8)

# Gate half-extents [m]  (width W, height H in gate-local x and z)
GATE_W = 1.2 / 2.0
GATE_H = 1.2 / 2.0


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def build_gate_tensors(
    gates_position: "np.ndarray",
    gates_rpy: "np.ndarray",
    device: str,
):
    """Convert numpy gate arrays to torch tensors on device."""


    gate_pos = torch.tensor(gates_position, dtype=torch.float32, device=device)   # (G, 3)
    gate_rpy = torch.tensor(gates_rpy,      dtype=torch.float32, device=device)   # (G, 3)
    gate_R   = _rpy_deg_to_rotmat(gate_rpy)                                       # (G, 3, 3)
    return gate_pos, gate_R


def _rpy_deg_to_rotmat(rpy_deg: Tensor) -> Tensor:
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


def gate_relative_state(
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

    # R^T maps world -> gate frame
    delta   = drone_pos - curr_pos                                      # (N, 3)
    rel_pos = torch.bmm(curr_R.transpose(1, 2), delta.unsqueeze(-1)).squeeze(-1)   # (N, 3)
    rel_vel = torch.bmm(curr_R.transpose(1, 2), drone_vel.unsqueeze(-1)).squeeze(-1)

    gate_fwd = curr_R[:, :, 1]   # b2 column = gate facing direction    # (N, 3)
    return rel_pos, rel_vel, gate_fwd


def check_crossing(rel_pos: Tensor) -> Tensor:
    """
    Returns boolean mask (N,) of envs that just crossed the gate plane
    (gate-local y > 0).  Call *after* step, compare sign change externally
    if you need exact crossing detection — here we use simple threshold.
    """
    return rel_pos[:, 1] > 0.0


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def get_observation(
    states:   Tensor,    # (N, 17)
    rel_pos:  Tensor,    # (N, 3)  position in gate frame
    rel_vel:  Tensor,    # (N, 3)  velocity in gate frame
    gate_fwd: Tensor,    # (N, 3)  gate facing dir in world
) -> Tensor:
    """
    16-element observation:
      [0:3]   relative position to gate (gate frame)
      [3:6]   relative velocity         (gate frame)
      [6:9]   gate facing direction     (world frame)
      [9:13]  quaternion [w, x, y, z]
      [13:16] body rates [p, q, r]
    """
    return torch.cat([
        rel_pos,               # (N, 3)
        rel_vel,               # (N, 3)
        gate_fwd,              # (N, 3)
        states[:, 6:10],       # quaternion  (N, 4)
        states[:, 10:13],      # body rates  (N, 3)
    ], dim=-1)                 # (N, 16)


# ---------------------------------------------------------------------------
# Reward / loss
# ---------------------------------------------------------------------------

def compute_reward(
    states:   Tensor,    # (N, 17)
    rel_pos:  Tensor,    # (N, 3)  in gate frame
    prev_rel_pos_y: Tensor,  # (N,) gate-local y from previous step
    gate_idx: Tensor,    # (N,) current gate index
    num_gates: int,
    cfg,
) -> tuple[Tensor, Tensor]:
    """
    Returns
    -------
    reward       : (N,) per-env scalar reward for this step
    new_gate_idx : (N,) updated gate indices after crossing detection
    """
    N      = states.shape[0]
    device = states.device
    reward = torch.zeros(N, device=device)

    # ── 1. Dense proximity reward (encourage approaching gate centroid) ───
    dist_to_gate = torch.linalg.norm(rel_pos, dim=-1)          # (N,)
    reward -= cfg.w_proximity * dist_to_gate

    # ── 2. Body-rate penalty ─────────────────────────────────────────────
    rate_penalty = (states[:, 10:13] ** 2).sum(dim=-1)         # (N,)
    reward -= cfg.w_rate * rate_penalty

    # ── 3. Gate crossing detection  (sign flip in gate-local y) ──────────
    #   prev_y < 0  and  curr_y >= 0  → just crossed the plane
    crossed = (prev_rel_pos_y < 0.0) & (rel_pos[:, 1] >= 0.0)  # (N,)

    in_x = rel_pos[:, 0].abs() < GATE_W
    in_z = rel_pos[:, 2].abs() < GATE_H
    success = crossed & in_x & in_z
    miss    = crossed & ~(in_x & in_z)

    reward = torch.where(success, reward + cfg.r_success, reward)
    reward = torch.where(miss,    reward - cfg.r_miss,    reward)

    # Advance gate index for successful crossings
    new_gate_idx = torch.where(
        success,
        (gate_idx + 1) % num_gates,
        gate_idx,
    )

    return reward, new_gate_idx


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------

def reset(cfg, quadrotor, controller, gate_pos, gate_R):
    """Full episode reset — place all envs at first gate approach."""
    N, device = cfg.num_envs, cfg.device

    # Start slightly behind gate 0 in gate-local frame, then transform to world
    first_R   = gate_R[0]                                   # (3, 3)
    first_pos = gate_pos[0]                                 # (3,)

    # Spawn behind gate 0 (gate-local y = -1.0), add small xy scatter
    offset_gate = torch.zeros((N, 3), device=device)
    offset_gate[:, 1] = -1.5                                # 1.5 m behind gate
    offset_gate[:, 0] = (torch.rand(N, device=device) - 0.5) * 0.4
    offset_gate[:, 2] = (torch.rand(N, device=device) - 0.5) * 0.4

    spawn_world = (first_R @ offset_gate.T).T + first_pos   # (N, 3)

    states = torch.zeros((N, 17), device=device)
    states[:, 0:3] = spawn_world
    states[:, 6]   = 1.0   # unit quaternion w = 1

    gate_idx = torch.zeros(N, dtype=torch.long, device=device)

    params = randomize_parameters(cfg.dynamics, N, device)
    quadrotor.set_parameters(params)
    controller.update_params(
        alloc_matrix=quadrotor._alloc_matrix,
        mass=quadrotor.m,
        max_TWR=quadrotor.max_TWR,
        J=quadrotor.J,
    )
    return states, gate_idx, params


def reset_terminated(
    states:     Tensor,
    gate_idx:   Tensor,
    terminated: Tensor,
    gate_pos:   Tensor,
    gate_R:     Tensor,
    cfg,
) -> tuple[Tensor, Tensor]:
    """
    Partial reset — returns a NEW states tensor (no inplace writes).

    Using torch.where keeps the operation out-of-place so the autograd
    graph built over the current truncation window is not corrupted.
    States are always detached after reset because the new initial
    conditions carry no gradient information.
    """
    if not terminated.any():
        return states, gate_idx

    N, device = states.shape[0], states.device

    # Build fresh states for ALL envs (cheap, only a few hundred floats)
    # Each env uses its *current* gate_idx so it respawns at the right gate.
    curr_R   = gate_R[gate_idx]    # (N, 3, 3)
    curr_pos = gate_pos[gate_idx]  # (N, 3)

    offset = torch.zeros((N, 3), device=device)
    offset[:, 1] = -1.5
    offset[:, 0] = (torch.rand(N, device=device) - 0.5) * 0.4
    offset[:, 2] = (torch.rand(N, device=device) - 0.5) * 0.4
    spawn = torch.bmm(curr_R, offset.unsqueeze(-1)).squeeze(-1) + curr_pos  # (N, 3)

    reset_states = torch.zeros((N, 17), device=device)
    reset_states[:, 0:3] = spawn
    reset_states[:, 6]   = 1.0   # unit quaternion

    # Out-of-place blend: keep live states for active envs, use reset for terminated
    mask = terminated.unsqueeze(-1).float()          # (N, 1)
    # Detach states before blending — reset envs break the graph anyway,
    # and active envs will be detached at the next truncation boundary.
    new_states = torch.where(
        terminated.unsqueeze(-1).expand_as(states),
        reset_states,
        states.detach(),
    )
    return new_states, gate_idx


# ---------------------------------------------------------------------------
# Controller builder
# ---------------------------------------------------------------------------

def build_controller(cfg, quadrotor):
    alloc = DirectAllocation(quadrotor._alloc_matrix)
    return CTBR(
        allocator=alloc,
        mass=quadrotor.m,
        max_TWR=quadrotor.max_TWR,
        J=quadrotor.J,
        max_rate=cfg.env.w_max,
        kp_rate=cfg.env.kp_rate,
        dt=cfg.dt,
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(cfg: DictConfig):
    start    = time.time()
    device   = cfg.device
    num_envs = cfg.num_envs
    cm       = cfg.cm
    assert cm == "ctbr", f"Racing env currently supports only ctbr, got: {cm}"

    print(f"\n{'='*85}")
    print(f" {'-'*22}  Racing (gate-progress BPTT)  {'-'*22} ")
    print(f"  Envs: {num_envs}  |  Episodes: {cfg.episodes}  |  "
          f"Steps: {cfg.steps}  |  Horizon: {cfg.truncation}")
    print(f"{'='*85}\n")

    out_dir = f"/home/adame/torchAirBender/outputs/policies/RACING/gate-progress/{cm}"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load gates ────────────────────────────────────────────────────────
    gates_position, gates_rpy = load_gates_from_yaml(cfg.env.gates_yaml)
    gate_pos, gate_R = build_gate_tensors(gates_position, gates_rpy, device)
    num_gates = gate_pos.shape[0]

    print(f"  Loaded {num_gates} gates from {cfg.env.gates_yaml}\n")

    traj = LemniscataTrajectory(cfg.num_envs, cfg.device, scale=3, speed=4)
    # ── Model & optimiser ────────────────────────────────────────────────
    quadrotor  = QuadrotorDynamics(cfg)
    controller = build_controller(cfg, quadrotor)

    policy = MLP(
        layer_sizes       = list(cfg.env.policy) + [ACT_DIMS[cm]],
        activation        = nn.ReLU,
        output_activation = nn.Sigmoid(),
        output_bias_init  = 0.0,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)

    # For rendering the final episode
    traj_data = torch.empty((cfg.steps, CM_COLS[cm]), device=device)
    best_reward = -float("inf")

    for ep in range(cfg.episodes):
        states, gate_idx, last_params = reset(cfg, quadrotor, controller, gate_pos, gate_R)

        ep_total_reward = torch.zeros(num_envs, device=device)
        window_loss     = torch.zeros(1, device=device)
        window_start    = 0
        ep_loss         = 0.0
        num_updates     = 0

        # Initial gate-relative state (needed for crossing detection).
        # rel_pos / rel_vel / gate_fwd are used in the observation but must
        # NOT carry gradients themselves — they are re-derived each step from
        # `states` which *does* carry gradients.  Keeping them detached avoids
        # the AsStridedBackward version-mismatch error.
        with torch.no_grad():
            rel_pos, rel_vel, gate_fwd = gate_relative_state(
                states[:, 0:3], states[:, 3:6], gate_idx, gate_pos, gate_R
            )
        prev_rel_y = rel_pos[:, 1].clone()   # always detached (no_grad scope)

        for t in range(cfg.steps):
            if t % cfg.truncation == 0:
                states       = states.detach()
                window_start = t
                window_loss  = torch.zeros(1, device=device)
                # Recompute gate state from freshly detached states
                with torch.no_grad():
                    rel_pos, rel_vel, gate_fwd = gate_relative_state(
                        states[:, 0:3], states[:, 3:6], gate_idx, gate_pos, gate_R
                    )
                prev_rel_y = rel_pos[:, 1].clone()

            # ── Observation & forward pass ────────────────────────────────
            # Build obs from detached gate quantities + live states (gradients
            # flow through states[:, 6:10] and states[:, 10:13] only).
            obs     = get_observation(states, rel_pos.detach(), rel_vel.detach(), gate_fwd.detach())
            raw     = policy(obs)
            actions = controller(states, raw)
            states  = quadrotor.step(states, actions[:, 0:4])

            pos_ref, vel_ref, acc_ref, _  = traj.get_reference(t)

            # ── Gate-relative state (post-step) ───────────────────────────
            # Detached: used for reward shaping and next-step observation.
            # The gradient signal enters through `states` in the loss, not
            # through the gate-frame coordinates.
            with torch.no_grad():
                rel_pos, rel_vel, gate_fwd = gate_relative_state(
                    states[:, 0:3], states[:, 3:6], gate_idx, gate_pos, gate_R
                )

            # ── Reward ────────────────────────────────────────────────────
            reward, gate_idx = compute_reward(
                states, rel_pos, prev_rel_y, gate_idx, num_gates, cfg.env
            )
            prev_rel_y = rel_pos[:, 1].clone()   # detached (no_grad scope)

            ep_total_reward += reward.detach()

            # ── BPTT loss  (-reward, minimise) ────────────────────────────
            window_loss -= reward.mean()

            # ── Termination ───────────────────────────────────────────────
            dist_to_gate = torch.linalg.norm(rel_pos, dim=-1)
            too_far      = dist_to_gate > cfg.env.max_dist_to_gate
            terminated   = too_far

            # ── Render data (last episode only) ───────────────────────────
            if ep == cfg.episodes - 1:
                dummy_ref = states[0, 0:3]   # placeholder — no trajectory
                traj_data[t] = torch.cat([
                    states[0].detach(),           # 0:17  state
                    pos_ref[0].detach(),            # 17:20 p_ref (self-pos placeholder)
                    states[0, 3:6].detach(),       # 20:23 v_ref (self-vel placeholder)
                    torch.zeros(3, device=device), # 23:26 a_ref
                    actions[0].detach(),           # 26:34 motor thrusts + wrench
                ], dim=0)

            # ── BPTT update ───────────────────────────────────────────────
            if (t + 1) % cfg.truncation == 0 or (t + 1) == cfg.steps:
                window_len = (t + 1) - window_start
                loss = window_loss / window_len
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss     += loss.item()
                num_updates += 1

            states, gate_idx = reset_terminated(
                states, gate_idx, terminated, gate_pos, gate_R, cfg
            )

        avg_loss   = ep_loss / max(num_updates, 1)
        mean_rew   = ep_total_reward.mean().item()
        gates_seen = gate_idx.float().mean().item()

        print(f"  Episode {ep+1:>4}/{cfg.episodes}  |  "
              f"Loss: {avg_loss:+.4f}  |  "
              f"MeanReward: {mean_rew:+.1f}  |  "
              f"AvgGate: {gates_seen:.1f}/{num_gates}")

        # ── Checkpointing ─────────────────────────────────────────────────
        if mean_rew > best_reward:
            best_reward = mean_rew
            torch.save(policy.state_dict(), os.path.join(out_dir, "policy_best.pt"))
            print(f"  >> New best reward: {best_reward:+.1f} — saved.")

    torch.save(policy.state_dict(), os.path.join(out_dir, "policy_final.pt"))
    torch.jit.script(policy.cpu()).save(os.path.join(out_dir, "policy_final_scripted.pt"))
    print(f"\nTotal training time: {time.time() - start:.1f}s")

    # ── Render final episode ─────────────────────────────────────────────
    traj_np = traj_data.cpu().numpy()
    # ref_traj = pos_ref.cpu().numpy()
    RacingRenderer(
        trajectory     = traj_np,
        ref_trajectory = traj_np[:, 17:20],   # use actual drone path as "ref"
        gates_position = gates_position,
        gates_rpy      = gates_rpy,
    ).run()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test(cfg: DictConfig):
    device = cfg.device
    cm     = cfg.cm

    policy_path = cfg.env.get("policy_path",
        f"/home/adame/torchAirBender/outputs/policies/RACING/gate-progress/{cm}/policy_best.pt")

    # Force single env for clean test rollout
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["num_envs"] = 1
    cfg = OmegaConf.create(cfg_dict)

    gates_position, gates_rpy = load_gates_from_yaml(cfg.env.gates_yaml)
    gate_pos, gate_R = build_gate_tensors(gates_position, gates_rpy, device)
    num_gates = gate_pos.shape[0]

    quadrotor  = QuadrotorDynamics(cfg)
    controller = build_controller(cfg, quadrotor)

    policy = MLP(
        layer_sizes       = list(cfg.env.policy) + [ACT_DIMS[cm]],
        activation        = nn.ReLU,
        output_activation = nn.Sigmoid(),
        output_bias_init  = 0.0,
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    print(f"  Loaded policy: {policy_path}")

    states, gate_idx, params = reset(cfg, quadrotor, controller, gate_pos, gate_R)

    traj_data = torch.empty((cfg.steps, CM_COLS[cm]), device=device)
    total_reward    = 0.0
    gates_completed = 0

    rel_pos, rel_vel, gate_fwd = gate_relative_state(
        states[:, 0:3], states[:, 3:6], gate_idx, gate_pos, gate_R
    )
    prev_rel_y = rel_pos[:, 1].clone()

    print(f"\n{'='*60}")
    print(f"  Test rollout — {cfg.steps} steps")
    print(f"{'='*60}")

    with torch.no_grad():
        for t in range(cfg.steps):
            obs     = get_observation(states, rel_pos, rel_vel, gate_fwd)
            raw     = policy(obs)
            actions = controller(states, raw)
            states  = quadrotor.step(states, actions[:, 0:4])

            rel_pos, rel_vel, gate_fwd = gate_relative_state(
                states[:, 0:3], states[:, 3:6], gate_idx, gate_pos, gate_R
            )

            reward, gate_idx = compute_reward(
                states, rel_pos, prev_rel_y, gate_idx, num_gates, cfg.env
            )
            prev_rel_y = rel_pos[:, 1].clone()

            total_reward    += reward[0].item()
            gates_completed  = gate_idx[0].item()

            if (t + 1) % 200 == 0:
                dist = torch.linalg.norm(rel_pos[0]).item()
                print(f"  t={t+1:>4}  gate={gate_idx[0].item()}  "
                      f"dist_to_gate={dist:.2f}m  reward={total_reward:+.1f}")

            traj_data[t] = torch.cat([
                states[0].detach(),
                states[0, 0:3].detach(),
                states[0, 3:6].detach(),
                torch.zeros(3, device=device),
                actions[0].detach(),
            ], dim=0)

            dist_to_gate = torch.linalg.norm(rel_pos, dim=-1)
            too_far      = dist_to_gate > cfg.env.max_dist_to_gate
            states, gate_idx = reset_terminated(
                states, gate_idx, too_far, gate_pos, gate_R, cfg
            )

    print(f"\n  Total reward:    {total_reward:+.2f}")
    print(f"  Gates completed: {gates_completed}")

    traj_np = traj_data.cpu().numpy()
    RacingRenderer(
        trajectory     = traj_np,
        ref_trajectory = traj_np[:, 0:3],
        gates_position = gates_position,
        gates_rpy      = gates_rpy,
    ).run()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test(cfg: DictConfig):
    device = cfg.device
    cm     = cfg.cm

    policy_path = cfg.env.get("policy_path",
        f"/home/adame/torchAirBender/outputs/policies/RACING/gate-progress/{cm}/policy_best.pt")

    # Force single env for clean test rollout
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["num_envs"] = 1
    cfg = OmegaConf.create(cfg_dict)

    gates_position, gates_rpy = load_gates_from_yaml(cfg.env.gates_yaml)
    gate_pos, gate_R = build_gate_tensors(gates_position, gates_rpy, device)
    num_gates = gate_pos.shape[0]

    quadrotor  = QuadrotorDynamics(cfg)
    controller = build_controller(cfg, quadrotor)

    policy = MLP(
        layer_sizes       = list(cfg.env.policy) + [ACT_DIMS[cm]],
        activation        = nn.ReLU,
        output_activation = nn.Sigmoid(),
        output_bias_init  = 0.0,
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    print(f"  Loaded policy: {policy_path}")

    states, gate_idx, params = reset(cfg, quadrotor, controller, gate_pos, gate_R)

    traj_data = torch.empty((cfg.steps, CM_COLS[cm]), device=device)
    total_reward    = 0.0
    gates_completed = 0

    rel_pos, rel_vel, gate_fwd = gate_relative_state(
        states[:, 0:3], states[:, 3:6], gate_idx, gate_pos, gate_R
    )
    prev_rel_y = rel_pos[:, 1].clone()

    print(f"\n{'='*60}")
    print(f"  Test rollout — {cfg.steps} steps")
    print(f"{'='*60}")

    with torch.no_grad():
        for t in range(cfg.steps):
            obs     = get_observation(states, rel_pos, rel_vel, gate_fwd)
            raw     = policy(obs)
            actions = controller(states, raw)
            states  = quadrotor.step(states, actions[:, 0:4])

            rel_pos, rel_vel, gate_fwd = gate_relative_state(
                states[:, 0:3], states[:, 3:6], gate_idx, gate_pos, gate_R
            )

            reward, gate_idx = compute_reward(
                states, rel_pos, prev_rel_y, gate_idx, num_gates, cfg.env
            )
            prev_rel_y = rel_pos[:, 1].clone()

            total_reward    += reward[0].item()
            gates_completed  = gate_idx[0].item()

            if (t + 1) % 200 == 0:
                dist = torch.linalg.norm(rel_pos[0]).item()
                print(f"  t={t+1:>4}  gate={gate_idx[0].item()}  "
                      f"dist_to_gate={dist:.2f}m  reward={total_reward:+.1f}")

            traj_data[t] = torch.cat([
                states[0].detach(),
                states[0, 0:3].detach(),
                states[0, 3:6].detach(),
                torch.zeros(3, device=device),
                actions[0].detach(),
            ], dim=0)

            dist_to_gate = torch.linalg.norm(rel_pos, dim=-1)
            too_far      = dist_to_gate > cfg.env.max_dist_to_gate
            states, gate_idx = reset_terminated(
                states, gate_idx, too_far, gate_pos, gate_R, cfg
            )

    print(f"\n  Total reward:    {total_reward:+.2f}")
    print(f"  Gates completed: {gates_completed}")

    traj_np = traj_data.cpu().numpy()
    RacingRenderer(
        trajectory     = traj_np,
        ref_trajectory = traj_np[:, 0:3],
        gates_position = gates_position,
        gates_rpy      = gates_rpy,
    ).run()
# ==================================================================
#                    TESTING THE POLICIES
# ==================================================================

def test(cfg: DictConfig):
    policies = [
        {"type": "bptt",
         "cm": "ctbr", 
         "path": "/home/adame/torchAirBender/outputs/policies/GATES/fig8/ctbr/policy_best_0.18.pt",  
         "color": (0.2, 0.6, 1.0)},

        # {"type": "ppo",
        #  "cm": "ctbr",
        #  "path": "/home/adame/torchAirBender/outputs/policies/PPO/tt_ctbr_results.zip",
        # "color": (1.0, 0.55, 0.0)},
    ]
    for p in policies:
        p["label"] = p["type"] + "_" + p["cm"] + "_" + Path(p["path"]).stem.split("_", 1)[-1]

    randomize = True
    seed      = None
    save      = True
    csv_path  = "/home/adame/torchAirBender/outputs/policies/GATES/fig8/exported_data/traj-bptt.csv"

    if seed is not None:
        torch.manual_seed(seed)

    device    = cfg.device
    quadrotor = QuadrotorDynamics(cfg)
    traj      = TrajectoryManager.from_harmonics(cfg.env.traj, cfg.num_envs, device)
    traj = LemniscataTrajectory(cfg.num_envs, cfg.device, scale=3, speed=4)

    ref_traj = None
    drones   = []
    params   = None

    for spec in policies:
        print(f"\n  Rolling out: {spec['label']}  ({spec['path']})")

        controller = build_controller(spec["cm"], quadrotor, cfg)

        # ── Load policy ──────────────────────────────────────────────────
        if spec["type"] == "ppo":
            from stable_baselines3 import PPO as SB3PPO
            sb3_model = SB3PPO.load(spec["path"], device=device)
            def get_action(obs):
                # obs: (N, 16) tensor → numpy → predict → tensor
                action_np, _ = sb3_model.predict(
                    obs.cpu().numpy(), deterministic=True
                )
                return torch.tensor(action_np, device=device, dtype=torch.float32)

        else:  # bptt
            policy = MLP(
                layer_sizes       = list(cfg.env.policy) + [ACT_DIMS[spec["cm"]]],
                activation        = nn.ReLU,
                output_activation = nn.Sigmoid(),
                output_bias_init  = 0.0,
            ).to(device)
            policy.load_state_dict(torch.load(spec["path"], map_location=device))
            policy.eval()
            def get_action(obs):
                return policy(obs)

        # ── Randomize dynamics ───────────────────────────────────────────
        if randomize:
            params = randomize_parameters(cfg.dynamics, cfg.num_envs, device)
            quadrotor.set_parameters(params)
            controller.update_params(
                alloc_matrix = quadrotor._alloc_matrix,
                J            = quadrotor.J,
            )

        # ── Reset state at trajectory start ─────────────────────────────
        pos0, vel0, acc0, _ = traj.get_reference(0)
        states = torch.zeros((cfg.num_envs, 17), device=device)
        states[:, 0:3]  = pos0.detach()
        states[:, 3:6]  = vel0.detach()
        states[:, 6:10] = acc_to_quat(acc0.detach())

        traj_data = torch.empty((cfg.steps, CM_COLS[spec["cm"]]), device=device)

        # ── Rollout ──────────────────────────────────────────────────────
        with torch.no_grad():
            for t in range(cfg.steps):
                pos_ref, vel_ref, acc_ref, _ = traj.get_reference(t)
                obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
                raw     = get_action(obs)
                actions = controller(states, raw)
                states  = quadrotor.step(states, actions[:, 0:4])

                dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
                too_far = dist > cfg.env.max_dist_to_target

                traj_data[t] = torch.cat([
                    states[0].detach(),     # 0:17  — full state
                    pos_ref[0].detach(),    # 17:20 — p_ref
                    vel_ref[0].detach(),    # 20:23 — v_ref
                    acc_ref[0].detach(),    # 23:26 — a_ref
                    actions[0].detach(),    # 26:N  — motor thrusts + wrench
                ], dim=0)

                states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)

        traj_np = traj_data.cpu().numpy()

        # ── Optionally save reference CSV ────────────────────────────────
        # if ref_traj is None and save:
        #     os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        #     ref_data = traj_np[:, 17:26]  # [px,py,pz,vx,vy,vz,ax,ay,az]
        #     with open(csv_path, "w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["time", "px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az"])
        #         for i, row in enumerate(ref_data):
        #             writer.writerow([i * cfg.dt, *row.tolist()])
        #     print(f"  Saved reference CSV: {csv_path}")

        if ref_traj is None and save:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            ref_data = traj_np[:, 0:23]  # [px,py,pz,vx,vy,vz,ax,ay,az]
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz", "M1", "M2", "M3", "M4", "px_ref", "py_ref", "pz_ref", "vx_ref", "vy_ref", "vz_ref"])
                for i, row in enumerate(ref_data):
                    writer.writerow([i * cfg.dt, *row.tolist()])
            print(f"  Saved reference CSV: {csv_path}")

        ref_traj = ref_traj if ref_traj is not None else traj_np[:, 17:20]

        drones.append({
            "traj":  traj_np,
            "color": spec.get("color", (0.2, 0.6, 1.0)),
            "label": spec["label"],
        })

        # ── Plot dashboard ───────────────────────────────────────────────
        plot_rollout(
            traj_np    = traj_np,
            dt         = cfg.dt,
            label      = spec["label"],
        )

    # ── Render all drones together ───────────────────────────────────────
    MultiDroneRenderer(drones=drones, ref_trajectory=ref_traj).run()
