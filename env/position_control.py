import os
import time
import torch
from torch import Tensor
from torch import nn
from omegaconf import DictConfig, OmegaConf

from utils.nn import MLP
from utils.randomize import randomize_parameters
from utils.plotter import plot_rollout
from utils.replay_multi import PositionControlRenderer

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, CTBR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_in_cube(n: int, boundary: float, device) -> Tensor:
    """Sample n positions uniformly in [-b/2, b/2] x [-b/2, b/2] x [0, b]."""
    half = boundary / 2.0
    pos = torch.rand((n, 3), device=device)
    pos[:, 0] = pos[:, 0] * boundary - half
    pos[:, 1] = pos[:, 1] * boundary - half
    pos[:, 2] = pos[:, 2] * boundary
    return pos


def reset(cfg: DictConfig, quadrotor: QuadrotorDynamics, controller: CTBR):
    n, device = cfg.num_envs, cfg.device

    states = torch.zeros((n, 13), device=device)
    states[:, :3]  = _sample_in_cube(n, cfg.env.boundary, device)
    states[:, 6]   = 1.0  # unit quaternion w=1

    targets = _sample_in_cube(n, cfg.env.boundary, device)

    params = randomize_parameters(cfg.dynamics, n, device)
    quadrotor.set_parameters(params)
    controller.update_params(
        alloc_matrix = quadrotor._alloc_matrix,
        mass         = quadrotor.m,
        max_TWR      = quadrotor.max_TWR,
        J            = quadrotor.J,
    )
    return states, targets, params


def reset_terminated(
        states: Tensor,
        targets: Tensor,
        terminated: Tensor,
        cfg: DictConfig,
) -> tuple[Tensor, Tensor]:
    if not terminated.any():
        return states, targets
    idx = terminated.nonzero(as_tuple=True)[0]
    n   = idx.numel()
    device = states.device

    new_states        = torch.zeros((n, 13), device=device)
    new_states[:, :3] = _sample_in_cube(n, cfg.env.boundary, device)
    new_states[:, 6]  = 1.0

    states[idx]  = new_states
    targets[idx] = _sample_in_cube(n, cfg.env.boundary, device)
    return states, targets


def get_observation(states: Tensor, targets: Tensor) -> Tensor:
    """
    16-element observation — same layout as trajectory tracking:
      [0:3]   pos_ref  - pos    (position error)
      [3:6]   vel_ref  - vel    (velocity error, vel_ref=0 for point control)
      [6:9]   acc_ref          (reference acc,   acc_ref=0 for point control)
      [9:13]  quaternion [w, x, y, z]
      [13:16] body rates [p, q, r]
    """
    zeros3 = torch.zeros_like(states[:, 0:3])
    return torch.cat([
        targets - states[:, 0:3],   # position error   (N, 3)
        zeros3 - states[:, 3:6],    # velocity error   (N, 3)  vel_ref = 0
        zeros3,                     # reference acc    (N, 3)  acc_ref = 0
        states[:, 6:10],            # quaternion       (N, 4)
        states[:, 10:13],           # body rates       (N, 3)
    ], dim=-1)                      # (N, 16)


def compute_loss(
        states: Tensor,
        targets: Tensor,
        weights: DictConfig,
        mask: Tensor | None = None,
) -> Tensor:
    if mask is not None and not mask.any():
        return states.sum() * 0.0

    s = states[mask]  if mask is not None else states
    t = targets[mask] if mask is not None else targets

    # Position
    pos_loss  = torch.linalg.norm(t - s[:, 0:3], dim=-1).mean()

    # Velocity damping (vel_ref = 0)
    vel_loss  = torch.linalg.norm(s[:, 3:6], dim=-1).mean()

    # Body rate penalty
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    # Attitude alignment: body-z should point along gravity-corrected thrust dir
    from utils.math import quat_to_rotmat
    p_error    = t - s[:, 0:3]
    gravity    = torch.tensor([0., 0., -9.81], device=s.device)
    # desired thrust ~ position error direction (crude) + gravity cancel
    thrust_dir = torch.nn.functional.normalize(p_error - gravity, dim=-1)
    body_z     = quat_to_rotmat(s[:, 6:10])[:, :, 2]
    att_loss   = (1.0 - (body_z * thrust_dir).sum(dim=-1).clamp(-1, 1)).mean()

    return (weights.pos        * pos_loss  +
            weights.vel        * vel_loss  +
            weights.att        * att_loss  +
            weights.body_rates * rate_loss)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(cfg: DictConfig):
    start  = time.time()
    device = cfg.device

    out_dir = f"/home/adame/torchAirBender/outputs/policies/PC-CTBR"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*85}")
    print(f" {'-'*20}  Training Position Control (CTBR)  {'-'*20} ")
    print(f"  Envs: {cfg.num_envs}  |  Episodes: {cfg.episodes}  |  "
          f"Steps: {cfg.steps}  |  Horizon: {cfg.truncation}")
    print(f"{'='*85}\n")

    quadrotor  = QuadrotorDynamics(cfg)
    alloc      = DirectAllocation(quadrotor._alloc_matrix)
    controller = CTBR(
        allocator = alloc,
        mass      = quadrotor.m,
        max_TWR   = quadrotor.max_TWR,
        J         = quadrotor.J,
        max_rate  = cfg.env.w_max,
        kp_rate   = cfg.env.kp_rate,
        dt        = cfg.dt,
    )

    policy = MLP(
        layer_sizes       = list(cfg.env.policy) + [4],
        activation        = nn.ReLU,
        output_activation = nn.Sigmoid(),
        output_bias_init  = 0.0,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)

    best_loss = float("inf")

    for ep in range(cfg.episodes):
        states, targets, last_params = reset(cfg, quadrotor, controller)

        # states = torch.zeros((cfg.num_envs, 13), device=device)
        # states[:, 0] = -1.5
        # states[:, 1] = -2.5
        # states[:, 2] = 1.5
        # states[:, 6]   = 1.0  # unit quaternion w=1
        # targets = torch.tensor([1.0, -2.5, 1.5], device=device).unsqueeze(0).expand(cfg.num_envs, -1)

        ep_loss      = 0.0
        num_updates  = 0
        window_loss  = torch.zeros(1, device=device)
        window_start = 0

        for t in range(cfg.steps):
            if t % cfg.truncation == 0:
                states       = states.detach()
                window_start = t
                window_loss  = torch.zeros(1, device=device)

            obs     = get_observation(states, targets)
            raw     = policy(obs)
            actions = controller(states, raw)
            states  = quadrotor.step(states, actions[:, 0:4])

            # Termination: out of bounds or too far from target
            pos  = states[:, 0:3]
            half = cfg.env.boundary / 2.0
            out_of_bounds = (
                (pos[:, 0].abs() > half) |
                (pos[:, 1].abs() > half) |
                (pos[:, 2] < 0)          |
                (pos[:, 2] > cfg.env.boundary)
            )
            dist     = torch.linalg.norm(targets - pos, dim=-1)
            too_far  = dist > cfg.env.max_dist_to_target
            terminated = out_of_bounds | too_far
            alive      = ~terminated

            window_loss += compute_loss(states, targets, cfg.env.loss_weights, mask=alive)

            if (t + 1) % cfg.truncation == 0 or (t + 1) == cfg.steps:
                window_len = (t + 1) - window_start
                loss = window_loss / window_len
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss     += loss.item()
                num_updates += 1

            states, targets = reset_terminated(states, targets, terminated, cfg)

        avg_loss = ep_loss / max(num_updates, 1)
        print(f"  Episode {ep+1:>4}/{cfg.episodes}  |  Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), os.path.join(out_dir, "policy_best.pt"))
            scripted = torch.jit.script(policy)
            scripted.save(os.path.join(out_dir, "policy_best_scripted.pt"))

    torch.save(policy.state_dict(), os.path.join(out_dir, "policy_final.pt"))
    scripted = torch.jit.script(policy)
    scripted.save(os.path.join(out_dir, "policy_final_scripted.pt"))
    print(f"\nTotal training time: {time.time() - start:.1f}s")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test(cfg: DictConfig):
    policy_path = "/home/adame/torchAirBender/outputs/policies/PC-CTBR/policy_best.pt"
    device      = cfg.device

    # single env
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["num_envs"] = 1
    cfg = OmegaConf.create(cfg_dict)

    quadrotor  = QuadrotorDynamics(cfg)
    alloc      = DirectAllocation(quadrotor._alloc_matrix)
    controller = CTBR(
        allocator = alloc,
        mass      = quadrotor.m,
        max_TWR   = quadrotor.max_TWR,
        J         = quadrotor.J,
        max_rate  = cfg.env.w_max,
        kp_rate   = cfg.env.kp_rate,
        dt        = cfg.dt,
    )

    policy = MLP(
        layer_sizes       = list(cfg.env.policy) + [4],
        activation        = nn.ReLU,
        output_activation = nn.Sigmoid(),
        output_bias_init  = 0.0,
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    print(f"Loaded policy from: {policy_path}")

    states, targets, params = reset(cfg, quadrotor, controller)
    traj_data = torch.empty((cfg.steps, 13 + 9 + 8), device=device)  # state + target + actions
    
    # states = torch.zeros((1, 13), device=device)
    # states[:, 0] = -1.5
    # states[:, 1] = -2.5
    # states[:, 2] = 1.5
    # states[:, 6]   = 1.0  # unit quaternion w=1
    # targets = torch.tensor([[1.0, -2.5, 1.5]], device=device)
    print(f"\n{'='*60}")
    print(f"  Target: {targets[0].cpu().numpy()}")
    print(f"  Start:  {states[0, :3].cpu().numpy()}")
    print(f"{'='*60}\n")


    total_loss = 0.0
    with torch.no_grad():
        for t in range(cfg.steps):
            # re-sample target every 250 steps
            if (t + 1) % 1000 == 0:
                targets = _sample_in_cube(cfg.num_envs, cfg.env.boundary, device)
                print(f"  New target at step {t+1}: {targets[0].cpu().numpy()}")

            obs     = get_observation(states, targets)
            raw     = policy(obs)
            actions = controller(states, raw)
            states  = quadrotor.step(states, actions[:, 0:4])

            dist       = torch.linalg.norm(targets - states[:, 0:3], dim=-1)
            too_far    = dist > cfg.env.max_dist_to_target
            pos        = states[:, 0:3]
            half       = cfg.env.boundary / 2.0
            out_of_bounds = (
                (pos[:, 0].abs() > half) |
                (pos[:, 1].abs() > half) |
                (pos[:, 2] < 0)          |
                (pos[:, 2] > cfg.env.boundary)
            )
            terminated = out_of_bounds | too_far

            total_loss += compute_loss(states, targets, cfg.env.loss_weights).item()

            traj_data[t] = torch.cat([
                states[0].detach(),   # 0:13
                targets[0].detach(),  # 13:16
                torch.zeros(6, device=device),  # 16:22
                actions[0].detach(),  # 22:26
            ], dim=0)

            if terminated[0]:
                print(f"  !! Terminated at step {t+1} — dist: {dist[0]:.3f} m")
                states, targets = reset_terminated(states, targets, terminated, cfg)

    avg_loss   = total_loss / cfg.steps
    final_dist = torch.linalg.norm(targets[0] - states[0, :3]).item()
    print(f"\n  Avg Loss:   {avg_loss:.4f}")
    print(f"  Final Dist: {final_dist:.4f} m")

    plot_rollout(
        traj_np    = traj_data.cpu().numpy(),
        dt         = cfg.dt,
        label      = "PC-CTBR",
        arm_length = float(params.arm_length[0].cpu()),
        arm_angle  = float(params.arm_angle[0].cpu()),
        mass       = float(params.mass[0].cpu()),
    )

    traj_np = traj_data.cpu().numpy()

    renderer = PositionControlRenderer(
        trajectory  = traj_np,
        target_pos  = traj_np[:, 13:16],   # (T, 3) — target changes every 250 steps
        boundary    = cfg.env.boundary,
        arm_length  = float(params.arm_length[0].cpu()),
        arm_angle   = float(params.arm_angle[0].cpu()),
        mass        = float(params.mass[0].cpu()),
        dt          = cfg.dt,
    )
    renderer.run()