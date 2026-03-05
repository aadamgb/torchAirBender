import os
import time
import torch
from torch import Tensor
from torch import nn
from omegaconf import DictConfig, OmegaConf

from utils.nn import MLP 
from utils.randomize import randomize_parameters
from utils.replay import PositionControlRenderer

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.srt_controller import SRTController


def _sample_in_cube(n: int, boundary: float, device) -> Tensor:
    """Sample n positions uniformly in the boundary cube (x,y: [-h,h], z: [0,b])."""
    half = boundary / 2.0
    pos = torch.rand((n, 3), device=device)
    pos[:, 0] = pos[:, 0] * boundary - half
    pos[:, 1] = pos[:, 1] * boundary - half
    pos[:, 2] = pos[:, 2] * boundary
    return pos

def reset(
        cfg: DictConfig,
        ):
    
    num_envs = cfg.num_envs
    device = cfg.device

    # ------------ Randomize initial position -----------
    states = torch.zeros((num_envs, 13), device=device)
    states[:, :3]   = _sample_in_cube(num_envs, cfg.env.target_boundary, device)
    states[:, 6] = 1.0  # seting quaternion w to 1 (FIXED)

    # ----------- Randomize target positon --------------
    targets = _sample_in_cube(num_envs, cfg.env.target_boundary, device)

    # ----------- Randomize drone params --------------
    params = randomize_parameters(
        cfg=cfg.dynamics,
        num_envs=num_envs,
        device=device,
    )

    return states, targets, params

def reset_terminated(
        states: Tensor,
        targets: Tensor,
        terminated: Tensor,  # (N,) bool
        cfg: DictConfig,
) -> tuple[Tensor, Tensor]:
    """Reset only the environments that terminated."""
    if not terminated.any():
        return states, targets

    idx = terminated.nonzero(as_tuple=True)[0]
    n   = idx.numel()
    device = states.device

    new_states        = torch.zeros((n, 13), device=device)
    new_states[:, :3] = _sample_in_cube(n, cfg.env.target_boundary, device)
    new_states[:, 6]  = 1.0

    states[idx]  = new_states
    targets[idx] = _sample_in_cube(n, cfg.env.target_boundary, device)

    return states, targets

def get_observation(
        states: Tensor, 
        targets: Tensor
        )-> Tensor:
    
    p       = states[:, 0:3]
    rest    = states[:, 3:13]   # v, q, w
    p_error = targets - p

    return torch.cat([p_error, rest], dim=-1)

def compute_loss(
    states: Tensor,
    targets: Tensor,    
    weights: DictConfig,
    mask: Tensor | None = None,  # (N,) bool — True = include in loss
) -> Tensor:
    """
    Position tracking + distance-weighted velocity damping + body rate penalty.
    Terminated envs are excluded via mask.
    """

    if mask is not None and not mask.any():
        # return torch.zeros(1, device=states.device, requires_grad=True)
        return states.sum() * 0.0

    s = states[mask] if mask is not None else states
    t = targets[mask] if mask is not None else targets

    # Position term
    p_error = t - s[:, 0:3]                         # (N, 3)
    distances = torch.linalg.norm(p_error, dim=-1)  # (N,)
    pos_loss = distances.mean()

    # Velocity term
    vel_loss = (s[:, 3:6] ** 2).sum(dim=-1).mean()

    # Body rate penalty
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    # Total loss
    return weights.pos * pos_loss + weights.vel * vel_loss + weights.body_rates * rate_loss



def train(cfg: DictConfig):
    start = time.time()
    # Path for saving the best and last policies
    output_dir = "/home/adame/torchAirBender/outputs/policies"
    os.makedirs(output_dir, exist_ok=True)

    # Simulation params
    dt = cfg.dt
    device = cfg.device
    num_envs = cfg.num_envs
    episodes = cfg.episodes
    steps = cfg.steps
    truncation = cfg.truncation
    
    print(f"\n{'='*75}")
    print(f"  Position Control Training")
    print(f"  Envs: {num_envs}  |  "
          f"  Episodes     : {episodes}  |  "
          f"  Steps: {steps}  |  "
        f"  Horizon: {truncation}"
        )
    print(f"{'='*75}\n")

    # Initialize tensors
    targets = torch.rand((num_envs, 3), device=device)
    states = torch.zeros((num_envs, 13), device=device)

    # Pre-allocate truncation loss buffer (replace the list approach)
    truncation_losses = torch.empty(truncation, device=device)  # for logging only
    
    # Store the trajectory of env 0 for visualization (states + actions)
    traj_env0 = torch.empty((steps, 17), device=device)  # 13 state + 4 actions

    quadrotor = QuadrotorDynamics(cfg)
    policy = MLP(
        layer_sizes=cfg.env.nn,
        activation=nn.ReLU,
        output_activation=nn.Sigmoid(),
        output_bias_init=0.0,
    ).to(device)
    controller = SRTController(cfg)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)


    #-------------------------------------------
    # Main Simulation Loop
    #------------------------------------------
    best_loss = float("inf")
    for ep in range(episodes):
        states, targets, randomized_params = reset(cfg)
        quadrotor.set_parameters(randomized_params)

        ep_loss = 0.0
        num_updates = 0

        for t in range(steps):

            if t % truncation == 0:
                states = states.detach()
                window_start = t
                window_loss_sum = torch.zeros(1, device=device)  # differentiable accumulator

            obs = get_observation(states, targets)
            raw_action     = policy(obs)                  
            actions = controller(raw_action)              
            states = quadrotor.step(state=states, action=actions)

            # --- termination: out of bounds ---
            pos = states[:, 0:3]                          # (N, 3)
            half = cfg.env.target_boundary / 2.0

            out_of_bounds = (
                (pos[:, 0].abs() > half) |                # x
                (pos[:, 1].abs() > half) |                # y
                (pos[:, 2] < 0) |                         # z below ground
                (pos[:, 2] > cfg.env.target_boundary)     # z above ceiling
            )                                             # (N,) bool

            terminated = out_of_bounds
            alive      = ~terminated                                             # (N,) bool

            step_loss = compute_loss(states, targets, weights=cfg.env.loss_weights, mask=alive)

            # Detached copy for logging/inspection
            truncation_losses[t % truncation] = step_loss.detach()

            # Differentiable accumulator for backward
            window_loss_sum = window_loss_sum + step_loss

            if ep == episodes - 1:
                traj_env0[t] = torch.cat([
                    states[0].detach(),
                    actions[0].detach()
                ], dim=0)

            if (t + 1) % truncation == 0 or (t + 1) == steps:
                window_len = (t + 1) - window_start
                loss = window_loss_sum / window_len

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                ep_loss += truncation_losses[:window_len].mean().item()
                num_updates += 1

            # --- soft reset terminated envs (detached, after backward) ---
            states, targets = reset_terminated(states, targets, terminated, cfg)

        avg_ep_loss = ep_loss / num_updates
         
        print(f"  Episode {ep+1:>4}/{episodes}  |  Avg Loss: {avg_ep_loss:.4f}")  #if (ep + 1) % 10 == 0 else None

        # Save best policy
        if avg_ep_loss < best_loss:
            best_loss = avg_ep_loss
            torch.save(policy.state_dict(), os.path.join(output_dir, "position_control_best.pt"))


    print(f"Total training time: {time.time() - start:.2f}s")
    # Save the final policy
    torch.save(policy.state_dict(), os.path.join(output_dir, f"position_control_final.pt"))


    # Replay the trajectory
    renderer = PositionControlRenderer(
        target_pos=targets[0].detach().cpu().numpy(),
        boundary=cfg.env.target_boundary,
        trajectory=traj_env0.detach().cpu().numpy(),  # (T, 13)
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()



#==============================================================
# Load policy and test one env one episode
#==============================================================

def test(cfg: DictConfig):
    policy_path = "/home/adame/torchAirBender/outputs/policies/position_control_best.pt"
    policy_path = "/home/adame/torchAirBender/outputs/policies/BEST.pt"
    dt       = cfg.dt
    device   = cfg.device
    steps    = 2000

    # ── single env ──────────────────────────────────────────────────────────
    test_cfg = cfg.copy() if hasattr(cfg, "copy") else cfg
    # Override num_envs to 1 for testing

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["num_envs"] = 1
    cfg = OmegaConf.create(cfg_dict)

    # ── load policy ─────────────────────────────────────────────────────────
    policy = MLP(
        layer_sizes=cfg.env.nn,
        activation=nn.ReLU,
        output_activation=nn.Sigmoid(),
        output_bias_init=0.0,
    ).to(device)

    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    print(f"Loaded policy from: {policy_path}")

    # ── select controller ────────────────────────────────────────────────────
    controller = SRTController(cfg)

    # ── init ─────────────────────────────────────────────────────────────────
    states, targets, randomized_params = reset(cfg)
    quadrotor = QuadrotorDynamics(cfg)
    quadrotor.set_parameters(randomized_params)

    traj = torch.empty((steps, 20), device=device)  # 13 state + 4 actions + 3 target

    print(f"\n{'='*60}")
    print(f"  Target: {targets[0].cpu().numpy()}")
    print(f"  Start:  {states[0, :3].cpu().numpy()}")
    print(f"{'='*60}\n")

    # ── rollout ──────────────────────────────────────────────────────────────
    total_loss = 0.0
    with torch.no_grad():
        for t in range(steps):

            # Sampling a new target every 250 steps
            if (t +1) % 250 == 0:
                _, targets, _= reset(cfg)

            obs     = get_observation(states, targets)
            raw     = policy(obs)                  # (N, 4) in (0, 1)
            actions = controller(raw)              # (N, 4) in (0, max_thrust)

            states  = quadrotor.step(state=states, action=actions)

            distances  = torch.linalg.norm(targets - states[:, 0:3], dim=-1)
            pos = states[:, 0:3]                          # (N, 3)
            half = cfg.env.target_boundary / 1.5

            out_of_bounds = (
                (pos[:, 0].abs() > half) |                # x
                (pos[:, 1].abs() > half) |                # y
                (pos[:, 2] < 0) |                         # z below ground
                (pos[:, 2] > cfg.env.target_boundary)     # z above ceiling
            )                                             # (N,) bool

            terminated = out_of_bounds

            step_loss  = compute_loss(states, targets, cfg.env.loss_weights)
            total_loss += step_loss.item()

            traj[t] = torch.cat([states[0], actions[0], targets[0]], dim=0)

            if terminated[0]:
                print(f"  !! Terminated at step {t+1} — distance: {distances[0]:.3f}")
                states, targets = reset_terminated(states, targets, terminated, cfg)

    avg_loss = total_loss / steps
    final_dist = torch.linalg.norm(targets[0] - states[0, :3]).item()
    print(f"\n  Avg Loss:   {avg_loss:.4f}")
    print(f"  Final Dist: {final_dist:.4f}")

    # ── replay ───────────────────────────────────────────────────────────────
    renderer = PositionControlRenderer(
        target_pos=traj[:, 17:20].detach().cpu().numpy(),
        boundary=cfg.env.target_boundary,
        trajectory=traj.detach().cpu().numpy(),
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()