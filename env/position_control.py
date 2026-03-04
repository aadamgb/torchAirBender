import os
import torch
from torch import Tensor
from torch import nn
from omegaconf import DictConfig

from utils.nn import MLP 
from utils.randomize import randomize_parameters
from utils.replay import PositionControlRenderer

from dynamics.quadrotor_dynamics import QuadrotorDynamics
# from controller.srt_controller import SRTController


def reset(
        cfg: DictConfig,
        ):
    
    num_envs = cfg.num_envs
    device = cfg.device

    # ------------ Randomize initial position -----------
    states = torch.zeros((num_envs, 13), device=device)
    states[:, :3] = torch.rand((num_envs, 3), device=device)
    states[:, 6] = 1.0  # seting quaternion w to 1 (FIXED)

    # ----------- Randomize target positon --------------
    half = cfg.env.target_boundary / 2.0  # half-width of the cube

    targets = torch.rand((num_envs, 3), device=device)   # [0, 1)
    targets[:, 0] = targets[:, 0] * cfg.env.target_boundary - half  # x: [-half, half)
    targets[:, 1] = targets[:, 1] * cfg.env.target_boundary - half  # y: [-half, half)
    targets[:, 2] = targets[:, 2] * cfg.env.target_boundary         # z: [0, boundary)

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

    half = cfg.env.target_boundary / 2.0

    new_states        = torch.zeros((n, 13), device=device)
    new_states[:, :3] = torch.rand((n, 3), device=device)
    new_states[:, 6]  = 1.0

    new_targets       = torch.rand((n, 3), device=device)
    new_targets[:, 0] = new_targets[:, 0] * cfg.env.target_boundary - half
    new_targets[:, 1] = new_targets[:, 1] * cfg.env.target_boundary - half
    new_targets[:, 2] = new_targets[:, 2] * cfg.env.target_boundary

    states[idx]  = new_states
    targets[idx] = new_targets

    # print(f"Env {idx} is terminated!!")

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
    mask: Tensor | None = None,  # (N,) bool — True = include in loss
    lambda_pos: float = 2.0,
    lambda_rate: float = 0.1,
    lambda_vel: float = 0.25,
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

    # --------------------------------
    # Position term
    # --------------------------------
    p_error = t - s[:, 0:3]                         # (N, 3)
    distances = torch.linalg.norm(p_error, dim=-1)  # (N,)
    pos_loss = distances.mean()

    # --------------------------------
    # Velocity term
    # --------------------------------
    vel_loss = (s[:, 3:6] ** 2).sum(dim=-1).mean()

    # --------------------------------
    # Body rate penalty
    # --------------------------------
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    # --------------------------------
    # Total loss
    # --------------------------------
    return lambda_pos * pos_loss + lambda_vel * vel_loss + lambda_rate * rate_loss



def train(cfg: DictConfig):
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
    print(f"  Envs: {cfg.num_envs}  |  "
          f"  Episodes     : {episodes}  |  "
          f"  Steps: {cfg.steps}  |  "
        f"  Horizon: {cfg.truncation}"
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

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)


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
                window_loss_sum = torch.zeros(1, device=device)  # live differentiable accumulator

            obs = get_observation(states, targets)
            actions = policy(obs) * 2.0
            from controller.srt_controller import SRTController

            controller = SRTController(cfg)

            # in the loop:
            raw     = policy(obs)                  # (N, 4) in (0, 1)
            actions = controller(raw)              # (N, 4) in (0, max_thrust)
            states  = quadrotor.step(state=states, action=actions)
            
            # print(actions[0]) if t == steps -1 else None
            states = quadrotor.step(state=states, action=actions)

            # --- termination: envs too far from target ---
            distances  = torch.linalg.norm(targets - states[:, 0:3], dim=-1)     # (N,)
            terminated = distances > cfg.env.target_boundary                     # (N,) bool

            # pos = states[:, 0:3]                          # (N, 3)
            # half = cfg.env.target_boundary / 2.0

            # out_of_bounds = (
            #     (pos[:, 0].abs() > half) |                # x
            #     (pos[:, 1].abs() > half) |                # y
            #     (pos[:, 2] < 0) |                         # z below ground
            #     (pos[:, 2] > cfg.env.target_boundary)     # z above ceiling
            # )                                             # (N,) bool

            # terminated = out_of_bounds

            alive      = ~terminated                                             # (N,) bool

            step_loss = compute_loss(states, targets, mask=alive)

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

            # # Adding termination penalty, maybe remove idk tbh
            # if terminated.any():
            #     term_loss = cfg.env.termination_penalty * terminated.float().mean()
            #     step_loss = step_loss + term_loss

        avg_ep_loss = ep_loss / num_updates
         
        print(f"  Episode {ep+1:>4}/{episodes}  |  Avg Loss: {avg_ep_loss:.4f}") if (ep + 1) % 10 == 0 else None

        # Save best policy
        if avg_ep_loss < best_loss:
            best_loss = avg_ep_loss
            torch.save(policy.state_dict(), os.path.join(output_dir, "position_control_best.pt"))
            # print(f"    → New best policy saved (loss = {best_loss:.4f})")


    # Save the final policy
    torch.save(policy.state_dict(), os.path.join(output_dir, f"position_control_final.pt"))


    # Replay the trajectory
    # renderer = PositionControlRenderer(
    #     target_pos=targets[0].detach().cpu().numpy(),
    #     boundary=cfg.env.target_boundary,
    #     trajectory=traj_env0.detach().cpu().numpy(),  # (T, 13)
    #     arm_length=float(randomized_params.arm_length[0].cpu()),
    #     arm_angle=float(randomized_params.arm_angle[0].cpu()),
    #     mass=float(randomized_params.mass[0].cpu()),
    #     dt=dt,
    # )
    # renderer.run()


    traj = traj_env0.detach().cpu().numpy()
    print(traj[:, 13:17].max())