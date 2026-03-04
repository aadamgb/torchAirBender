import torch
from torch import Tensor
from torch import nn
from omegaconf import DictConfig

from utils.nn import MLP 
from utils.randomize import randomize_parameters
from utils.replay import PositionControlRenderer

from dynamics.quadrotor_dynamics import QuadrotorDynamics


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
    # Generting key for reproducability
    # generator = torch.Generator(device=device)
    # generator.manual_seed(42)
    params = randomize_parameters(
        cfg=cfg.dynamics,
        num_envs=num_envs,
        device=device,
        # generator=generator,
    )
    return states, targets, params

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
    lambda_pos: float = 0.75,
    lambda_rate: float = 0.1,
    lambda_vel: float = 0.25,
) -> Tensor:
    """
    Position tracking + distance-weighted velocity damping + body rate penalty.

    Encourages the drone to slow down when near the target.
    """

    # --------------------------------
    # Position term
    # --------------------------------
    p_error = targets - states[:, 0:3]                  # (N, 3)
    distances = torch.linalg.norm(p_error, dim=-1)      # (N,)
    pos_loss = distances.mean()

    # --------------------------------
    # Distance-weighted velocity term
    # --------------------------------
    velocities = states[:, 3:6]                         # (N, 3)
    # vel_loss = (velocities ** 2).sum(dim=-1)            # (N,)
    vel_loss = (velocities ** 2).sum(dim=-1).mean()            # (N,)

    # Weight increases near target
    # vel_weight = 1.0 / (1.0 + distances)                # (N,)
    # vel_loss = (vel_weight * speed_sq).mean()

    # --------------------------------
    # Body rate penalty
    # --------------------------------
    omega = states[:, 10:13]                            # (N, 3)
    rate_loss = (omega ** 2).sum(dim=-1).mean()

    # --------------------------------
    # Total loss
    # --------------------------------
    loss = (
        lambda_pos  * pos_loss
        + lambda_vel * vel_loss
        + lambda_rate * rate_loss
    )

    return loss



def train(cfg: DictConfig):
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
            states = quadrotor.step(state=states, action=actions)

            step_loss = compute_loss(states, targets)

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
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                ep_loss += truncation_losses[:window_len].mean().item()
                num_updates += 1

        avg_ep_loss = ep_loss / num_updates
        if (ep + 1) % 10 == 0: 
            print(f"  Episode {ep+1:>4}/{episodes}  |  Avg Loss: {avg_ep_loss:.4f}")


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