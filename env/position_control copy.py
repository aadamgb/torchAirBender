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
        states
        ):
    
    num_envs = cfg.num_envs
    device = cfg.device

    # ------------ Randomize initial position -----------
    states[:, :3] = torch.rand((num_envs, 3), device=device)
    states[:, 6] = 1.0  # seting quaternion w to 1 (FIXED)

    # ----------- Randomize target positon --------------
    target = torch.rand((num_envs, 3), device=device)

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
    return states, target, params

def get_observation(
        states: Tensor, 
        targets: Tensor
        )-> Tensor:
    
    p       = states[:, 0:3]
    rest    = states[:, 3:13]   # v, q, w
    p_error = targets - p

    return torch.cat([p_error, rest], dim=-1)


def compute_loss(
    states:  Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Mean position tracking error across all environments.

    Returns a scalar — mean L2 distance to target over the batch.

    Args:
        states  : (N, 13)
        targets : (N, 3)

    Returns:
        loss : scalar tensor
    """
    # Position error
    p_error = targets - states[:, 0:3]

    # L2 norm over position dimension
    distances = torch.linalg.norm(p_error, dim=-1)

    # Mean over batch
    return distances.mean()




def train(cfg: DictConfig):
    dt = cfg.dt
    device = cfg.device
    num_envs = cfg.num_envs
    episodes = cfg.episodes
    steps = cfg.steps
    truncation = 100
    
    print(f"\n{'='*75}")
    print(f"  Position Control Training")
    print(f"  Envs: {cfg.num_envs}  |  "
          f"  Episodes     : {episodes}  |  "
          f"  Steps: {cfg.steps}  |  "
        )
    print(f"{'='*75}\n")

    # Initialize tensors
    targets = torch.rand((num_envs, 3), device=device)
    states = torch.zeros((num_envs, 13), device=device)

    # Pre-allocate horizon loss buffer (replace the list approach)
    horizon_losses = torch.empty(truncation, device=device)
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
        states, targets, randomized_params = reset(cfg, states)
        quadrotor.set_parameters(randomized_params)

        ep_loss = 0.0
        num_updates = 0

        for t in range(steps):

            # --- Truncation boundary: detach state to cut the compute graph ---
            if t % truncation == 0:
                states = states.detach()
                horizon_start = t  # track where the window began
            
            obs = get_observation(states, targets)
            actions = policy(obs) * 8.0
            states = quadrotor.step(state=states, action=actions)

            # Accumulate per-step loss within the window
            horizon_losses[t % truncation] = compute_loss(states, targets).detach()

            # Save env 0 last episode trajectory for visualization
            if ep == episodes - 1:
                traj_env0[t] = torch.cat([states[0].detach(), actions[0].detach()], dim=0) 

            # --- Update at end of each truncation window ---
            if (t + 1) % truncation == 0 or (t + 1) == steps:
                horizon_len = (t + 1) - horizon_start
                loss = horizon_losses[:horizon_len].mean()

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                ep_loss += loss.item()
                num_updates += 1

        avg_ep_loss = ep_loss / num_updates
        print(f"  Episode {ep+1:>4}/{episodes}  |  Avg Loss: {avg_ep_loss:.4f}")


    # Replay the trajectory
    renderer = PositionControlRenderer(
        target_pos=targets[0].detach().cpu().numpy(),
        trajectory=traj_env0.detach().cpu().numpy(),  # (T, 13)
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )
    renderer.run()