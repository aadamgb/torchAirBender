import os
import time
import torch
from torch import Tensor
from torch import nn

from omegaconf import DictConfig, OmegaConf

from utils.nn import MLP
from utils.randomize import QuadrotorParams, randomize_parameters
from utils.replay import RacingRenderer
from utils.math import quat_to_rotmat

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import SRTController_old

from miscellaneous.loader import load_gates_from_yaml, load_TOGT


# ==============================================================
# Trajectory helpers
# ==============================================================

def generate_trajectory_params(
    num_envs: int,
    device,
    cfg: DictConfig,
) -> dict:
    """
    Generates random harmonic trajectory coefficients for each environment.

    Returns a dict with tensors of shape (num_envs, 3, num_harmonics):
        amps     : amplitudes
        freqs    : frequencies
        phases   : phases
        z_offset : scalar float added to z position at eval time
    """
    amps   = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.A + 0.5
    freqs  = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.w + 0.2
    phases = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.phi * torch.pi

    return {"amps": amps, "freqs": freqs, "phases": phases, "z_offset": cfg.z_offset}


def get_target(
    t: float,
    params: dict,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Evaluates the trajectory at time t for all environments.

    Args:
        t      : current time (scalar float)
        params : dict from generate_trajectory_params

    Returns:
        pos : (num_envs, 3)
        vel : (num_envs, 3)
        acc : (num_envs, 3)
    """
    amps, freqs, phases = params["amps"], params["freqs"], params["phases"]

    t_tensor = torch.full(
        (amps.shape[0], 1, 1), t, device=amps.device, dtype=amps.dtype
    )

    angle = freqs * t_tensor + phases                               # (N, 3, H)

    pos = torch.sum(amps * torch.sin(angle),              dim=2)    # (N, 3)
    vel = torch.sum(amps * freqs * torch.cos(angle),      dim=2)    # (N, 3)
    acc = torch.sum(-amps * freqs**2 * torch.sin(angle),  dim=2)    # (N, 3)

    pos[:, 2] = pos[:, 2] + params["z_offset"]

    return pos, vel, acc


def build_get_ref(cfg: DictConfig, device, dt: float, speed_scale=1.0 ):
    """
    Returns a get_ref(t: int) -> (pos, vel, acc) callable.

    If cfg.load_traj is True, loads from CSV (positions broadcast to all envs).
    Otherwise generates random harmonic trajectory params per env.

    Returns:
        get_ref    : callable(t: int) -> (pos, vel, acc) each (N, 3)
        traj_params: dict or None (None when loading from CSV)
    """
    if cfg.env.load_traj == True:
        # print("Loading trajectory from CSV!")
        loaded = load_TOGT(cfg.env.test_traj, cfg.steps, device)

        def get_ref(t: int):
            # map t -> scaled index into the trajectory
            scaled_t = min(int(t * speed_scale), len(loaded["pos"]) - 1)
            pos = loaded["pos"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1)
            vel = loaded["vel"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1) * speed_scale
            acc = loaded["acc"][scaled_t].unsqueeze(0).expand(cfg.num_envs, -1) * speed_scale ** 2
            return pos, vel, acc

        return get_ref, None

    else:
        traj_params = generate_trajectory_params(cfg.num_envs, device, cfg.env.traj)

        def get_ref(t: int):
            return get_target(t * dt, traj_params)

        return get_ref, traj_params


# ==============================================================
# State helpers
# ==============================================================

def acc_to_quat(acc_ref: Tensor, g: float = 9.81) -> Tensor:
    """
    Computes desired quaternion from reference acceleration.
    Desired thrust direction = acc_ref + gravity_vector.

    Args:
        acc_ref : (N, 3)
    Returns:
        q_des   : (N, 4)  [w, x, y, z]
    """
    gravity    = torch.tensor([0.0, 0.0, g], device=acc_ref.device)
    thrust_dir = acc_ref + gravity
    thrust_dir = torch.nn.functional.normalize(thrust_dir, dim=-1)

    world_z       = torch.zeros_like(thrust_dir)
    world_z[:, 2] = 1.0

    axis = torch.linalg.cross(world_z, thrust_dir)                   # (N, 3)
    dot  = (world_z * thrust_dir).sum(dim=-1, keepdim=True)          # (N, 1)

    w   = torch.sqrt(torch.clamp((1.0 + dot) / 2.0, min=1e-6))
    xyz = torch.nn.functional.normalize(axis, dim=-1) * torch.sqrt(
        torch.clamp((1.0 - dot) / 2.0, min=0.0)
    )

    return torch.cat([w, xyz], dim=-1)                               # (N, 4)


def reset(
    cfg: DictConfig,
    pos0: Tensor,
    vel0: Tensor,
    acc0: Tensor,
) -> tuple[Tensor, QuadrotorParams]:
    """Reset all envs to the given t=0 reference."""
    num_envs = cfg.num_envs
    device   = cfg.device

    states = torch.zeros((num_envs, 13), device=device)
    states[:, 0:3]  = pos0.detach()
    states[:, 3:6]  = vel0.detach()
    states[:, 6:10] = acc_to_quat(acc0.detach())

    params = randomize_parameters(cfg.dynamics, num_envs, device)
    return states, params


def reset_terminated(
    states:     Tensor,
    terminated: Tensor,    # (N,) bool
    pos_ref:    Tensor,    # (N, 3)
    vel_ref:    Tensor,    # (N, 3)
    acc_ref:    Tensor,    # (N, 3)
) -> Tensor:
    """Snap terminated envs back to current reference position and velocity."""
    if not terminated.any():
        return states

    idx = terminated.nonzero(as_tuple=True)[0]

    states[idx, 0:3]   = pos_ref[idx].detach()
    states[idx, 3:6]   = vel_ref[idx].detach()
    states[idx, 6:10]  = acc_to_quat(acc_ref[idx].detach())
    states[idx, 10:13] = 0.0

    return states


# ==============================================================
# Observation & loss
# ==============================================================

def get_observation(
    states:  Tensor,
    pos_ref: Tensor,   # (N, 3)
    vel_ref: Tensor,   # (N, 3)
    acc_ref: Tensor,   # (N, 3)
) -> Tensor:
    p = states[:, 0:3]
    v = states[:, 3:6]
    q = states[:, 6:10]
    w = states[:, 10:13]

    p_error = pos_ref - p
    v_error = vel_ref - v

    return torch.cat([p_error, v_error, acc_ref, q, w], dim=-1)    # (N, 16)


def compute_loss(
    states:  Tensor,
    pos_ref: Tensor,
    vel_ref: Tensor,
    acc_ref: Tensor,
    weights: DictConfig,
    mask:    Tensor | None = None,
) -> Tensor:
    if mask is not None and not mask.any():
        return states.sum() * 0.0

    s = states[mask] if mask is not None else states
    p = pos_ref[mask] if mask is not None else pos_ref
    v = vel_ref[mask] if mask is not None else vel_ref
    a = acc_ref[mask] if mask is not None else acc_ref

    pos_loss  = torch.linalg.norm(p - s[:, 0:3], dim=-1).mean()
    vel_loss  = torch.linalg.norm(v - s[:, 3:6], dim=-1).mean()
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    gravity    = torch.tensor([0.0, 0.0, -9.81], device=s.device)
    thrust_dir = torch.nn.functional.normalize(a + gravity, dim=-1)
    R          = quat_to_rotmat(s[:, 6:10])
    body_z     = R[:, :, 2]
    cos_angle  = (body_z * thrust_dir).sum(dim=-1).clamp(-1, 1)
    att_loss   = (1.0 - cos_angle).mean()

    return (
        weights.pos        * pos_loss  +
        weights.vel        * vel_loss  +
        weights.att        * att_loss  +
        weights.body_rates * rate_loss
    )


# ==============================================================
# Renderer helper
# ==============================================================

def _build_racing_renderer(
    cfg:               DictConfig,
    traj:              Tensor,          # (steps, 20)
    randomized_params: QuadrotorParams,
    dt:                float,
    ref_trajectory=None,                # (steps, 3) numpy or None
) -> RacingRenderer:
    gates_position, gates_rpy = load_gates_from_yaml(
        "/home/adame/torchAirBender/miscellaneous/race_tracks/uzh_7g_moved.yaml"
    )
    return RacingRenderer(
        gates_position=gates_position,
        gates_rpy=gates_rpy,
        gate_mesh_path="/home/adame/torchAirBender/miscellaneous/gate.obj",
        gate_scale=1.0,
        gate_color=(0.25, 0.0, 0.5),
        ref_trajectory=ref_trajectory,
        trajectory=traj[:, :17].cpu().numpy(),
        arm_length=float(randomized_params.arm_length[0].cpu()),
        arm_angle=float(randomized_params.arm_angle[0].cpu()),
        mass=float(randomized_params.mass[0].cpu()),
        dt=dt,
    )


# ==============================================================
# Train
# ==============================================================

def train(cfg: DictConfig):
    start      = time.time()
    output_dir = "/home/adame/torchAirBender/outputs/policies/racing"
    os.makedirs(output_dir, exist_ok=True)

    dt         = cfg.dt
    device     = cfg.device
    num_envs   = cfg.num_envs
    episodes   = cfg.episodes
    steps      = cfg.steps
    truncation = cfg.truncation

    print(f"\n{'='*75}")
    print(f"  Racing Training")
    print(f"  Envs: {num_envs}  |  Episodes: {episodes}  |  Steps: {steps}  |  Horizon: {truncation}")
    print(f"{'='*75}\n")

    truncation_losses = torch.empty(truncation, device=device)
    traj_env0         = torch.empty((steps, 20), device=device)    # 13 state + 4 actions + 3 ref pos

    quadrotor = QuadrotorDynamics(cfg)
    policy = MLP(
        layer_sizes=cfg.env.nn,
        activation=nn.ReLU,
        output_activation=nn.Sigmoid(),
        output_bias_init=0.0,
    ).to(device)
    policy.load_state_dict(torch.load(
        "/home/adame/torchAirBender/outputs/las_mejores/show2rob.pt",
        map_location=device,
    ))

    controller = SRTController_old(cfg)
    optimizer  = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)

    best_loss = float("inf")
    speed_scale = 0.85

    for ep in range(episodes):

        # --- build ref trajectory for this episode ---
        # speed_scale = max(0.4, min(0.6, 0.2 + ep * (0.8 / episodes)))
        get_ref, _ = build_get_ref(cfg, device, dt, speed_scale=speed_scale)
        pos0, vel0, acc0 = get_ref(0)
        states, randomized_params = reset(cfg, pos0, vel0, acc0)
        quadrotor.set_parameters(randomized_params)

        ep_loss     = 0.0
        num_updates = 0

        for t in range(steps):

            if t % truncation == 0:
                states          = states.detach()
                window_start    = t
                window_loss_sum = torch.zeros(1, device=device)

            pos_ref, vel_ref, acc_ref = get_ref(t)

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            raw     = policy(obs)
            actions = controller(raw)
            states  = quadrotor.step(state=states, action=actions)

            dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target
            alive   = ~too_far

            step_loss = compute_loss(
                states, pos_ref, vel_ref, acc_ref, cfg.env.loss_weights, mask=alive
            )

            # Maybe remove idk
            if too_far.any():
                termination_penalty = too_far.float() * cfg.env.termination_penalty
                step_loss = step_loss + termination_penalty.mean()

            truncation_losses[t % truncation] = step_loss.detach()
            window_loss_sum = window_loss_sum + step_loss

            if ep == episodes - 1:
                traj_env0[t] = torch.cat(
                    [states[0].detach(), actions[0].detach(), pos_ref[0].detach()],
                    dim=0,
                )

            if (t + 1) % truncation == 0 or (t + 1) == steps:
                window_len = (t + 1) - window_start
                loss = window_loss_sum / window_len
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss     += truncation_losses[:window_len].mean().item()
                num_updates += 1

            states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)

        avg_ep_loss = ep_loss / num_updates
        if avg_ep_loss < 1.5:
            print(f"Increasing speed scale by 0.025 🔥🔥")
            torch.save(policy.state_dict(), os.path.join(output_dir, f"racing_at_{speed_scale:.2f}.pt"))
            speed_scale += 0.01

        print(f"  Episode {ep+1:>4}/{episodes}  |  Avg Loss: {avg_ep_loss:.4f}")
        print(f"   - Speed scale is: {speed_scale:.2f}")

        if avg_ep_loss < best_loss:
            best_loss = avg_ep_loss
            torch.save(policy.state_dict(), os.path.join(output_dir, "racing_best.pt"))

    print(f"Total training time: {time.time() - start:.2f}s")
    torch.save(policy.state_dict(), os.path.join(output_dir, "racing_final.pt"))

    renderer = _build_racing_renderer(
        cfg, traj_env0, randomized_params, dt,
        ref_trajectory=traj_env0[:, 17:20].cpu().numpy(),
    )
    renderer.run()


# ==============================================================
# Test
# ==============================================================

def test(cfg: DictConfig):
    policy_path = "/home/adame/torchAirBender/outputs/las_mejores/racing/racing_at_0.87.pt"
    dt     = cfg.dt
    device = cfg.device
    steps  = cfg.steps

    # single env
    cfg_dict              = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["num_envs"]  = 1
    cfg                   = OmegaConf.create(cfg_dict)

    # load policy
    policy = MLP(
        layer_sizes=cfg.env.nn,
        activation=nn.ReLU,
        output_activation=nn.Sigmoid(),
        output_bias_init=0.0,
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    print(f"Loaded policy from: {policy_path}")

    # build ref callable
    get_ref, _ = build_get_ref(cfg, device, dt, speed_scale=0.87)

    # init
    controller        = SRTController_old(cfg)
    pos0, vel0, acc0  = get_ref(0)
    states, randomized_params = reset(cfg, pos0, vel0, acc0)

    quadrotor = QuadrotorDynamics(cfg)
    quadrotor.set_parameters(randomized_params)

    traj = torch.empty((steps, 20), device=device)

    print(f"\n{'='*60}")
    print(" Testing Racing ")
    print(f"{'='*60}\n")

    total_loss   = 0.0
    sq_error_sum = 0.0

    with torch.inference_mode():
        for t in range(steps):

            pos_ref, vel_ref, acc_ref = get_ref(t)

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            raw     = policy(obs)
            actions = controller(raw)
            states  = quadrotor.step(state=states, action=actions)

            dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target

            sq_error_sum += (dist ** 2).item()

            step_loss   = compute_loss(states, pos_ref, vel_ref, acc_ref, cfg.env.loss_weights)
            total_loss += step_loss.item()

            traj[t] = torch.cat([states[0], actions[0], pos_ref[0]], dim=0)

            if too_far[0]:
                print(f"  !! Terminated at step {t+1} — dist: {dist[0]:.3f}")
                states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)

    print(f"\n  Avg Loss : {total_loss / steps:.4f}")
    print(f"  RMSE Pos : {(sq_error_sum / steps) ** 0.5:.4f} m")

    print(f"Max thrust per motor: {actions.max():.3f}")
    print(f"Min thrust per motor: {actions.min():.3f}")
    print(f"Max velocity reached: {traj[:, 3:6].norm(dim=-1).max():.3f} m/s")
    print(f"Max ref velocity:     {traj[:, 17:20].diff(dim=0).norm(dim=-1).max() / dt:.3f} m/s")

    renderer = _build_racing_renderer(
        cfg, traj, randomized_params, dt,
        ref_trajectory=traj[:, 17:20].cpu().numpy(),
    )
    renderer.run()

    import matplotlib.pyplot as plt
    import numpy as np

    # convert to cpu numpy
    traj_np = traj.cpu().numpy()

    actions = traj_np[:, 13:17]  # 4 motors
    time = np.arange(steps) * dt

    plt.figure(figsize=(10,5))

    plt.plot(time, actions[:,0], label="motor 1")
    plt.plot(time, actions[:,1], label="motor 2")
    plt.plot(time, actions[:,2], label="motor 3")
    plt.plot(time, actions[:,3], label="motor 4")

    plt.xlabel("Time [s]")
    plt.ylabel("Motor command")
    plt.title("Quadrotor Actions Over Rollout")
    plt.legend()
    plt.grid(True)

    plt.show()