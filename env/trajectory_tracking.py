import os
import time
import torch
from torch import nn
from omegaconf import DictConfig
from pathlib import Path

from utils.nn import MLP
from utils.randomize import randomize_parameters
from utils.replay_multi import MultiDroneRenderer
from utils.trajectory import TrajectoryManager
from utils.math import acc_to_quat

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, SRT, CTBR, LVYR

ACT_DIMS = {
    "srt":    4,
    "ctbr":   4,
    "lvyr":   4,
    "lvyr+g": 7,
}

def reset(cfg, traj, quadrotor, controller):
    pos0, vel0, acc0, _ = traj.get_reference(0)
    states = torch.zeros((cfg.num_envs, 13), device=cfg.device)
    states[:, 0:3]  = pos0.detach()
    states[:, 3:6]  = vel0.detach()
    states[:, 6:10] = acc_to_quat(acc0.detach())

    params = randomize_parameters(cfg.dynamics, cfg.num_envs, cfg.device)
    quadrotor.set_parameters(params)
    controller.update_params(
        # mass         = quadrotor.m,
        alloc_matrix = quadrotor._alloc_matrix,
        J            = quadrotor.J,
    )
    return states, params


def reset_terminated(states, terminated, pos_ref, vel_ref, acc_ref):
    if not terminated.any():
        return states
    idx = terminated.nonzero(as_tuple=True)[0]
    states[idx, 0:3]   = pos_ref[idx].detach()
    states[idx, 3:6]   = vel_ref[idx].detach()
    states[idx, 6:10]  = acc_to_quat(acc_ref[idx].detach())
    states[idx, 10:13] = 0.0
    return states


def get_observation(states, pos_ref, vel_ref, acc_ref):
    return torch.cat([
        pos_ref - states[:, 0:3],   # position error  (N, 3)
        vel_ref - states[:, 3:6],   # velocity error  (N, 3)
        acc_ref,                    # reference acc   (N, 3)
        states[:, 6:10],            # quaternion      (N, 4)
        states[:, 10:13],           # body rates      (N, 3)
    ], dim=-1)                      # (N, 16)


def compute_loss(states, pos_ref, vel_ref, acc_ref, weights, mask=None):
    if mask is not None and not mask.any():
        return states.sum() * 0.0

    s, p, v, a = (x[mask] if mask is not None else x
                  for x in (states, pos_ref, vel_ref, acc_ref))

    pos_loss  = torch.linalg.norm(p - s[:, 0:3], dim=-1).mean()
    vel_loss  = torch.linalg.norm(v - s[:, 3:6], dim=-1).mean()
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    from utils.math import quat_to_rotmat
    gravity    = torch.tensor([0., 0., -9.81], device=s.device)
    thrust_dir = torch.nn.functional.normalize(a + gravity, dim=-1)
    body_z     = quat_to_rotmat(s[:, 6:10])[:, :, 2]
    att_loss   = (1.0 - (body_z * thrust_dir).sum(dim=-1).clamp(-1, 1)).mean()

    return (weights.pos * pos_loss + weights.vel * vel_loss +
            weights.att * att_loss + weights.body_rates * rate_loss)

def build_controller(cm_type, quadrotor, cfg):
    alloc = DirectAllocation(quadrotor._alloc_matrix)
    # TODO: ideally most of the params should be gotten from quadrotor object
    if cm_type == "srt":
        return SRT()
    elif cm_type == "ctbr":
        return CTBR(
            allocator=alloc,
            J=quadrotor.J,
            # max_thrust=cfg.dynamics.max_thrust, # TODO
            # min_thrust=cfg.dynamics.min_thrust,
            max_rate=cfg.env.w_max,
            kp_rate=cfg.env.kp_rate,
            dt=cfg.dt
        )
    elif cm_type in ("lvyr", "lvyr+g"):
        return LVYR(
            allocator=alloc,
            m=quadrotor.m,
            J=quadrotor.J,
            g=quadrotor.g,
            # max_vel=cfg.env.v_max,
            # max_yaw_rate=cfg.env.w_max,
        )

    elif cm_type == "lvyr-indi":
        raise NotImplementedError("INDI inner loop not yet implemented")
    else:
        raise ValueError(f"Unknown control mode: {cm_type}")
    
def train(cfg: DictConfig):
    start    = time.time()
    dt, device, num_envs, cm = cfg.dt, cfg.device, cfg.num_envs, cfg.cm

    out_dir  = f"/home/adame/torchAirBender/outputs/policies/TT/{cm}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*85}")
    print(f" {'-'*26}  Training Trajectory Tracking  {'-'*26} ")
    print(f"  Control Mode: {cm}  |  Envs: {num_envs}  |  Episodes: {cfg.episodes}  |  Steps: {cfg.steps}  |  Horizon: {cfg.truncation}")
    print(f"{'='*85}\n")

    quadrotor  = QuadrotorDynamics(cfg)
    controller = build_controller(cm, quadrotor, cfg)
    policy    = MLP(layer_sizes=list(cfg.env.policy) + [ACT_DIMS[cm]], 
                    activation=nn.Tanh,                             # Tanh seems to work better than ReLU, but a bit more unstable sometimes
                    output_activation=nn.Sigmoid(), 
                    output_bias_init=0.0
                    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)
    traj      = TrajectoryManager.from_harmonics(cfg.env.traj, num_envs, device)

    best_loss    = float("inf")
    traj_env0    = torch.empty((cfg.steps, 16 + 4 + ACT_DIMS[cm]), device=device)

    for ep in range(cfg.episodes):
        traj.randomize()
        states, last_params = reset(cfg, traj, quadrotor, controller)

        ep_loss, num_updates  = 0.0, 0
        sq_error_sum          = torch.zeros(1, device=device)
        num_samples           = 0
        window_loss           = torch.zeros(1, device=device)
        window_start          = 0

        for t in range(cfg.steps):
            if t % cfg.truncation == 0:
                states       = states.detach()
                window_start = t
                window_loss  = torch.zeros(1, device=device)

            pos_ref, vel_ref, acc_ref, _ = traj.get_reference(t)

            obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
            raw     = policy(obs)
            
            if cm == "lvyr_g":
                gains = raw[:, 4:7]   # (N, 3) — kv, kR, kw per env
                raw   = raw[:, 0:4]   # (N, 4) — the actual control commands
                actions = controller(states, raw, gains=gains)
            else:
                actions = controller(states, raw)

            # Forward pass through the dynamics
            states  = quadrotor.step(states, actions)

            dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target

            sq_error_sum += (dist ** 2).sum()
            num_samples  += dist.numel()

            window_loss += compute_loss(
                states, pos_ref, vel_ref, acc_ref,
                cfg.env.loss_weights, mask=~too_far
            )

            if ep == cfg.episodes - 1:
                if ep == cfg.episodes - 1:
                    traj_env0[t] = torch.cat(
                        [states[0].detach(), actions[0].detach(), pos_ref[0].detach(), raw[0].detach()], dim=0
                    )

            if (t + 1) % cfg.truncation == 0 or (t + 1) == cfg.steps:
                loss = window_loss / ((t + 1) - window_start)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss     += loss.item()
                num_updates += 1

            states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)

        avg_loss = ep_loss / max(num_updates, 1)
        rmse     = torch.sqrt(sq_error_sum / num_samples).item()
        print(f"  Episode {ep+1:>4}/{cfg.episodes}  |  Loss: {avg_loss:.4f}  |  RMSE: {rmse:.3f} m")

        if rmse < cfg.env.rmse_threshold:
            print(f"  >> RMSE threshold reached, w: {cfg.env.traj.w:.2f} → {cfg.env.traj.w + 0.25:.2f} 🔥")
            torch.save(policy.state_dict(), os.path.join(out_dir, f"policy_w{cfg.env.traj.w:.2f}.pt"))
            cfg.env.traj.w += 0.25

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), os.path.join(out_dir, "policy_best.pt"))

    torch.save(policy.state_dict(), os.path.join(out_dir, "policy_final.pt"))
    print(f"\nTotal training time: {time.time() - start:.1f}s")

    # MultiDroneRenderer(
    #     # drones         = traj_env0.cpu().numpy(),
    #     trajectory         = traj_env0.cpu().numpy(),
    #     ref_trajectory = traj_env0[:, 17:20].cpu().numpy(),
    #     arm_length     = float(last_params.arm_length[0].cpu()),
    #     arm_angle      = float(last_params.arm_angle[0].cpu()),
    #     mass           = float(last_params.mass[0].cpu()),
    #     dt             = cfg.dt,
    # ).run()

    # # TODO: Add this to a plotter script to analyze info
    # import matplotlib.pyplot as plt
    # import numpy as np

    # # convert to cpu numpy
    # traj_np = traj_env0.cpu().numpy()

    # timee = np.arange(cfg.steps) * dt 

    # # Plot motor commands
    # plt.figure(figsize=(10, 5))
    # actions = traj_np[:, 13:17]  # 4 motors
    # plt.plot(timee, actions[:, 0], label="motor 1")
    # plt.plot(timee, actions[:, 1], label="motor 2")
    # plt.plot(timee, actions[:, 2], label="motor 3")
    # plt.plot(timee, actions[:, 3], label="motor 4")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Motor command")
    # plt.title("Quadrotor Motor Commands Over Rollout")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plot control gains
    # plt.figure(figsize=(10, 5))
    # actions = traj_np[:, 20:27]  # control commands and gains
    # plt.plot(timee, actions[:, 0] * 10.0, label="vx")
    # plt.plot(timee, actions[:, 1] * 10.0, label="vy")
    # plt.plot(timee, actions[:, 2] * 10.0, label="vz")
    # plt.plot(timee, actions[:, 3], label="wz")
    # plt.plot(timee, actions[:, 4] * 5.0, label="kv")
    # plt.plot(timee, actions[:, 5] * 5.0, label="kR")
    # plt.plot(timee, actions[:, 6] * 2.0, label="kW")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Control value")
    # plt.title("Quadrotor Actions Over Rollout")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


# ==================================================================
#                    TESTING THE POLICIES
# ==================================================================

def test(cfg: DictConfig):
    # ── Manual configuration ─────────────────────────────────────────────
    policies = [
        {"cm": "srt",  
        "path": "/home/adame/torchAirBender/outputs/policies/TT/srt/policy_final.pt",  
        "color": (0.2, 1.0, 0.4)}, 

        {"cm": "ctbr", 
         "path": "/home/adame/torchAirBender/outputs/policies/TT/ctbr/policy_final.pt", 
         "color": (0.2, 0.6, 1.0)},

        {"cm": "lvyr", 
         "path": "/home/adame/torchAirBender/outputs/policies/TT/lvyr/policy_final.pt", 
         "color": (1.0, 0.4, 0.1)},

        {"cm": "lvyr+g", 
         "path": "/home/adame/torchAirBender/outputs/policies/TT/lvyr+g/policy_final.pt", 
         "color": (0.8, 0.2, 1.0)},
    ]
    for p in policies:
        p["label"] = p["cm"] + "_" + Path(p["path"]).stem.split("_", 1)[-1]
    randomize = True   # set False for identical dynamics across all policies
    seed      = None
    # ─────────────────────────────────────────────────────────────────────

    if seed is not None:
        torch.manual_seed(seed)

    device = cfg.device

    quadrotor = QuadrotorDynamics(cfg)

    # Single shared trajectory, fixed by seed
    traj = TrajectoryManager.from_harmonics(cfg.env.traj, cfg.num_envs, device)
    traj.randomize()

    drones   = []
    ref_traj = None

    for spec in policies:
        print(f"  Rolling out: {spec['label']}  ({spec['path']})")

        controller = build_controller(spec["cm"], quadrotor, cfg)
        policy = MLP(
            layer_sizes=list(cfg.env.policy) + [ACT_DIMS[spec["cm"]]],
            activation=nn.Tanh,
            output_activation=nn.Sigmoid(),
            output_bias_init=0.0,
        ).to(device)
        policy.load_state_dict(torch.load(spec["path"], map_location=device))
        policy.eval()

        # Reset state
        pos0, vel0, acc0, _ = traj.get_reference(0)
        states = torch.zeros((cfg.num_envs, 13), device=device)
        states[:, 0:3]  = pos0.detach()
        states[:, 3:6]  = vel0.detach()
        states[:, 6:10] = acc_to_quat(acc0.detach())

        if randomize:
            params = randomize_parameters(cfg.dynamics, cfg.num_envs, device)
            quadrotor.set_parameters(params)
            controller.update_params(
                alloc_matrix=quadrotor._alloc_matrix,
                J=quadrotor.J,
            )

        traj_data = torch.empty((cfg.steps, 20), device=device)

        with torch.no_grad():
            for t in range(cfg.steps):
                pos_ref, vel_ref, acc_ref, _ = traj.get_reference(t)

                obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
                raw     = policy(obs)
                if spec["cm"] == "lvyr+g":
                    gains = raw[:, 4:7]   # (N, 3) — kv, kR, kw per env
                    raw   = raw[:, 0:4]   # (N, 4) — the actual control commands
                    actions = controller(states, raw, gains=gains)
                else:
                    actions = controller(states, raw)
                states  = quadrotor.step(states, actions)

                dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
                too_far = dist > cfg.env.max_dist_to_target

                traj_data[t] = torch.cat(
                    [states[0].detach(), actions[0].detach(), pos_ref[0].detach()], dim=0
                )

                states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_ref)

        traj_np  = traj_data.cpu().numpy()
        ref_traj = ref_traj if ref_traj is not None else traj_np[:, 17:20]
        
        drones.append({
            "traj":  traj_np,
            "color": spec.get("color", (0.2, 0.6, 1.0)),
            "label": spec["label"],
        })

    # Render
    last_params = randomize_parameters(cfg.dynamics, 1, device)   # TODO: This is wrong...params should be saved in drones list...
    MultiDroneRenderer(
        drones         = drones,
        ref_trajectory = ref_traj,
        arm_length     = float(last_params.arm_length[0].cpu()),
        arm_angle      = float(last_params.arm_angle[0].cpu()),
        mass           = float(last_params.mass[0].cpu()),
        dt             = cfg.dt,
    ).run()
