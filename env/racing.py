import os
import csv
import time
import torch
from torch import nn
from omegaconf import DictConfig
from pathlib import Path

from utils.nn import MLP
from utils.randomize import randomize_parameters
from utils.replay_multi import MultiDroneRenderer
from utils.trajectory import TrajectoryManager
from utils.math import acc_to_quat, quat_to_rotmat, quat_multiply
from utils.plotter import plot_rollout

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, SRT, CTBR, LVHR

from utils.plot_noise import StateLogger


ACT_DIMS = {"srt": 4, "ctbr":  4, "lvhr": 4,  "lvhr+g": 7}
CM_COLS  = {"srt": 30, "ctbr": 34, "lvhr": 38, "lvhr+g": 41}

CONTROL_HZ   = 100
PHYSICS_HZ   = 1000
SUBSTEPS     = PHYSICS_HZ // CONTROL_HZ   # 10

dt_control   = 1.0 / CONTROL_HZ           # 0.01  s
dt_physics   = 1.0 / PHYSICS_HZ           # 0.001 s

def reset_noise_biases(cfg) -> dict:
    N, device, mn = cfg.num_envs, cfg.device, cfg.env.m_noise
    return {
        "vel":  torch.randn(N, 3, device=device) * mn.v_bias_std,
        "rate": torch.randn(N, 3, device=device) * mn.w_bias_std,
    }

def reset(cfg, traj, quadrotor, controller):
    pos0, vel0, acc_lin0, *_ = traj.get_reference(0)
    states = torch.zeros((cfg.num_envs, 17), device=cfg.device)
    states[:, 0:3]  = pos0.detach()
    states[:, 3:6]  = vel0.detach()
    # states[:, 6:10] = acc_to_quat(acc_lin0.detach())
    states[:, 6] = 1.0

    params = randomize_parameters(cfg.dynamics, cfg.num_envs, cfg.device)
    quadrotor.set_parameters(params)
    controller.update_params(
        alloc_matrix = quadrotor._alloc_matrix,
        mass         = quadrotor.m,
        max_TWR      = quadrotor.max_TWR,
        J            = quadrotor.J,
    )
    biases = reset_noise_biases(cfg)
    return states, params, biases


def reset_terminated(states, terminated, pos_ref, vel_ref, acc_ref):
    if not terminated.any():
        return states
    idx = terminated.nonzero(as_tuple=True)[0]
    states[idx, 0:3]   = pos_ref[idx].detach()
    states[idx, 3:6]   = vel_ref[idx].detach()
    states[idx, 6:10]  = acc_to_quat(acc_ref[idx].detach())
    states[idx, 10:13] = 0.0
    states[idx, 13:17] = 0.0
    return states


def get_observation(cfg, states, pos_ref, vel_ref, acc_lin_ref, biases, logger=None):
    N = cfg.num_envs
    device = cfg.device
    mn     = cfg.env.m_noise

    # Add "measurement" noise
    with torch.inference_mode():
        noisy_states = states.clone()

        noisy_states[:, 0:3] += torch.randn(N, 3, device=cfg.device) \
        * mn.p_std

        noisy_states[:, 3:6] += torch.randn(N, 3, device=cfg.device) \
        * mn.v_std + torch.randn(N, 3, device=device) \
        * biases["vel"]

        noisy_states[:, 10:13] += torch.randn(N, 3, device=cfg.device) \
        * mn.w_std + torch.randn(N, 3, device=device) \
        * biases["rate"]

        attitude = quat_to_rotmat(noisy_states[:, 6:10]).reshape(N, 9)

        # Attitude Noise (maybe remove idk)
        angle_noise = torch.randn(N, 3, device=device) * mn.q_std   
        half        = torch.norm(angle_noise, dim=-1, keepdim=True).clamp(min=1e-8) / 2
        axis        = nn.functional.normalize(angle_noise, dim=-1)
        dq          = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
        noisy_q     = quat_multiply(noisy_states[:, 6:10], dq)
        noisy_q     = nn.functional.normalize(noisy_q, dim=-1)
        noisy_attitude = quat_to_rotmat(noisy_q).reshape(N, 9)

        if logger is not None:
            logger.log(states, noisy_states) 

    return torch.cat([
        pos_ref - noisy_states[:, 0:3],   
        vel_ref - noisy_states[:, 3:6],   
        acc_lin_ref,                    
        noisy_attitude,            
        noisy_states[:, 10:13],           
    ], dim=-1)                      


# def compute_loss(states, thrusts, 
#                  pos_ref, vel_ref, acc_ref,
#                  quat_ref, omega_ref, thrust_ref,
#                  w, mask=None):
#     if mask is not None and not mask.any():
#         return states.sum() * 0.0

#     s, t, p, v, qr, wr, tr = (x[mask] if mask is not None else x
#                        for x in (states, thrusts, pos_ref, vel_ref, quat_ref, omega_ref, thrust_ref))

#     pos_loss  = torch.linalg.norm(p - s[:, 0:3], dim=-1).mean()
#     vel_loss  = torch.linalg.norm(v - s[:, 3:6], dim=-1).mean()
#     omega_loss = torch.linalg.norm(wr - s[:, 10:13], dim=-1).mean()
#     quat_loss = (1.0 - (s[:, 6:10] * qr).sum(dim=-1).abs().clamp(0.0, 1.0)).mean()  # 1 - |q_ref · q|
#     # thrust_loss  = torch.linalg.norm(tr[:, [2, 0, 3, 1]] - t, dim=-1).mean()
#     thrust_loss  = torch.linalg.norm(tr[:, [3, 1, 0, 2]] - t, dim=-1).mean()

#     return (w.pos * pos_loss     + 
#             w.vel * vel_loss     +
#             w.omega * omega_loss +
#             w.quat * quat_loss   +
#             w.thrust * thrust_loss
#             )

def compute_loss(states, pos_ref, vel_ref, acc_ref, weights, mask=None):
    if mask is not None and not mask.any():
        return states.sum() * 0.0

    s, p, v, a = (x[mask] if mask is not None else x
                  for x in (states, pos_ref, vel_ref, acc_ref))

    pos_loss  = torch.linalg.norm(p - s[:, 0:3], dim=-1).mean()
    vel_loss  = torch.linalg.norm(v - s[:, 3:6], dim=-1).mean()
    rate_loss = (s[:, 10:13] ** 2).sum(dim=-1).mean()

    gravity    = torch.tensor([0., 0., -9.81], device=s.device)
    thrust_dir = torch.nn.functional.normalize(a + gravity, dim=-1)
    body_z     = quat_to_rotmat(s[:, 6:10])[:, :, 2]
    att_loss   = (1.0 - (body_z * thrust_dir).sum(dim=-1).clamp(-1, 1)).mean()

    return (weights.pos * pos_loss + weights.vel * vel_loss +
            weights.att * att_loss + weights.body_rates * rate_loss)

def build_controller(cm_type, quadrotor, cfg):
    alloc = DirectAllocation(quadrotor._alloc_matrix)
    if cm_type == "srt":
        return SRT(
            mass=quadrotor.m,
            max_TWR=quadrotor.max_TWR,
        )
    elif cm_type == "ctbr":
        return CTBR(
            allocator=alloc,
            mass=quadrotor.m,
            max_TWR=quadrotor.max_TWR,
            J=quadrotor.J,
            max_rate=cfg.env.w_max,
            kp_rate=cfg.env.kp_rate,
            dt=cfg.dt
        )
    elif cm_type in ("lvhr", "lvhr+g"):
        return LVHR(
            allocator=alloc,
            m=quadrotor.m,
            J=quadrotor.J,
            g=quadrotor.g,
        )
    else:
        raise ValueError(f"Unknown control mode: {cm_type}")
    
def train(cfg: DictConfig):
    start    = time.time()
    dt, device, num_envs, cm = cfg.dt, cfg.device, cfg.num_envs, cfg.cm

    print(f"\n{'='*85}")
    print(f" {'-'*26}  Training Trajectory Tracking  {'-'*26} ")
    print(f"  Control Mode: {cm}  |  Envs: {num_envs}  |  Episodes: {cfg.episodes}  |  Steps: {cfg.steps}  |  Horizon: {cfg.truncation}")
    print(f"{'='*85}\n")

    out_dir  = f"/home/adame/torchAirBender/outputs/policies/sample_eff/{cm}"
    os.makedirs(out_dir, exist_ok=True)

    quadrotor  = QuadrotorDynamics(cfg)
    # traj      = TrajectoryManager.from_harmonics(cfg.env.traj, num_envs, device)
    
    # logger = StateLogger(env_idx=0)   # track env 0
    logger = None

    path = "/home/adame/torchAirBender/miscellaneous/trajectories/TOGT/straight_line.csv"
    # path = "/home/adame/torchAirBender/miscellaneous/trajectories/TOGT/togt_traj.csv"
    traj = TrajectoryManager.from_togt(path, cfg.num_envs, cfg.device)

    controller = build_controller(cm, quadrotor, cfg)
    policy    = MLP(layer_sizes=list(cfg.env.policy) + [ACT_DIMS[cm]],                        
                    activation=nn.ReLU,                             
                    output_activation=nn.Sigmoid(), 
                    output_bias_init=0.0
                    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)

    traj_data = torch.empty((cfg.steps, CM_COLS[cm]), device=device)
    last_saved_w = 1.0
    SAVE_INTERVAL = 0.1
    best_loss    = float("inf")
    s = 1.0

    for ep in range(cfg.episodes):
        # traj.randomize()

        states, last_param, biases = reset(cfg, traj, quadrotor, controller)

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

            (pos_ref,  vel_ref, acc_lin_ref, acc_rot_ref,
             quat_ref, omega_ref,  thrust_ref, jerk_ref, snap_ref) = traj.get_reference(t)
            
            obs     = get_observation(cfg, states, pos_ref, vel_ref, acc_lin_ref, biases, logger=logger)
            raw     = policy(obs)

            if cm == "lvhr+g":
                actions = controller(states, raw[:, 0:4], gains=raw[:, 4:7])
            else:
                actions = controller(states, raw)
            
            # states  = quadrotor.step(states, actions[:, 0:4])            
            # states  = quadrotor.step(states, thrust_ref[:, [2, 0, 3, 1]] )
            # for _ in range(SUBSTEPS):            
                # states  = quadrotor.step(states, thrust_ref[:, [3, 1, 0, 2]])
            states  = quadrotor.step(states, actions[:, 0:4])            

            dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
            too_far = dist > cfg.env.max_dist_to_target

            sq_error_sum += (dist ** 2).sum()
            num_samples  += dist.numel()

            # window_loss += compute_loss(states, actions[:, 0:4], 
            #                             pos_ref, vel_ref, acc_lin_ref,
            #                             quat_ref, omega_ref, thrust_ref,
            #                             cfg.env.loss_weights, 
            #                             mask=~too_far)
            
            window_loss += compute_loss(states, pos_ref, vel_ref, acc_lin_ref, cfg.env.loss_weights, mask=~too_far)

            if ep == cfg.episodes - 1:
                traj_data[t] = torch.cat([
                    states[0].detach(),       # 0:17  — full state
                    pos_ref[0].detach(),      # 17:20 — p_ref
                    vel_ref[0].detach(),      # 20:23 — v_ref
                    acc_lin_ref[0].detach(),      # 23:26 — a_ref
                    actions[0].detach(),      # 26:N  — srt + (wrench + lvyr + gains)
                ], dim=0)

            if (t + 1) % cfg.truncation == 0 or (t + 1) == cfg.steps:
                loss = window_loss / ((t + 1) - window_start)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss     += loss.item()
                num_updates += 1

            states = reset_terminated(states, too_far, pos_ref, vel_ref, acc_lin_ref)

        avg_loss = ep_loss / max(num_updates, 1)
        rmse     = torch.sqrt(sq_error_sum / num_samples).item()
        print(f"  Episode {ep+1:>4}/{cfg.episodes}  |  Loss: {avg_loss:.4f}  |  RMSE: {rmse:.3f} m")

        if rmse < cfg.env.rmse_threshold:
            print(f"  >> RMSE threshold reached, s: {cfg.env.traj.w:.2f} → {cfg.env.traj.w + cfg.env.w_increase:.2f} 🔥")
            cfg.env.traj.w += cfg.env.w_increase

            if cfg.env.traj.w >= last_saved_w + SAVE_INTERVAL:
                torch.save(policy.state_dict(), os.path.join(out_dir, f"trck_{cfg.env.traj.w:.2f}.pt"))
                last_saved_w = cfg.env.traj.w


        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), os.path.join(out_dir, "trck_best.pt"))

    torch.save(policy.state_dict(), os.path.join(out_dir, "trck_final.pt"))
    torch.jit.script(policy.cpu()).save(os.path.join(out_dir, "trck_final_scripted.pt"))
    print(f"\nTotal training time: {time.time() - start:.1f}s")
    
    traj_np = traj_data.cpu().numpy()

    if not cfg.headless:
        # logger.plot(dt=cfg.dt)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(traj_np[:, 26:30])
        # plt.show()
        plot_rollout(
            traj_np        = traj_np,
            dt             = cfg.dt,
            label          = cfg.cm,
            arm_length = float(quadrotor.arm_length[0, 0].cpu()),
            arm_angle  = float(quadrotor.arm_angle[0].cpu()) * 180 / 3.14,
            mass       = float(quadrotor.m[0].cpu()),
            # save_path = f"/home/adame/torchAirBender/outputs/plots/{cm}_dashboard.png",  # uncomment to save instead of show
        )

        MultiDroneRenderer(trajectory=traj_np, ref_trajectory=traj_np[:, 17:20]).run()


        
# ==================================================================
#                    TESTING THE POLICIES
# ==================================================================

def test(cfg: DictConfig):
    policies = [
        {"cm": "ctbr", 
         "path": "/home/adame/torchAirBender/outputs/policies/TT/ctbr/trck_1.90.pt",  
         "color": (0.2, 0.6, 1.0)},
        # {"cm": "lvhr", 
        #  "path": "/home/adame/torchAirBender/outputs/policies/TT/lvhr/trck_1.80.pt",  
        #  "color": (0.2, 0.6, 1.0)},
    ]
    for p in policies:
        p["label"] = p["type"] + "_" + p["cm"] + "_" + Path(p["path"]).stem.split("_", 1)[-1]

    randomize = True
    seed      = None
    save      = False
    csv_path  = "/home/adame/torchAirBender/miscellaneous/trajectories/CAMP/harmonic.csv"

    if seed is not None:
        torch.manual_seed(seed)

    device    = cfg.device
    quadrotor = QuadrotorDynamics(cfg)
    traj      = TrajectoryManager.from_harmonics(cfg.env.traj, cfg.num_envs, device)

    ref_traj = None
    drones   = []
    params   = None

    for spec in policies:
        print(f"\n  Rolling out: {spec['label']}  ({spec['path']})")

        controller = build_controller(spec["cm"], quadrotor, cfg)

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

        # Randomize dynamics 
        if randomize:
            params = randomize_parameters(cfg.dynamics, cfg.num_envs, device)
            quadrotor.set_parameters(params)
            controller.update_params(
                alloc_matrix = quadrotor._alloc_matrix,
                J            = quadrotor.J,
            )

        # Reset state at trajectory start
        pos0, vel0, acc0, _ = traj.get_reference(0)
        states = torch.zeros((cfg.num_envs, 17), device=device)
        states[:, 0:3]  = pos0.detach()
        states[:, 3:6]  = vel0.detach()
        states[:, 6:10] = acc_to_quat(acc0.detach())

        traj_data = torch.empty((cfg.steps, CM_COLS[spec["cm"]]), device=device)

        # Rollout 
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

        # Optionally save reference CSV 
        if ref_traj is None and save:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            ref_data = traj_np[:, 17:26]  # [px,py,pz,vx,vy,vz,ax,ay,az]
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az"])
                for i, row in enumerate(ref_data):
                    writer.writerow([i * cfg.dt, *row.tolist()])
            print(f"  Saved reference CSV: {csv_path}")

        ref_traj = ref_traj if ref_traj is not None else traj_np[:, 17:20]

        drones.append({
            "traj":  traj_np,
            "color": spec.get("color", (0.2, 0.6, 1.0)),
            "label": spec["label"],
        })


        plot_rollout(
            traj_np    = traj_np,
            dt         = cfg.dt,
            label      = spec["label"],
            arm_length = float(params.arm_length[0, 0].cpu()),
            arm_angle  = float(params.arm_angle[0].cpu()),
            mass       = float(params.mass[0].cpu()),
        )

    MultiDroneRenderer(drones=drones, ref_trajectory=ref_traj).run()
