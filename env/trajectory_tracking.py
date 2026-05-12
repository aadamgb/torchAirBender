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
from utils.trajectory import TrajectoryManager, HypotrochoidTrajectory, CircularTrajectory
from utils.math import acc_to_quat, quat_to_rotmat
from utils.plotter import plot_rollout

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from controller.controllers import DirectAllocation, SRT, CTBR, LVHR


ACT_DIMS = {
    "srt":    4,
    "ctbr":   4,
    "lvhr":   4,
    "lvhr+g": 7,
}
CM_COLS = {"srt": 30, "ctbr": 34, "lvhr": 38, "lvhr+g": 41}

def reset(cfg, traj, quadrotor, controller):
    pos0, vel0, acc0, _ = traj.get_reference(0)
    states = torch.zeros((cfg.num_envs, 17), device=cfg.device)
    states[:, 0:3]  = pos0.detach()
    states[:, 3:6]  = vel0.detach()
    states[:, 6:10] = acc_to_quat(acc0.detach())

    params = randomize_parameters(cfg.dynamics, cfg.num_envs, cfg.device)
    quadrotor.set_parameters(params)
    controller.update_params(
        alloc_matrix = quadrotor._alloc_matrix,
        mass         = quadrotor.m,
        max_TWR      = quadrotor.max_TWR,
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
    states[idx, 13:17] = 0.0
    return states


def get_observation(states, pos_ref, vel_ref, acc_ref):

    return torch.cat([
        pos_ref - states[:, 0:3],   # position error  (N, 3)
        vel_ref - states[:, 3:6],   # velocity error  (N, 3)
        acc_ref,                    # reference acc   (N, 3)
        # states[:, 6:10],            # quaternion      (N, 4)
        quat_to_rotmat(states[:, 6:10]).reshape(states.shape[0], 9),            # quaternion      (N, 4)
        states[:, 10:13],           # body rates      (N, 3)
    ], dim=-1)                      # (N, 16)


def compute_loss(states, pos_ref, vel_ref, acc_ref, weights, mask=None):
    if mask is not None and not mask.any():
        return states.sum() * 0.0

    s, p, v, a = (x[mask] if mask is not None else x
                  for x in (states, pos_ref, vel_ref, acc_ref))

    pos_loss  = torch.linalg.norm(p - s[:, 0:3], dim=-1).mean()
    # pos_loss  =  torch.sum((p - s[:, 0:3])**2, dim=-1).mean()
    vel_loss  = torch.linalg.norm(v - s[:, 3:6], dim=-1).mean()
    # vel_loss  =  torch.sum((v - s[:, 3:6])**2, dim=-1).mean()
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

    elif cm_type == "lvyr-indi":
        raise NotImplementedError("INDI inner loop not yet implemented")
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

    export_dir = f"/home/adame/torchAirBender/outputs/policies/sample_eff/{cm}/exported_data"
    os.makedirs(export_dir, exist_ok=True)
    csv_path = os.path.join(export_dir, f"metrics_{cm}.csv")

    quadrotor  = QuadrotorDynamics(cfg)
    traj      = TrajectoryManager.from_harmonics(cfg.env.traj, num_envs, device)

    # path = "/home/adame/torchAirBender/miscellaneous/trajectories/TOGT/togt_traj.csv"
    # traj = TrajectoryManager.from_togt(path, cfg.num_envs, cfg.device)

    controller = build_controller(cm, quadrotor, cfg)
    
    policy    = MLP(layer_sizes=list(cfg.env.policy) + [ACT_DIMS[cm]], 
                    # activation=nn.Tanh,                           
                    activation=nn.ReLU,                             
                    output_activation=nn.Sigmoid(), 
                    output_bias_init=0.0
                    ).to(device)
    
    # ⚠️Load policy⚠️
    # policy.load_state_dict(torch.load("/home/adame/torchAirBender/outputs/policies/CAMP/uzh/ctbr/uzh_starter.pt", map_location=device))
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.env.lr)
    # optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.env.lr)

    traj_data = torch.empty((cfg.steps, CM_COLS[cm]), device=device)
    last_saved_w = 1.0
    SAVE_INTERVAL = 0.1
    best_loss    = float("inf")
    s = 0.3

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Step", "Loss"])

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

                pos_ref, vel_ref, acc_ref, _ = traj.get_reference(t, speed_scale=s)

                obs     = get_observation(states, pos_ref, vel_ref, acc_ref)
                raw     = policy(obs)

                if cm == "lvhr+g":
                    actions = controller(states, raw[:, 0:4], gains=raw[:, 4:7])
                else:
                    actions = controller(states, raw)

                states  = quadrotor.step(states, actions[:, 0:4])            # Forward pass through the dynamics

                dist    = torch.linalg.norm(pos_ref - states[:, 0:3], dim=-1)
                too_far = dist > cfg.env.max_dist_to_target

                sq_error_sum += (dist ** 2).sum()
                num_samples  += dist.numel()

                window_loss += compute_loss(states, pos_ref, vel_ref, acc_ref, cfg.env.loss_weights, mask=~too_far)

                if ep == cfg.episodes - 1:
                    traj_data[t] = torch.cat([
                        states[0].detach(),       # 0:17  — full state
                        pos_ref[0].detach(),      # 17:20 — p_ref
                        vel_ref[0].detach(),      # 20:23 — v_ref
                        acc_ref[0].detach(),      # 23:26 — a_ref
                        actions[0].detach(),      # 26:N  — srt + (wrench + lvyr + gains)
                    ], dim=0)

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

            # wall_time = time.time() - start
            total_steps = (ep + 1) * cfg.steps * num_envs
            csv_writer.writerow([total_steps, float(avg_loss)])
            csv_file.flush()

            if rmse < cfg.env.rmse_threshold:
                print(f"  >> RMSE threshold reached, s: {cfg.env.traj.w:.2f} → {cfg.env.traj.w + cfg.env.w_increase:.2f} 🔥")
                # print(f"  >> RMSE threshold reached, s: {s:.2f} → {s + cfg.env.w_increase:.2f} 🔥")
                cfg.env.traj.w += cfg.env.w_increase
                # s += cfg.env.w_increase

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
    # ---- Plot ------
    plot_rollout(
        traj_np        = traj_np,
        dt             = cfg.dt,
        label          = cfg.cm,
        arm_length = float(quadrotor.arm_length[0, 0].cpu()),
        arm_angle  = float(quadrotor.arm_angle[0].cpu()) * 180 / 3.14,
        mass       = float(quadrotor.m[0].cpu()),
        # save_path = f"/home/adame/torchAirBender/outputs/plots/{cm}_dashboard.png",  # uncomment to save instead of show
    )
    # ---- Render ------
    MultiDroneRenderer(trajectory=traj_np, ref_trajectory=traj_np[:, 17:20]).run()


# ==================================================================
#                    TESTING THE POLICIES
# ==================================================================

def test(cfg: DictConfig):
    policies = [
        {"type": "bptt",
         "cm": "srt", 
         "path": "/home/adame/torchAirBender/outputs/policies/TT/srt/trck_1.80.pt",  
         "color": (0.2, 0.6, 1.0)},
        {"type": "bptt",
         "cm": "ctbr", 
         "path": "/home/adame/torchAirBender/outputs/policies/TT/ctbr/trck_1.90.pt",  
         "color": (0.2, 0.6, 1.0)},
        {"type": "bptt",
         "cm": "lvhr", 
         "path": "/home/adame/torchAirBender/outputs/policies/TT/lvhr/trck_1.80.pt",  
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

        # ── Plot dashboard ───────────────────────────────────────────────
        plot_rollout(
            traj_np    = traj_np,
            dt         = cfg.dt,
            label      = spec["label"],
            arm_length = float(params.arm_length[0, 0].cpu()),
            arm_angle  = float(params.arm_angle[0].cpu()),
            mass       = float(params.mass[0].cpu()),
        )

    # ── Render all drones together ───────────────────────────────────────
    MultiDroneRenderer(drones=drones, ref_trajectory=ref_traj).run()
