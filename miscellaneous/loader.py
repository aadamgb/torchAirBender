import yaml
import numpy as np
import torch
import pandas as pd                                     # for trajectory loading

def load_gates_from_yaml(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads gate positions and RPY angles from a YAML file.
    
    Returns:
        gates_position : (N, 3) float32 ENU positions
        gates_rpy      : (N, 3) float32 Roll/Pitch/Yaw in degrees
    """
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    gates = data["gates"]
    positions = np.array([g["position"] for g in gates], dtype=np.float32)
    rpys      = np.array([g["rpy"]      for g in gates], dtype=np.float32)
    return positions, rpys



def load_TOGT(path: str, steps = None, device=None) -> dict:
    """
    Loads a trajectory CSV and returns tensors ready for use in test().

    Returns a dict with:
        pos : (steps, 3)
        vel : (steps, 3)
        acc : (steps, 3)
        dt  : float  — inferred from the t column
    """
    df = pd.read_csv(path)

    if steps is None:
        steps = len(df)
        
    assert len(df) >= steps, f"Trajectory too short: {len(df)} rows < {steps} steps"

    pos = torch.tensor(df[["p_x", "p_y", "p_z"]].values[:steps],         dtype=torch.float32, device=device)
    vel = torch.tensor(df[["v_x", "v_y", "v_z"]].values[:steps],         dtype=torch.float32, device=device)
    acc = torch.tensor(df[["a_lin_x", "a_lin_y", "a_lin_z"]].values[:steps], dtype=torch.float32, device=device)

    dt = float(df["t"].iloc[1] - df["t"].iloc[0])

    return {"pos": pos, "vel": vel, "acc": acc, "dt": dt}


def load_LOL(path: str, steps = None, device=None, dt: float = 0.001) -> dict:
    """
    Loads a headerless LOL trajectory CSV and returns tensors ready for use in test().

    Format (no header): t, x, y, z, vx, vy, vz, ax, ay, az

    Args:
        dt : desired timestep. If larger than the native dt, rows are subsampled.
             e.g. native dt=0.001, pass dt=0.01 → every 10th row is kept.
    """
    df = pd.read_csv(path, header=None, names=["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"])

    native_dt = float(df["t"].iloc[1] - df["t"].iloc[0])
    stride    = max(1, round(dt / native_dt))
    df        = df.iloc[::stride].reset_index(drop=True)

    if steps is None:
        steps = len(df)

    assert len(df) >= steps, f"Trajectory too short after subsampling: {len(df)} rows < {steps} steps"

    pos = torch.tensor(df[["x",  "y",  "z" ]].values[:steps], dtype=torch.float32, device=device)
    vel = torch.tensor(df[["vx", "vy", "vz"]].values[:steps], dtype=torch.float32, device=device)
    acc = torch.tensor(df[["ax", "ay", "az"]].values[:steps], dtype=torch.float32, device=device)

    actual_dt = native_dt * stride
    print(f"[load_LOL] native_dt={native_dt:.4f}s  stride={stride}  → dt={actual_dt:.4f}s  rows={len(df)}")

    return {"pos": pos, "vel": vel, "acc": acc, "dt": actual_dt}