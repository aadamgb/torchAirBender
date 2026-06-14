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
    quat = torch.tensor(df[["q_w", "q_x", "q_y", "q_z"]].values[:steps],         dtype=torch.float32, device=device)
    vel = torch.tensor(df[["v_x", "v_y", "v_z"]].values[:steps],         dtype=torch.float32, device=device)
    omega = torch.tensor(df[["w_x", "w_y", "w_z"]].values[:steps],         dtype=torch.float32, device=device)
    acc_lin = torch.tensor(df[["a_lin_x", "a_lin_y", "a_lin_z"]].values[:steps], dtype=torch.float32, device=device)
    acc_rot = torch.tensor(df[["a_rot_x", "a_rot_y", "a_rot_z"]].values[:steps], dtype=torch.float32, device=device)
    thrust = torch.tensor(df[["u_1", "u_2", "u_3", "u_4"]].values[:steps],         dtype=torch.float32, device=device)
    jerk = torch.tensor(df[["jerk_x", "jerk_y", "jerk_z"]].values[:steps], dtype=torch.float32, device=device)
    snap = torch.tensor(df[["snap_x", "snap_y", "snap_z"]].values[:steps], dtype=torch.float32, device=device)

    dt = float(df["t"].iloc[1] - df["t"].iloc[0])

    return {"pos": pos, "quat": quat, "vel": vel, "omega": omega , 
            "acc_lin": acc_lin, "acc_rot": acc_rot, "thrust": thrust, 
            "jerk": jerk, "snap": snap, "dt": dt}


def load_LOL(path: str, steps = None, device=None, dt: float = 0.001) -> dict:
    """
    Loads a headerless LOL trajectory CSV and returns tensors ready for use in test().

    Format (no header): t, x, y, z, vx, vy, vz, ax, ay, az

    Args:
        dt : desired timestep. If larger than the native dt, rows are subsampled.
             e.g. native dt=0.001, pass dt=0.01 → every 10th row is kept.
    """
    canonical_cols = ["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
    aliases = {
        "t":  ["t", "time"],
        "x":  ["x", "px", "p_x"],
        "y":  ["y", "py", "p_y"],
        "z":  ["z", "pz", "p_z"],
        "vx": ["vx", "v_x"],
        "vy": ["vy", "v_y"],
        "vz": ["vz", "v_z"],
        "ax": ["ax", "a_x", "a_lin_x"],
        "ay": ["ay", "a_y", "a_lin_y"],
        "az": ["az", "a_z", "a_lin_z"],
    }

    # First try header-aware parsing (supports: time,px,py,pz,...).
    df_header = pd.read_csv(path)
    lower_to_original = {str(c).strip().lower(): c for c in df_header.columns}
    rename_map = {}
    for canonical, keys in aliases.items():
        for key in keys:
            if key in lower_to_original:
                rename_map[lower_to_original[key]] = canonical
                break

    if len(rename_map) == len(canonical_cols):
        df = df_header.rename(columns=rename_map)[canonical_cols]
    else:
        # Fallback for legacy headerless format.
        df = pd.read_csv(path, header=None, names=canonical_cols)

    # Force numeric types; invalid rows (e.g., accidental partial lines) are dropped.
    for col in canonical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=canonical_cols).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(f"Trajectory CSV must contain at least 2 valid numeric rows: {path}")

    dts = np.diff(df["t"].to_numpy(dtype=np.float64))
    positive_dts = dts[dts > 0]
    if len(positive_dts) == 0:
        raise ValueError(f"Trajectory CSV has non-increasing time values: {path}")
    native_dt = float(np.median(positive_dts))
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