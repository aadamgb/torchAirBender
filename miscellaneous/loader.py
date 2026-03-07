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



def load_trajectory(path: str, steps: int, device) -> dict:
    """
    Loads a trajectory CSV and returns tensors ready for use in test().

    Returns a dict with:
        pos : (steps, 3)
        vel : (steps, 3)
        acc : (steps, 3)
        dt  : float  — inferred from the t column
    """
    df = pd.read_csv(path)

    assert len(df) >= steps, f"Trajectory too short: {len(df)} rows < {steps} steps"

    pos = torch.tensor(df[["p_x", "p_y", "p_z"]].values[:steps],         dtype=torch.float32, device=device)
    vel = torch.tensor(df[["v_x", "v_y", "v_z"]].values[:steps],         dtype=torch.float32, device=device)
    acc = torch.tensor(df[["a_lin_x", "a_lin_y", "a_lin_z"]].values[:steps], dtype=torch.float32, device=device)

    dt = float(df["t"].iloc[1] - df["t"].iloc[0])

    return {"pos": pos, "vel": vel, "acc": acc, "dt": dt}