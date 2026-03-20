import torch
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig


class TrajectoryManager:
    """
    Unified trajectory interface for TOGT racing tracks, random harmonics,
    and hypotrochoid curves.

    Usage:
        # Racing track
        traj = TrajectoryManager.from_togt(path, num_envs, device)

        # Random harmonics
        traj = TrajectoryManager.from_harmonics(cfg, num_envs, device)

        # Hypotrochoid
        traj = TrajectoryManager.from_hypotrochoid(cfg, num_envs, device)

        # In training loop
        pos, vel, acc, b1d = traj.get_reference(t, speed_scale=0.8)
        traj.randomize()   # call at episode start for stochastic modes
    """

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device   = device
        self._mode    = None     # "togt" | "harmonics" | "hypotrochoid"

    # ── constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_togt(cls, path: str, num_envs: int, device: str) -> "TrajectoryManager":
        from miscellaneous.loader import load_TOGT
        obj = cls(num_envs, device)
        obj._mode       = "togt"
        obj._trajectory = load_TOGT(path, device=device)
        obj._length     = obj._trajectory["pos"].shape[0]
        return obj

    @classmethod
    def from_harmonics(cls, cfg: DictConfig, num_envs: int, device: str) -> "TrajectoryManager":
        obj = cls(num_envs, device)
        obj._mode    = "harmonics"
        obj._cfg     = cfg
        obj._params  = obj._generate_harmonic_params()
        return obj

    @classmethod
    def from_hypotrochoid(cls, cfg: DictConfig, num_envs: int, device: str) -> "TrajectoryManager":
        """
        Hypotrochoid trajectories. Each environment gets its own (R, r, d, omega, phi)
        sampled from the ranges in cfg. Call traj.randomize() at episode start to
        re-sample all per-env parameters.

        Required cfg fields:
            R_range:         [1.0, 6.0]
            r_range:         [0.5, 4.0]   # enforced < R
            d_range:         [0.5, 5.0]
            omega_range:     [0.3, 1.2]   # rad/s
            z_offset:        1.5          # metres AGL
            dt:              0.02
            xy_offset_range: [0.0, 2.0]   # optional XY center jitter
        """
        obj = cls(num_envs, device)
        obj._mode    = "hypotrochoid"
        obj._cfg     = cfg
        obj._params  = obj._generate_hypotrochoid_params()
        return obj

    # ── public API ────────────────────────────────────────────────────────

    def get_reference(self, t: int, speed_scale: float = 1.0):
        if self._mode == "togt":
            return self._get_togt(t, speed_scale)
        elif self._mode == "harmonics":
            return self._get_harmonics(t)
        else:
            return self._get_hypotrochoid(t, speed_scale)

    def randomize(self):
        """
        Re-randomize trajectory params. Call at the start of each episode.
        No-op for TOGT since the track is fixed.
        """
        if self._mode == "harmonics":
            self._params = self._generate_harmonic_params()
        elif self._mode == "hypotrochoid":
            self._params = self._generate_hypotrochoid_params()

    @property
    def get_ref0(self) -> Tensor:
        """Convenience: returns (N, 3) starting position."""
        pos, vel, acc, b1 = self.get_reference(0)
        return pos, vel, acc, b1

    # ── private: dispatch ─────────────────────────────────────────────────

    def _get_togt(self, t: int, speed_scale: float):
        scaled_t = min(int(t * speed_scale), self._length - 1)
        pos = self._trajectory["pos"][scaled_t].unsqueeze(0).expand(self.num_envs, -1)
        vel = self._trajectory["vel"][scaled_t].unsqueeze(0).expand(self.num_envs, -1) * speed_scale
        acc = self._trajectory["acc"][scaled_t].unsqueeze(0).expand(self.num_envs, -1) * speed_scale ** 2
        b1d = self._compute_b1d(self._trajectory["vel"][scaled_t])
        return pos, vel, acc, b1d

    def _get_harmonics(self, t: int):
        t_sec         = t * self._cfg.dt
        pos, vel, acc = _eval_harmonics(t_sec, self._params)
        b1d           = self._compute_b1d(vel)
        return pos, vel, acc, b1d

    def _get_hypotrochoid(self, t: int, speed_scale: float):
        # speed_scale multiplies omega, which correctly scales vel by s and acc by s²
        scaled_params          = dict(self._params)
        scaled_params["omega"] = self._params["omega"] * speed_scale
        t_sec                  = t * self._cfg.dt
        pos, vel, acc          = _eval_hypotrochoid(t_sec, scaled_params)
        b1d                    = self._compute_b1d(vel)
        return pos, vel, acc, b1d

    # ── private: param generation ─────────────────────────────────────────

    def _generate_harmonic_params(self) -> dict:
        cfg    = self._cfg
        amps   = torch.rand((self.num_envs, 3, cfg.num_harmonics), device=self.device) * cfg.A   + 0.5
        freqs  = torch.rand((self.num_envs, 3, cfg.num_harmonics), device=self.device) * cfg.w   + 0.2
        phases = torch.rand((self.num_envs, 3, cfg.num_harmonics), device=self.device) * cfg.phi * torch.pi
        return {"amps": amps, "freqs": freqs, "phases": phases, "z_offset": cfg.z_offset}

    def _generate_hypotrochoid_params(self) -> dict:
        cfg = self._cfg
        N   = self.num_envs
        dev = self.device

        def _uniform(lo, hi):
            return torch.rand(N, device=dev) * (hi - lo) + lo

        R = _uniform(*cfg.R_range)
        # Enforce r < R to stay in the hypotrochoid regime (r >= R gives an epicycloid)
        r = torch.rand(N, device=dev) * (R - 0.1 - cfg.r_range[0]) + cfg.r_range[0]
        r_min = torch.full((N,), cfg.r_range[0], device=dev)
        r_max = R - 0.1
        r = torch.clamp(r, r_min, r_max)

        d     = _uniform(*cfg.d_range)
        omega = _uniform(*cfg.omega_range)
        phi   = torch.rand(N, device=dev) * 2 * torch.pi

        xy_range = getattr(cfg, "xy_offset_range", [0.0, 0.0])
        signs    = 2 * torch.randint(0, 2, (N,), device=dev).float() - 1
        cx       = _uniform(*xy_range) * signs
        cy       = _uniform(*xy_range) * signs.roll(1)

        return {"R": R, "r": r, "d": d, "omega": omega, "phi": phi,
                "cx": cx, "cy": cy, "z_offset": cfg.z_offset}

    # ── private: shared utils ─────────────────────────────────────────────

    def _compute_b1d(self, v: Tensor) -> Tensor:
        """v can be (3,) for shared trajectory or (N, 3) for per-env."""
        if v.dim() == 1:
            v_heading    = v.clone()
            v_heading[2] = 0.0
            return F.normalize(v_heading, dim=-1).unsqueeze(0).expand(self.num_envs, -1)
        else:
            v_heading       = v.clone()
            v_heading[:, 2] = 0.0
            return F.normalize(v_heading, dim=-1)


# ── module-level evaluators ───────────────────────────────────────────────────

def _eval_hypotrochoid(t: float, params: dict) -> tuple[Tensor, Tensor, Tensor]:
    """
    Evaluate per-environment hypotrochoid positions, velocities, and accelerations.

    All params tensors are (N,). Returns (N, 3) tensors.
    """
    R, r, d    = params["R"], params["r"], params["d"]
    omega, phi = params["omega"], params["phi"]
    cx, cy     = params["cx"], params["cy"]
    k          = R - r
    ratio      = k / r
    theta      = omega * t + phi

    x  =  k * torch.cos(theta)           + d * torch.cos(ratio * theta) + cx
    y  =  k * torch.sin(theta)           - d * torch.sin(ratio * theta) + cy
    z  = torch.full_like(x, params["z_offset"])

    vx = omega * (-k * torch.sin(theta)  - d * ratio * torch.sin(ratio * theta))
    vy = omega * ( k * torch.cos(theta)  - d * ratio * torch.cos(ratio * theta))
    vz = torch.zeros_like(vx)

    ax = omega**2 * (-k * torch.cos(theta) - d * ratio**2 * torch.cos(ratio * theta))
    ay = omega**2 * (-k * torch.sin(theta) + d * ratio**2 * torch.sin(ratio * theta))
    az = torch.zeros_like(ax)

    pos = torch.stack([x, y, z], dim=1)
    vel = torch.stack([vx, vy, vz], dim=1)
    acc = torch.stack([ax, ay, az], dim=1)
    return pos, vel, acc


def _eval_harmonics(t: float, params: dict) -> tuple[Tensor, Tensor, Tensor]:
    amps, freqs, phases = params["amps"], params["freqs"], params["phases"]
    t_tensor = torch.full((amps.shape[0], 1, 1), t, device=amps.device, dtype=amps.dtype)
    angle    = freqs * t_tensor + phases
    pos      = torch.sum(amps * torch.sin(angle),              dim=2)
    vel      = torch.sum(amps * freqs * torch.cos(angle),      dim=2)
    acc      = torch.sum(-amps * freqs**2 * torch.sin(angle),  dim=2)
    pos[:, 2] = pos[:, 2] + params["z_offset"]
    return pos, vel, acc


# ── backward-compatible wrappers ──────────────────────────────────────────────

def generate_trajectory_params(num_envs: int, device, cfg: DictConfig) -> dict:
    amps   = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.A   + 0.5
    freqs  = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.w   + 0.2
    phases = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.phi * torch.pi
    return {"amps": amps, "freqs": freqs, "phases": phases, "z_offset": cfg.z_offset}


def get_target(t: float, params: dict) -> tuple[Tensor, Tensor, Tensor]:
    return _eval_harmonics(t, params)