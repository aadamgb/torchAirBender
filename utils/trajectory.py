import torch
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig
import math

class TrajectoryManager:
    """
    Unified trajectory interface for loaded tracks (TOGT/LOL) and random harmonics.

    Usage:
        # Racing track
        traj = TrajectoryManager.from_togt(path, num_envs, device)

        # LOL trajectory
        traj = TrajectoryManager.from_lol(path, num_envs, device, dt=0.01)

        # Random harmonics
        traj = TrajectoryManager.from_harmonics(cfg, num_envs, device)

        # In training loop
        pos, vel, acc, b1d = traj.get_reference(t, speed_scale=0.8)
        traj.randomize()   # call at episode start for harmonics
    """

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device   = device
        self._mode    = None     # "togt", "lol", or "harmonics"

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
    def from_lol(
        cls,
        path: str,
        num_envs: int,
        device: str,
        steps: int | None = None,
        dt: float = 0.01,
    ) -> "TrajectoryManager":
        from miscellaneous.loader import load_LOL
        obj = cls(num_envs, device)
        obj._mode       = "lol"
        obj._trajectory = load_LOL(path, steps=steps, device=device, dt=dt)
        obj._length     = obj._trajectory["pos"].shape[0]
        return obj

    @classmethod
    def from_harmonics(cls, cfg: DictConfig, num_envs: int, device: str) -> "TrajectoryManager":
        obj = cls(num_envs, device)
        obj._mode    = "harmonics"
        obj._cfg     = cfg
        obj._params  = obj._generate_params()
        return obj

    # ── public API ────────────────────────────────────────────────────────

    def get_reference(self, t: int, speed_scale: float = 1.0):
            if self._mode in ("togt", "lol"):
                return self._get_togt(t, speed_scale)
            else:
                return self._get_harmonics(t)        

    def randomize(self):
        """
        Re-randomize trajectory params. Call at the start of each episode.
        No-op for TOGT since the track is fixed.
        """
        if self._mode == "harmonics":
            self._params = self._generate_params()

    @property
    def get_ref0(self) -> Tensor:
        """ Convenience: returns (N, 3) starting position. """
        pos, vel, acc, b1 = self.get_reference(0)
        return pos, vel, acc, b1

    # ── private ───────────────────────────────────────────────────────────

    def _get_togt(self, t: int, speed_scale: float):
        scaled_t = min(int(t * speed_scale), self._length - 1)

        pos = self._trajectory["pos"][scaled_t].unsqueeze(0).expand(self.num_envs, -1)
        vel = self._trajectory["vel"][scaled_t].unsqueeze(0).expand(self.num_envs, -1) * speed_scale
        acc = self._trajectory["acc"][scaled_t].unsqueeze(0).expand(self.num_envs, -1) * speed_scale ** 2
        b1d = self._compute_b1d(self._trajectory["vel"][scaled_t])

        return pos, vel, acc, b1d

    def _get_harmonics(self, t: int):
        t_sec        = t * self._cfg.dt
        pos, vel, acc = _eval_harmonics(t_sec, self._params)
        b1d          = self._compute_b1d(vel)    # pass full (N, 3), not vel[0]
        return pos, vel, acc, b1d
        
    def _compute_b1d(self, v: Tensor) -> Tensor:
        """v can be (3,) for shared trajectory or (N, 3) for per-env."""
        if v.dim() == 1:
            v_heading    = v.clone()
            v_heading[2] = 0.0
            return F.normalize(v_heading, dim=-1).unsqueeze(0).expand(self.num_envs, -1)
        else:
            v_heading       = v.clone()          # (N, 3)
            v_heading[:, 2] = 0.0
            return F.normalize(v_heading, dim=-1)  # (N, 3)

    def _generate_params(self) -> dict:
        cfg = self._cfg
        amps   = torch.rand((self.num_envs, 3, cfg.num_harmonics), device=self.device) * cfg.A   + 0.5
        freqs  = torch.rand((self.num_envs, 3, cfg.num_harmonics), device=self.device) * cfg.w   + 0.2
        phases = torch.rand((self.num_envs, 3, cfg.num_harmonics), device=self.device) * cfg.phi * torch.pi
        return {"amps": amps, "freqs": freqs, "phases": phases, "z_offset": cfg.z_offset}


# ── standalone harmonic evaluator (kept for backward compatibility) ────────

def generate_trajectory_params(num_envs: int, device, cfg: DictConfig) -> dict:
    amps   = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.A   + 0.5
    freqs  = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.w   + 0.2
    phases = torch.rand((num_envs, 3, cfg.num_harmonics), device=device) * cfg.phi * torch.pi
    return {"amps": amps, "freqs": freqs, "phases": phases, "z_offset": cfg.z_offset}


def _eval_harmonics(t: float, params: dict) -> tuple[Tensor, Tensor, Tensor]:
    amps, freqs, phases = params["amps"], params["freqs"], params["phases"]
    t_tensor = torch.full((amps.shape[0], 1, 1), t, device=amps.device, dtype=amps.dtype)
    angle    = freqs * t_tensor + phases
    pos      = torch.sum(amps * torch.sin(angle),             dim=2)
    vel      = torch.sum(amps * freqs * torch.cos(angle),     dim=2)
    acc      = torch.sum(-amps * freqs**2 * torch.sin(angle), dim=2)
    pos[:, 2] = pos[:, 2] + params["z_offset"]
    return pos, vel, acc


def get_target(t: float, params: dict) -> tuple[Tensor, Tensor, Tensor]:
    """ Backward compatible wrapper. """
    return _eval_harmonics(t, params)






#==============================================================================
# Hypertrochoid for testing purposes (probably delete in the future)
#==============================================================================

class HypotrochoidTrajectory:
    def __init__(self, num_envs, device, R=5.0, r=3.0, d=5.0, speed=1.0, dt=0.01):
        self.num_envs = num_envs
        self.device = device
        self.R = R
        self.r = r
        self.d = d
        self.speed = speed # Controls how fast the drone moves along the path
        self.dt = dt
        
        # Precompute the constant term
        self.k = (R - r) / r

    def get_reference(self, t_step):
        # Convert time step to theta (angle)
        theta = t_step * self.dt * self.speed
        
        # Position (x, y) - z is fixed at a height (e.g., 2.0)
        # Using the formulas from the image
        x = (self.R - self.r) * torch.cos(torch.tensor(theta)) + \
            self.d * torch.cos(torch.tensor(self.k * theta))
        y = (self.R - self.r) * torch.sin(torch.tensor(theta)) - \
            self.d * torch.sin(torch.tensor(self.k * theta))
        
        pos = torch.zeros((self.num_envs, 3), device=self.device)
        pos[:, 0] = x
        pos[:, 1] = y
        pos[:, 2] = 2.0 # Fixed flight altitude

        # Velocity (First derivative w.r.t time)
        # dx/dt = dtheta/dt * [- (R-r) sin(theta) - d * k * sin(k*theta)]
        vel_scale = self.speed
        vx = -((self.R - self.r) * torch.sin(torch.tensor(theta)) + \
               self.d * self.k * torch.sin(torch.tensor(self.k * theta))) * vel_scale
        vy = ((self.R - self.r) * torch.cos(torch.tensor(theta)) - \
              self.d * self.k * torch.cos(torch.tensor(self.k * theta))) * vel_scale
        
        vel = torch.zeros((self.num_envs, 3), device=self.device)
        vel[:, 0] = vx
        vel[:, 1] = vy

        # Acceleration (Second derivative w.r.t time)
        acc_scale = self.speed**2
        ax = -((self.R - self.r) * torch.cos(torch.tensor(theta)) + \
               self.d * (self.k**2) * torch.cos(torch.tensor(self.k * theta))) * acc_scale
        ay = -((self.R - self.r) * torch.sin(torch.tensor(theta)) - \
               self.d * (self.k**2) * torch.sin(torch.tensor(self.k * theta))) * acc_scale
        
        acc = torch.zeros((self.num_envs, 3), device=self.device)
        acc[:, 0] = ax
        acc[:, 1] = ay

        # Jumps (not used by most controllers, but kept for interface compatibility)
        jerk = torch.zeros((self.num_envs, 3), device=self.device)

        return pos, vel, acc, jerk
    



class CircularTrajectory:
    """Simple circular trajectory in the XY plane."""
    def __init__(self, num_envs, device, radius=2.0, speed=1.0, height=1.5, dt=0.01):
        self.num_envs = num_envs
        self.device   = device
        self.radius   = radius
        self.speed    = speed      # m/s along the circle
        self.height   = height     # fixed Z
        self.dt       = dt
        self.omega    = speed / radius  # angular velocity (rad/s)

    def get_reference(self, t: int):
        angle = self.omega * t * self.dt

        # Position
        px = self.radius * math.cos(angle)
        py = self.radius * math.sin(angle)
        pz = self.height
        pos = torch.tensor([[px, py, pz]], device=self.device).expand(self.num_envs, -1)

        # Velocity (analytical derivative)
        vx = -self.radius * self.omega * math.sin(angle)
        vy =  self.radius * self.omega * math.cos(angle)
        vz = 0.0
        vel = torch.tensor([[vx, vy, vz]], device=self.device).expand(self.num_envs, -1)

        # Acceleration (analytical second derivative)
        ax = -self.radius * self.omega**2 * math.cos(angle)
        ay = -self.radius * self.omega**2 * math.sin(angle)
        az = 0.0
        acc = torch.tensor([[ax, ay, az]], device=self.device).expand(self.num_envs, -1)

        # Jerk (zero for constant-speed circle, but keep the 4-tuple interface)
        jerk = torch.zeros((self.num_envs, 3), device=self.device)

        return pos, vel, acc, jerk