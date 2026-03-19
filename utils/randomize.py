import torch
from torch import Tensor
from typing import NamedTuple
from omegaconf import DictConfig


class QuadrotorParams(NamedTuple):
    """
    Per-environment physical parameters.
    Each field is a (num_envs,) tensor, except J and C_D which are (num_envs, 3).
    """
    mass: Tensor        # (N,)
    arm_length: Tensor  # (N,)
    arm_angle: Tensor  # (N,)
    km: Tensor           # (N, )
    J: Tensor           # (N, 3)
    # kf: Tensor          # (N,)
    # km: Tensor          # (N,)
    # motor_tau: Tensor   # (N,)
    # C_D: Tensor         # (N, 3)


def randomize_parameters(
    cfg: DictConfig,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None
) -> QuadrotorParams:
    """
    Args:
        cfg        : OmegaConf config
        num_envs   : number of parallel environments
        device     : torch device
        dtype      : float precision
        generator  : optional torch.Generator for reproducibility
    """

    # ------------------------------------------------------------------
    # 1. Scale factor c ~ Uniform(sf.min, sf.max)
    # ------------------------------------------------------------------
    c = torch.rand(
        num_envs, device=device, dtype=dtype, generator=generator
    ) * (cfg.sf.max - cfg.sf.min) + cfg.sf.min

    # ------------------------------------------------------------------
    # 2. Arm length (linear in c)
    # ------------------------------------------------------------------
    l_min, l_max = cfg.arm_length.min, cfg.arm_length.max
    l = c * (l_max - l_min) + l_min

    alpha = torch.full(
        (num_envs,), cfg.arm_angle.nominal, device=device, dtype=dtype   # THis is in degrees be carful!!!
    ) # alpha does not scale, only add noise

    # ------------------------------------------------------------------
    # 3. Mass (scales as l^3)
    # ------------------------------------------------------------------
    m_min, m_max = cfg.mass.min, cfg.mass.max
    mass = (
        (l**3 - l_min**3) / (l_max**3 - l_min**3)
    ) * (m_max - m_min) + m_min

    # ------------------------------------------------------------------
    # 4. Inertia (scales as l^5)
    # ------------------------------------------------------------------
    Ixx_min, Ixx_max = cfg.inertia.xx.min, cfg.inertia.xx.max
    Iyy_min, Iyy_max = cfg.inertia.yy.min, cfg.inertia.yy.max
    Izz_min, Izz_max = cfg.inertia.zz.min, cfg.inertia.zz.max

    scale_l5 = (l**5 - l_min**5) / (l_max**5 - l_min**5)
    Ixx = scale_l5 * (Ixx_max - Ixx_min) + Ixx_min
    Iyy = scale_l5 * (Iyy_max - Iyy_min) + Iyy_min
    Izz = scale_l5 * (Izz_max - Izz_min) + Izz_min
    J = torch.stack([Ixx, Iyy, Izz], dim=-1)

    # # ------------------------------------------------------------------
    # # 5. Drag coefficients (scales as l^2)
    # # ------------------------------------------------------------------
    # scale_l2 = (l**2 - l_min**2) / (l_max**2 - l_min**2)
    # CDx = scale_l2 * (cfg.C_D.x.max - cfg.C_D.x.min) + cfg.C_D.x.min
    # CDy = scale_l2 * (cfg.C_D.y.max - cfg.C_D.y.min) + cfg.C_D.y.min
    # CDz = scale_l2 * (cfg.C_D.z.max - cfg.C_D.z.min) + cfg.C_D.z.min
    # C_D = torch.stack([CDx, CDy, CDz], dim=-1)

    # # ------------------------------------------------------------------
    # # 6–8. Log-uniform parameters
    # # ------------------------------------------------------------------
    # kf = cfg.kf.min * (cfg.kf.max / cfg.kf.min) ** c
    km = cfg.km.min * (cfg.km.max / cfg.km.min) ** c
    # motor_tau = cfg.motor_tau.min * (cfg.motor_tau.max / cfg.motor_tau.min) ** c

    # ------------------------------------------------------------------
    # 9. Independent multiplicative noise
    # ------------------------------------------------------------------
    # def add_noise(x):
    #     noise = (
    #         torch.rand_like(x, generator=generator) * 2.0 - 1.0
    #     ) * cfg.nf
    #     return x * (1.0 + noise)
    # def add_noise(x):
    #     noise = (
    #         torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator) * 2.0 - 1.0
    #     ) * cfg.nf
    #     return x * (1.0 + noise)
    # def add_noise(x):
    #     return x * (1 + torch.empty(num_envs, device=device)
    #                 .uniform_(-cfg.nf, cfg.nf))
    def add_noise(x):
        noise = torch.empty_like(x).uniform_(-cfg.nf, cfg.nf)
        return x * (1 + noise)
    
    def add_noise_diag_vec(J):
        noise = torch.empty_like(J)
        noise[:,0].uniform_(-cfg.nf, cfg.nf)   # Jx
        noise[:,1].uniform_(-cfg.nf, cfg.nf)   # Jy
        noise[:,2].uniform_(-cfg.nf, cfg.nf)    # Jz usually larger
        return J * (1 + noise)

    mass = add_noise(mass)
    l = add_noise(l)    # careful with this one, it can be a bit problematic for srt
    alpha = add_noise(alpha)   
    J = add_noise_diag_vec(J)  
    # kf = add_noise(kf)
    km = add_noise(km)
    # motor_tau = add_noise(motor_tau)
    # C_D = add_noise(C_D)

    return QuadrotorParams(
        mass=mass,
        arm_length=l,
        arm_angle=alpha,
        # kf=kf,
        km=km,
        J=J,
        # motor_tau=motor_tau,
        # C_D=C_D,
    )