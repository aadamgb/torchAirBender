# controller/srt_controller.py
# SRT = Square Root Throttle, or whatever your PMM maps to —
# just a direct [0,1] sigmoid output scaled to [0, max_thrust]

import torch
from torch import Tensor
from omegaconf import DictConfig
from controller.base_controller import BaseController


class SRTController(BaseController):
    """
    Minimal controller for the point-mass model.
    Maps sigmoid policy output -> per-motor thrust [N].

    Future extensions:
        - Motor lag / first-order dynamics
        - Drag model
        - RPM -> thrust curve (nonlinear)
    """

    def __init__(self, cfg: DictConfig):
        self.max_thrust = cfg.env.max_thrust   # e.g. 2.0

    def __call__(self, raw: Tensor) -> Tensor:
        # raw: (N, 4) in (0, 1) from Sigmoid
        return raw * self.max_thrust           # (N, 4) in (0, max_thrust)