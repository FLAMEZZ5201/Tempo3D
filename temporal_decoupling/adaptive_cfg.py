"""Adaptive CFG scheduling strategies for diffusion inference."""

import torch
import numpy as np


def adaptive_cfg_schedule(t: float) -> float:
    """Piecewise-constant CFG: low early, standard mid, linearly rising late."""
    if t < 0.3:
        return 5.0
    if t < 0.7:
        return 15.0
    return 25.0 + (t - 0.7) / 0.3 * 10.0


def smooth_adaptive_cfg_schedule(t: float) -> float:
    """Sigmoid-smoothed CFG schedule."""
    cfg_early_to_mid = 5.0 + 10.0 / (1 + np.exp(-20 * (t - 0.3)))
    if t < 0.7:
        return cfg_early_to_mid
    cfg_late = 25.0 + (t - 0.7) / 0.3 * 10.0
    return 15.0 + (cfg_late - 15.0) / (1 + np.exp(-15 * (t - 0.7)))


def frequency_selective_cfg(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    t: float,
    cfg_low: float = 15.0,
    cfg_high: float = 35.0,
) -> torch.Tensor:
    """Apply separate CFG weights to low- and high-frequency components."""
    from temporal_decoupling.spectral_loss import create_low_freq_mask

    B, H, W, D, C = v_cond.shape
    low_mask  = create_low_freq_mask(H, W, D, cutoff=0.25, device=v_cond.device)
    high_mask = 1 - low_mask
    cfg_high_t = cfg_high * ((t - 0.7) / 0.3 + 1.0) if t > 0.7 else cfg_low

    v_final = torch.zeros_like(v_cond)
    for c in range(C):
        cond_fft = torch.fft.fftshift(torch.fft.fftn(v_cond[:, :, :, :, c],   dim=(-3, -2, -1)), dim=(-3, -2, -1))
        unc_fft  = torch.fft.fftshift(torch.fft.fftn(v_uncond[:, :, :, :, c], dim=(-3, -2, -1)), dim=(-3, -2, -1))
        diff = cond_fft - unc_fft
        merged = unc_fft + cfg_low * diff * low_mask + cfg_high_t * diff * high_mask
        v_final[:, :, :, :, c] = torch.fft.ifftn(
            torch.fft.ifftshift(merged, dim=(-3, -2, -1)), dim=(-3, -2, -1)
        ).real
    return v_final


def apply_cfg(v_cond: torch.Tensor, v_uncond: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    return v_uncond + cfg_scale * (v_cond - v_uncond)


class AdaptiveCFGScheduler:
    """Dynamic CFG weight scheduler.

    Args:
        schedule_type: ``'fixed'`` | ``'adaptive'`` | ``'smooth'``
        cfg_min: CFG scale for early timesteps (t < t_early).
        cfg_mid: CFG scale for mid timesteps.
        cfg_max: Maximum CFG scale reached at t=1 (adaptive mode).
        t_early: Boundary between early and mid phases.
        t_late:  Boundary between mid and late phases.
        use_frequency_selective: If True, apply per-band CFG instead of scalar CFG.
    """

    def __init__(
        self,
        schedule_type: str = 'adaptive',
        cfg_min: float = 5.0,
        cfg_mid: float = 15.0,
        cfg_max: float = 35.0,
        t_early: float = 0.3,
        t_late: float = 0.7,
        use_frequency_selective: bool = False,
    ):
        self.schedule_type = schedule_type
        self.cfg_min = cfg_min
        self.cfg_mid = cfg_mid
        self.cfg_max = cfg_max
        self.t_early = t_early
        self.t_late = t_late
        self.use_frequency_selective = use_frequency_selective

    def get_cfg_scale(self, t: float) -> float:
        if self.schedule_type == 'fixed':
            return self.cfg_mid
        if self.schedule_type == 'adaptive':
            if t < self.t_early:
                return self.cfg_min
            if t < self.t_late:
                return self.cfg_mid
            alpha = (t - self.t_late) / (1.0 - self.t_late)
            return self.cfg_mid + alpha * (self.cfg_max - self.cfg_mid)
        if self.schedule_type == 'smooth':
            return smooth_adaptive_cfg_schedule(t)
        raise ValueError(f"Unknown schedule_type: {self.schedule_type!r}")

    def apply(self, v_cond: torch.Tensor, v_uncond: torch.Tensor, t: float) -> torch.Tensor:
        if self.use_frequency_selective and self.schedule_type != 'fixed':
            return frequency_selective_cfg(v_cond, v_uncond, t, self.cfg_mid, self.cfg_max)
        return apply_cfg(v_cond, v_uncond, self.get_cfg_scale(t))

    def __repr__(self):
        return (
            f"AdaptiveCFGScheduler(type={self.schedule_type!r}, "
            f"cfg=[{self.cfg_min}, {self.cfg_mid}, {self.cfg_max}], "
            f"freq_selective={self.use_frequency_selective})"
        )