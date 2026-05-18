"""Spectral frequency utilities for 3D latent analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def create_high_freq_mask(
    H: int, W: int, D: int, cutoff: float = 0.75, device=torch.device('cpu')
) -> torch.Tensor:
    fh, fw, fd = torch.meshgrid(
        torch.fft.fftfreq(H, device=device),
        torch.fft.fftfreq(W, device=device),
        torch.fft.fftfreq(D, device=device),
        indexing='ij',
    )
    return (torch.sqrt(fh**2 + fw**2 + fd**2) > cutoff * 0.5).float()


def create_low_freq_mask(
    H: int, W: int, D: int, cutoff: float = 0.25, device=torch.device('cpu')
) -> torch.Tensor:
    fh, fw, fd = torch.meshgrid(
        torch.fft.fftfreq(H, device=device),
        torch.fft.fftfreq(W, device=device),
        torch.fft.fftfreq(D, device=device),
        indexing='ij',
    )
    return (torch.sqrt(fh**2 + fw**2 + fd**2) <= cutoff * 0.5).float()


def _extract_high_freq_seq(v: torch.Tensor, cutoff: float) -> torch.Tensor:
    B, N, C = v.shape
    mask = (torch.abs(torch.fft.fftfreq(N, device=v.device)) > cutoff * 0.5).float()
    v_high = torch.zeros_like(v)
    for c in range(C):
        v_high[:, :, c] = torch.fft.ifft(torch.fft.fft(v[:, :, c], dim=1) * mask, dim=1).real
    return v_high


def _extract_high_freq_3d(v: torch.Tensor, cutoff: float) -> torch.Tensor:
    B, H, W, D, C = v.shape
    mask = create_high_freq_mask(H, W, D, cutoff, device=v.device)
    v_high = torch.zeros_like(v)
    for c in range(C):
        fft = torch.fft.fftshift(
            torch.fft.fftn(v[:, :, :, :, c], dim=(-3, -2, -1)), dim=(-3, -2, -1)
        )
        v_high[:, :, :, :, c] = torch.fft.ifftn(
            torch.fft.ifftshift(fft * mask, dim=(-3, -2, -1)), dim=(-3, -2, -1)
        ).real
    return v_high


def extract_high_freq_component(v: torch.Tensor, cutoff: float = 0.75) -> torch.Tensor:
    """Extract high-frequency components from a velocity field (seq or 3-D voxel format)."""
    if v.ndim == 3:
        return _extract_high_freq_seq(v, cutoff)
    if v.ndim == 5:
        return _extract_high_freq_3d(v, cutoff)
    raise ValueError(f"Unsupported tensor rank: {v.ndim}")


class DetailAwareLoss(nn.Module):
    """Time-frequency adaptive loss.

    Applies plain MSE for t <= t_threshold; adds a high-frequency penalty for t > t_threshold.
    """

    def __init__(
        self,
        t_threshold: float = 0.6,
        high_freq_cutoff: float = 0.75,
        high_freq_weight: float = 10.0,
    ):
        super().__init__()
        self.t_threshold = t_threshold
        self.high_freq_cutoff = high_freq_cutoff
        self.high_freq_weight = high_freq_weight

    def forward(
        self, v_pred: torch.Tensor, v_target: torch.Tensor, t: float
    ) -> Tuple[torch.Tensor, dict]:
        L_base = F.mse_loss(v_pred, v_target)
        if t > self.t_threshold:
            L_hf = F.mse_loss(
                extract_high_freq_component(v_pred, self.high_freq_cutoff),
                extract_high_freq_component(v_target, self.high_freq_cutoff),
            )
            lam = (t - self.t_threshold) / (1.0 - self.t_threshold)
            loss = L_base + lam * self.high_freq_weight * L_hf
            return loss, {
                'loss_base': L_base.item(), 'loss_hf': L_hf.item(),
                'lambda': lam, 'loss_total': loss.item(), 't': t,
            }
        return L_base, {'loss_base': L_base.item(), 'loss_total': L_base.item(), 't': t}


class FrequencySelectiveLoss(nn.Module):
    """Per-band weighted loss with time-adaptive high-frequency penalty."""

    def __init__(
        self,
        low_freq_cutoff: float = 0.25,
        high_freq_cutoff: float = 0.75,
        base_weight: float = 1.0,
        high_freq_weight: float = 10.0,
    ):
        super().__init__()
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_cutoff = high_freq_cutoff
        self.base_weight = base_weight
        self.high_freq_weight = high_freq_weight

    def forward(
        self, v_pred: torch.Tensor, v_target: torch.Tensor, t: float
    ) -> Tuple[torch.Tensor, dict]:
        B, H, W, D, C = v_pred.shape
        low_mask  = create_low_freq_mask(H, W, D, self.low_freq_cutoff, v_pred.device)
        high_mask = create_high_freq_mask(H, W, D, self.high_freq_cutoff, v_pred.device)
        mid_mask  = 1 - low_mask - high_mask
        w_high = self.high_freq_weight * ((t - 0.6) / 0.4) if t > 0.6 else 1.0

        total = low_l = mid_l = high_l = 0.0
        error = v_pred - v_target
        for c in range(C):
            e = torch.fft.fftshift(
                torch.fft.fftn(error[:, :, :, :, c], dim=(-3, -2, -1)), dim=(-3, -2, -1)
            )
            ll = torch.mean(torch.abs(e * low_mask) ** 2)
            ml = torch.mean(torch.abs(e * mid_mask) ** 2)
            hl = torch.mean(torch.abs(e * high_mask) ** 2)
            total += self.base_weight * (ll + ml) + w_high * hl
            low_l += ll.item(); mid_l += ml.item(); high_l += hl.item()

        total /= C
        return total, {
            'loss_total': total.item(),
            'loss_low': low_l / C, 'loss_mid': mid_l / C, 'loss_high': high_l / C, 't': t,
        }


def detail_aware_loss(
    v_pred: torch.Tensor, v_target: torch.Tensor, t: float, high_freq_weight: float = 10.0
) -> Tuple[torch.Tensor, dict]:
    return DetailAwareLoss(high_freq_weight=high_freq_weight)(v_pred, v_target, t)