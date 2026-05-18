"""3D mesh spectral analysis: voxelization, FFT, and frequency-band energy decomposition."""

import torch
import numpy as np
import trimesh
from typing import Dict, Optional
import matplotlib.pyplot as plt


class MeshSpectralAnalyzer:
    """Analyze a trimesh object's frequency-domain energy distribution.

    Args:
        voxel_resolution: Resolution of the intermediate voxel grid.
    """

    def __init__(self, voxel_resolution: int = 64):
        self.voxel_resolution = voxel_resolution

    def mesh_to_voxel(self, mesh: trimesh.Trimesh) -> Optional[torch.Tensor]:
        if mesh is None:
            return None
        v = mesh.vertices - mesh.vertices.mean(axis=0)
        scale = np.abs(v).max()
        if scale > 0:
            v = v / scale
        res = self.voxel_resolution
        grid = np.zeros((res, res, res), dtype=np.float32)
        coords = np.clip(((v + 1) / 2 * (res - 1)).astype(int), 0, res - 1)
        for c in coords:
            grid[c[0], c[1], c[2]] = 1.0
        try:
            vox = mesh.voxelized(pitch=2.0 / res)
            if vox.matrix.shape == (res, res, res):
                grid = vox.matrix.astype(np.float32)
        except Exception as e:
            print(f"Warning: trimesh voxelization failed, using vertex fallback: {e}")
        return torch.from_numpy(grid).float()

    def compute_3d_fft(self, voxel_grid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if voxel_grid is None:
            return None
        return torch.fft.fftshift(torch.abs(torch.fft.fftn(voxel_grid)))

    def separate_frequency_bands(
        self,
        fft_magnitude: Optional[torch.Tensor],
        low_cutoff: float = 0.15,
        high_cutoff: float = 0.85,
    ) -> Dict[str, float]:
        if fft_magnitude is None:
            return dict(low=0.0, mid=0.0, high=0.0, total=0.0,
                        low_ratio=0.0, mid_ratio=0.0, high_ratio=0.0)
        H, W, D = fft_magnitude.shape
        fh, fw, fd = torch.meshgrid(
            torch.fft.fftfreq(H).abs(),
            torch.fft.fftfreq(W).abs(),
            torch.fft.fftfreq(D).abs(),
            indexing='ij',
        )
        freq = torch.sqrt(fh**2 + fw**2 + fd**2)
        freq = freq / freq.max()
        low   = (fft_magnitude * (freq <= low_cutoff)).sum().item()
        high  = (fft_magnitude * (freq > high_cutoff)).sum().item()
        mid   = (fft_magnitude * ((freq > low_cutoff) & (freq <= high_cutoff))).sum().item()
        total = fft_magnitude.sum().item() + 1e-8
        return dict(low=low, mid=mid, high=high, total=total,
                    low_ratio=low/total, mid_ratio=mid/total, high_ratio=high/total)

    def analyze_mesh(self, mesh: trimesh.Trimesh) -> Dict[str, float]:
        return self.separate_frequency_bands(self.compute_3d_fft(self.mesh_to_voxel(mesh)))

    def visualize_spectrum(
        self, fft_magnitude: torch.Tensor, save_path: Optional[str] = None
    ):
        if fft_magnitude is None:
            return
        H, W, D = fft_magnitude.shape
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (s, title) in zip(axes, [
            (fft_magnitude[H // 2, :, :], 'XY'),
            (fft_magnitude[:, W // 2, :], 'XZ'),
            (fft_magnitude[:, :, D // 2], 'YZ'),
        ]):
            ax.imshow(torch.log(s + 1).cpu().numpy(), cmap='hot')
            ax.set_title(f'{title} Plane')
            ax.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def compute_mesh_frequency_spectrum(
    mesh: trimesh.Trimesh, voxel_resolution: int = 64
) -> np.ndarray:
    """Return a 16-bin radial frequency spectrum for the given mesh."""
    analyzer = MeshSpectralAnalyzer(voxel_resolution)
    fft = analyzer.compute_3d_fft(analyzer.mesh_to_voxel(mesh))
    if fft is None:
        return np.zeros(16)
    H, W, D = fft.shape
    fh, fw, fd = torch.meshgrid(
        torch.fft.fftfreq(H).abs(),
        torch.fft.fftfreq(W).abs(),
        torch.fft.fftfreq(D).abs(),
        indexing='ij',
    )
    freq = torch.sqrt(fh**2 + fw**2 + fd**2)
    freq = freq / freq.max()
    bins = np.zeros(16)
    for i in range(16):
        bins[i] = (fft * ((freq >= i / 16) & (freq < (i + 1) / 16))).sum().item()
    return bins