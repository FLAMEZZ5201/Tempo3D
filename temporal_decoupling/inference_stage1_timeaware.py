"""Single-view inference pipeline with time-aware LoRA and adaptive CFG."""

import sys
import types
from pathlib import Path
from typing import Optional, Dict, Union
from datetime import datetime

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from temporal_decoupling.time_aware_lora import DiTWithTimeAwareLORA
from temporal_decoupling.adaptive_cfg import AdaptiveCFGScheduler
from temporal_decoupling.mesh_spectral_analysis import MeshSpectralAnalyzer

try:
    from hy3dshape.background_remover import BackgroundRemover
except Exception:
    BackgroundRemover = None


def _configure_latent_context(pipeline, scale: float = 4.0):
    try:
        base, dim = pipeline.vae.latent_shape
        n = int(round(base * scale))
        if n == base:
            return
        pipeline.vae.latent_shape = (n, dim)
        if hasattr(pipeline.vae, 'transformer'):
            tr = pipeline.vae.transformer
            if hasattr(tr, 'n_ctx'):
                tr.n_ctx = n
            for blk in getattr(tr, 'resblocks', []):
                if hasattr(blk, 'attn') and hasattr(blk.attn, 'n_ctx'):
                    blk.attn.n_ctx = n
        if hasattr(pipeline.vae, 'encoder') and hasattr(pipeline.vae.encoder, 'num_latents'):
            pipeline.vae.encoder.num_latents = n
        if hasattr(pipeline.vae, 'geo_decoder'):
            cad = getattr(pipeline.vae.geo_decoder, 'cross_attn_decoder', None)
            if cad and hasattr(cad, 'attn') and hasattr(cad.attn, 'attention'):
                if hasattr(cad.attn.attention, 'n_data'):
                    cad.attn.attention.n_data = n
    except Exception:
        pass


class SingleViewInferencePipeline:
    """Single-view image-to-3D inference with a time-aware LoRA checkpoint.

    Args:
        model_path: HuggingFace repo or local path for the base Hunyuan3D-2.1 model.
        lora_weights_path: Path to a ``lora_weights.pt`` produced by Stage 1 training.
        cond_token_path: Optional path to a learned conditioning token (``.pt``).
        device: Target device (``'cuda'`` or ``'cpu'``).
        dtype: Inference dtype (default ``torch.float16``).
        cfg_schedule_type: CFG schedule mode — ``'adaptive'``, ``'smooth'``, or ``'fixed'``.
        cfg_min / cfg_mid / cfg_max: CFG scale bounds for the adaptive schedule.
        octree_resolution: Marching-cubes octree resolution for the base pipeline.
        enable_spectral_analysis: Whether to compute frequency-band statistics on the output mesh.
        r_early / r_late: LoRA rank for early and late timestep branches.
        enable_flashvdm: Enable FlashVDM decoder (faster but may reduce quality at high res).
    """

    def __init__(
        self,
        model_path: str,
        lora_weights_path: str,
        cond_token_path: Optional[str] = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
        cfg_schedule_type: str = 'adaptive',
        cfg_min: float = 5.0,
        cfg_mid: float = 15.0,
        cfg_max: float = 22.0,
        octree_resolution: int = 384,
        enable_spectral_analysis: bool = True,
        r_early: int = 8,
        r_late: int = 8,
        enable_flashvdm: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.enable_spectral = enable_spectral_analysis

        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path, device=device, dtype=dtype, octree_resolution=octree_resolution,
        )
        self.pipeline.model.to(dtype=dtype).eval()
        _configure_latent_context(self.pipeline)

        self.model = DiTWithTimeAwareLORA(
            dit_model=self.pipeline.model,
            target_modules=["attn.to_qkv"],
            r_early=r_early, r_late=r_late,
            lora_alpha_early=r_early, lora_alpha_late=r_late,
        )
        self.model.load_lora_weights(lora_weights_path)
        self.model.to(dtype=dtype)
        self.pipeline.model = self.model
        print(f"[SingleView] LoRA loaded: {lora_weights_path}")

        self.cond_token = None
        if cond_token_path and Path(cond_token_path).exists():
            self.cond_token = torch.load(cond_token_path, map_location='cpu')
            print(f"[SingleView] Cond token loaded: {cond_token_path}")
        else:
            print(f"[SingleView] Cond token: not used")

        self.cfg_scheduler = AdaptiveCFGScheduler(
            schedule_type=cfg_schedule_type,
            cfg_min=cfg_min, cfg_mid=cfg_mid, cfg_max=cfg_max,
        )

        try:
            self.pipeline.enable_flashvdm(enabled=enable_flashvdm)
            print(f"[SingleView] FlashVDM: {'enabled' if enable_flashvdm else 'disabled'}")
        except Exception:
            pass

        self.rembg = BackgroundRemover() if BackgroundRemover is not None else None
        self.spectral_analyzer = MeshSpectralAnalyzer(voxel_resolution=64) if enable_spectral_analysis else None

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, Image.Image],
        num_inference_steps: int = 50,
        guidance_scale: float = 15.0,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        remove_background: bool = True,
        octree_resolution: Optional[int] = None,
        verbose: bool = True,
        cond_token_scale: float = 0.7,
        init_noise_mult: float = 0.9,
    ) -> Dict:
        if isinstance(image, str):
            image = Image.open(image).convert("RGBA")
        if remove_background and self.rembg is not None:
            image = self.rembg(image)

        extra_tok = self.cond_token
        if extra_tok is not None:
            if cond_token_scale != 1.0:
                extra_tok = extra_tok * float(cond_token_scale)
            try:
                extra_tok = extra_tok.to(device=self.pipeline.device, dtype=self.dtype)
            except Exception:
                extra_tok = extra_tok.to(self.device).to(self.dtype)

        if seed is not None:
            torch.manual_seed(seed)

        orig_sigma = getattr(self.pipeline.scheduler, 'init_noise_sigma', None)
        if orig_sigma is not None and init_noise_mult != 1.0:
            try:
                self.pipeline.scheduler.init_noise_sigma = orig_sigma * float(init_noise_mult)
            except Exception:
                pass

        orig_encode_cond = self.pipeline.encode_cond

        def _patched_encode_cond(self_pipeline, *args, **kwargs):
            cond = orig_encode_cond(*args, **kwargs)
            if extra_tok is not None:
                try:
                    d = cond["main"].shape[-1]
                    # append learned token (with zero unconditional counterpart) along sequence axis
                    zero = torch.zeros(1, 1, d, dtype=cond["main"].dtype, device=cond["main"].device).half()
                    tok  = extra_tok.half().to(cond["main"].device)
                    cond["main"] = torch.cat([cond["main"], torch.cat([zero, tok], dim=0)], dim=1)
                except Exception:
                    pass
            return cond

        try:
            self.pipeline.encode_cond = types.MethodType(_patched_encode_cond, self.pipeline)
            meshes = self.pipeline(
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                extra_cond_tok=None,
                append_extra_cond_tok=False,
                classifier_scale=0.0,
                octree_resolution=octree_resolution or 384,
                mc_algo=None,
                output_type='trimesh',
                enable_pbar=verbose,
            )
        finally:
            if orig_sigma is not None:
                try:
                    self.pipeline.scheduler.init_noise_sigma = orig_sigma
                except Exception:
                    pass
            try:
                self.pipeline.encode_cond = orig_encode_cond
            except Exception:
                pass

        mesh = meshes[0]

        spectral_result = None
        if self.spectral_analyzer is not None:
            try:
                spectral_result = self.spectral_analyzer.analyze_mesh(mesh)
            except Exception:
                pass

        if output_path is not None:
            outp = Path(output_path)
            if not outp.suffix:
                outp = outp.with_suffix(".glb")
            outp.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Saving mesh to {outp}")
            mesh.export(str(outp))

        return {
            'mesh': mesh,
            'spectral_analysis': spectral_result,
            'metadata': {
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'octree_resolution': octree_resolution or 384,
                'timestamp': datetime.now().isoformat(),
            },
        }