"""Single-view inference entry point.
Usage:
  python inference_stage1_simple.py \\
    --image input.png \\
    --lora_path ./lora_weights.pt \\
    --output output.glb \\
    --cfg_max 40.0 \\
    --num_steps 50
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from temporal_decoupling.inference_stage1_timeaware import SingleViewInferencePipeline


def run_inference(
    image_path: str,
    lora_path: str,
    output_path: str = None,
    model_path: str = 'tencent/Hunyuan3D-2.1',
    num_steps: int = 50,
    cfg_max: float = 35.0,
    seed: int = 42,
    cond_token_path: str = None,
    r_early: int = 8,
    r_late: int = 8,
    enable_flashvdm: bool = False,
) -> dict:
    if output_path is None:
        ts = datetime.now().strftime("%H%M%S")
        output_path = f"./{Path(image_path).stem}_stage1_{ts}.glb"

    pipeline = SingleViewInferencePipeline(
        model_path=model_path,
        lora_weights_path=lora_path,
        cond_token_path=cond_token_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cfg_schedule_type='adaptive',
        cfg_min=5.0,
        cfg_mid=15.0,
        cfg_max=cfg_max,
        enable_spectral_analysis=True,
        r_early=r_early,
        r_late=r_late,
        enable_flashvdm=enable_flashvdm,
    )

    return pipeline(
        image=image_path,
        num_inference_steps=num_steps,
        seed=seed,
        output_path=output_path,
        remove_background=True,
        octree_resolution=512,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Tempo3D Stage 1 inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--image',          type=str, required=True,                      help='Input image path')
    parser.add_argument('--lora_path',      type=str, required=True,                      help='LoRA checkpoint (.pt)')
    parser.add_argument('--output',         type=str, default=None,                        help='Output mesh path')
    parser.add_argument('--model_path',     type=str, default='tencent/Hunyuan3D-2.1',    help='Base model path or HF repo')
    parser.add_argument('--cond_token_path',type=str, default=None,                        help='Learnable cond token (.pt)')
    parser.add_argument('--num_steps',      type=int, default=50,                          help='Diffusion steps')
    parser.add_argument('--cfg_max',        type=float, default=15.0,                      help='Max CFG scale')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--r_early',        type=int, default=8,                            help='LoRA rank for early timesteps')
    parser.add_argument('--r_late',         type=int, default=8,                            help='LoRA rank for late timesteps')
    parser.add_argument('--enable_flashvdm',action='store_true',                            help='Enable FlashVDM decoder')
    args = parser.parse_args()

    if not Path(args.image).exists():
        parser.error(f"Image not found: {args.image}")
    if not Path(args.lora_path).exists():
        parser.error(f"LoRA checkpoint not found: {args.lora_path}")

    run_inference(
        image_path=args.image,
        lora_path=args.lora_path,
        output_path=args.output,
        model_path=args.model_path,
        num_steps=args.num_steps,
        cfg_max=args.cfg_max,
        seed=args.seed,
        cond_token_path=args.cond_token_path,
        r_early=args.r_early,
        r_late=args.r_late,
        enable_flashvdm=args.enable_flashvdm,
    )

    print("Done.")


if __name__ == "__main__":
    main()