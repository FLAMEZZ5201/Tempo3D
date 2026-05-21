# Tempo3D

Official implementation of **Tempo3D: Efficient Temporal-Aware Fine-Tuning and Multi-View Latent Aggregation for 3D Generation** (ACM TOG / SIGGRAPH 2026).

<div align="center">

Huizhi Zhu, Jiongming Qin, Yusen Wang, Chunxia Xiao  
Wuhan University &nbsp;·&nbsp; Dongfeng Motor Corp. R&D Institute

[[Project Page](https://hzlabx.github.io/Tempo3D/)] &nbsp;·&nbsp; [[Paper](https://hzlabx.github.io/Tempo3D/static/Tempo3D.pdf)] &nbsp;·&nbsp; [[Model](https://huggingface.co/FLAMEZZ/Tempo3D-weights)] &nbsp;·&nbsp; [[Dataset](https://huggingface.co/datasets/FLAMEZZ/TempoDetail)]

</div>

---

## Release

- ✅ Inference code for single-view conditioned 3D generation
- ✅ Pretrained weights
- ✅ TempoDetail dataset


---


## Repository Structure

```
Tempo3D/
├── hy3dshape/                  # Hunyuan3D-2.1 shape pipeline (modified)
│   ├── models/
│   │   ├── autoencoders/       # ShapeVAE and surface extractors
│   │   ├── denoisers/          # DiT denoiser
│   │   └── diffusion/          # Flow matching scheduler
│   ├── pipelines.py            # Main generation pipeline
│   ├── preprocessors.py        # Image preprocessing
│   ├── postprocessors.py       # Mesh post-processing
│   └── background_remover.py   # Background removal utility
└── temporal_decoupling/        # Tempo3D core modules
    ├── time_aware_lora.py      # TS-LoRA implementation
    ├── adaptive_cfg.py         # Adaptive CFG scheduler
    ├── mesh_spectral_analysis.py  # Frequency-band mesh analysis
    ├── spectral_loss.py        # Spectral loss utilities
    ├── inference_stage1_simple.py     # CLI inference entry point
    └── inference_stage1_timeaware.py  # Core inference pipeline
```

---

## Environment Setup

### Requirements

- Python 3.10+
- CUDA 12.4 (recommended)
- PyTorch 2.5.1

### 1. Create conda environment

```bash
conda create -n tempo3d python=3.10 -y
conda activate tempo3d
```

### 2. Install PyTorch with CUDA

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


---

## Model Weights

### Base Model — Hunyuan3D-2.1

The base model is loaded automatically from HuggingFace on first run. By default the inference script uses `tencent/Hunyuan3D-2.1`.

To download manually:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', local_dir='./weights/tencent/Hunyuan3D-2.1')"
```

If downloaded locally, pass `--model_path ./weights/tencent/Hunyuan3D-2.1` to the inference script.

### TS-LoRA Weights

The TS-LoRA checkpoint is a `.pt` file produced by our training pipeline. Place it anywhere and pass the path via `--lora_path`.

**Download our pretrained TS-LoRA weights:** [FLAMEZZ/Tempo3D-weights](https://huggingface.co/FLAMEZZ/Tempo3D-weights)

Expected file layout:

```
weights/
├── tencent/Hunyuan3D-2.1/  # local base model
├── lora_weights.pt         # TS-LoRA checkpoint
└── cond_token.pt           # learnable token
```

---

## TempoDetail Dataset

TempoDetail is a curated dataset of 2,000 high-quality 3D assets.

**Download:** [FLAMEZZ/TempoDetail](https://huggingface.co/datasets/FLAMEZZ/TempoDetail)

The dataset is split into 4 parts (~500 assets each). Download all parts and extract:

```bash
for i in 1 2 3 4; do
    wget https://huggingface.co/datasets/FLAMEZZ/TempoDetail/resolve/main/TempoDetail_part${i}.tar
    tar -xf TempoDetail_part${i}.tar
done
```

Each asset is a `.glb` file organized as:
```
TempoDetail_glb/
├── 00001/
│   └── 00001.glb
├── 00002/
│   └── 00002.glb
...
```

### Dataset Preparation

To prepare your own dataset in the same format, use the scripts in `tools/`:

```
tools/
├── convert_to_glb.py       # Convert watertight OBJ meshes to GLB (TempoDetail already processed)
├── render.py               # Blender multi-view rendering (requires bpy)
├── watertight_and_sample.py  # Watertight mesh + SDF sampling (requires igl)
└── process_dataset.sh      # End-to-end pipeline
```

**Run the full pipeline:**

```bash
bash tools/process_dataset.sh \
    --src /path/to/raw_assets \
    --glb /path/to/TempoDetail_glb \
    --out /path/to/TempoDetail_preprocessed \
    --blender /path/to/blender
```

**Or run each step individually:**

```bash
# Step 1: convert OBJ to GLB
python tools/convert_to_glb.py --src /path/to/raw --dst /path/to/glb_out

# Step 2: render multi-view condition images (requires Blender)
blender --background --python tools/render.py -- \
    --object /path/to/mesh.glb \
    --output_folder /path/to/render_cond \
    --geo_mode --resolution 512

# Step 3: watertight mesh + SDF sampling
python tools/watertight_and_sample.py \
    --input_obj /path/to/mesh.ply \
    --output_prefix /path/to/geo_data/uid
```

---

## Inference

### Quick start

```bash
python temporal_decoupling/inference_stage1_simple.py \
    --image input.png \
    --lora_path weights/lora_weights.pt \
    --output output.glb
```

The output is a `.glb` mesh file viewable in any 3D viewer (e.g. MeshLab, Blender, Windows 3D Viewer).

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--image` | - | Input image path (PNG/JPG, RGBA preferred) |
| `--lora_path` | - | Path to TS-LoRA checkpoint (`.pt`) |
| `--output` | auto | Output mesh path (`.glb`); auto-named if omitted |
| `--model_path` | `tencent/Hunyuan3D-2.1` | HuggingFace repo ID or local model directory |
| `--cond_token_path` | - | Learnable token (`.pt`) |

### Example: with learnable token

```bash
python temporal_decoupling/inference_stage1_simple.py \
    --image input.png \
    --lora_path weights/lora_weights.pt \
    --cond_token_path weights/cond_token.pt \
    --output output.glb
```

### Using as a Python module

```python
from temporal_decoupling.inference_stage1_simple import run_inference

result = run_inference(
    image_path="input.png",
    lora_path="weights/lora_weights.pt",
    output_path="output.glb",
    num_steps=50,
    cfg_max=15.0,
    seed=42,
)

mesh = result["mesh"]           # trimesh.Trimesh object
meta = result["metadata"]       # inference metadata dict
```

---

## Citation

```bibtex
@article{zhu2026tempo3d,
  author    = {Zhu, Huizhi and Qin, Jiongming and Wang, Yusen and Xiao, Chunxia},
  title     = {Tempo3D: Efficient Temporal-Aware Fine-Tuning and Multi-View Latent Aggregation for 3D Generation},
  journal   = {ACM Trans. Graph.},
  year      = {2026},
  volume    = {45},
  number    = {4},
  articleno = {73},
  month     = {July},
  numpages  = {15},
  doi       = {10.1145/3811362},
  publisher = {ACM},
}
```

## Contact

For technical questions, please contact [zhuhuizhi@whu.edu.cn](mailto:zhuhuizhi@whu.edu.cn).

## Acknowledgements

This project builds upon [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1). We thank Tencent for open-sourcing their work.