"""Time-aware LoRA: low-rank adapters injected into a DiT via forward hooks."""

import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """Basic LoRA: W' = W + B @ A."""

    def __init__(self, in_features: int, out_features: int, r: int = 8,
                 lora_alpha: int = 8, lora_dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.scaling = float(lora_alpha) / float(max(1, r))
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.lora_A.to(dtype=x.dtype)
        B = self.lora_B.to(dtype=x.dtype)
        return self.lora_dropout(x) @ A.T @ B.T * self.scaling


class TimeAwareLORA(nn.Module):
    """Blend of early and late LoRA branches, weighted by normalized timestep."""

    def __init__(self, dim: int, r_early: int = 16, r_late: int = 32,
                 lora_alpha_early: int = 16, lora_alpha_late: int = 32, dropout: float = 0.0):
        super().__init__()
        self.lora_early = LoRALinear(dim, dim, r=r_early, lora_alpha=lora_alpha_early, lora_dropout=dropout)
        self.lora_late  = LoRALinear(dim, dim, r=r_late,  lora_alpha=lora_alpha_late,  lora_dropout=dropout)
        self.time_gate  = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 2), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor, t: float, use_gate: bool = False) -> torch.Tensor:
        if use_gate:
            w = self.time_gate(torch.tensor([[t]], device=x.device, dtype=x.dtype))[0]
            return w[0] * self.lora_early(x) + w[1] * self.lora_late(x)
        if t < 0.3:
            return self.lora_early(x)
        if t > 0.7:
            return self.lora_late(x)
        w_e = (0.7 - t) / 0.4
        w_l = (t - 0.3) / 0.4
        return w_e * self.lora_early(x) + w_l * self.lora_late(x)


class DiTWithTimeAwareLORA(nn.Module):
    """Wrap a DiT model and inject time-aware LoRA via forward hooks.

    Args:
        dit_model: Base DiT to wrap (weights are frozen).
        target_modules: Substrings matched against module names to select LoRA targets.
        r_early / r_late: LoRA rank for early and late timestep branches.
        lora_alpha_early / lora_alpha_late: Scaling factors.
    """

    def __init__(
        self,
        dit_model: nn.Module,
        target_modules: list = ("attn.to_qkv",),
        r_early: int = 16,
        r_late: int = 32,
        lora_alpha_early: int = 16,
        lora_alpha_late: int = 32,
    ):
        super().__init__()
        self.dit_model = dit_model
        self.target_modules = list(target_modules)

        for p in self.dit_model.parameters():
            p.requires_grad = False

        try:
            base_param = next(dit_model.parameters())
            base_device, base_dtype = base_param.device, base_param.dtype
        except StopIteration:
            base_device, base_dtype = torch.device("cpu"), torch.float32

        self.lora_modules = nn.ModuleDict()
        for name, module in dit_model.named_modules():
            if not any(t in name for t in self.target_modules):
                continue
            if hasattr(module, "out_features"):
                dim = int(module.out_features)
            elif hasattr(module, "weight"):
                dim = int(module.weight.shape[0])
            else:
                continue
            lora = TimeAwareLORA(
                dim=dim, r_early=r_early, r_late=r_late,
                lora_alpha_early=lora_alpha_early, lora_alpha_late=lora_alpha_late,
            ).to(device=base_device, dtype=base_dtype)
            self.lora_modules[name.replace(".", "_")] = lora

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                condition: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        t_frac = kwargs.pop("t_fraction", None)
        t_val = float(t_frac) if t_frac is not None else float(t.mean().item() if isinstance(t, torch.Tensor) else t)

        def _hook(module, _inp, out, lora):
            return out + lora(out, t_val)

        hooks = [
            module.register_forward_hook(lambda m, i, o, lm=self.lora_modules[name.replace(".", "_")]: _hook(m, i, o, lm))
            for name, module in self.dit_model.named_modules()
            if name.replace(".", "_") in self.lora_modules
        ]
        out = self.dit_model(z, t, condition, **kwargs)
        for h in hooks:
            h.remove()
        return out

    def get_trainable_parameters(self):
        return [p for p in self.lora_modules.parameters() if p.requires_grad]

    def save_lora_weights(self, path: str):
        if path.endswith(".safetensors"):
            from safetensors.torch import save_file
            save_file(self.lora_modules.state_dict(), path)
        else:
            torch.save({"lora_state_dict": self.lora_modules.state_dict(),
                        "config": {"target_modules": self.target_modules}}, path)

    def load_lora_weights(self, path: str):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            self.lora_modules.load_state_dict(load_file(path))
        else:
            ckpt = torch.load(path, map_location="cpu")
            self.lora_modules.load_state_dict(ckpt["lora_state_dict"])


def create_time_aware_lora_model(base_model: nn.Module, r_early: int = 16, r_late: int = 32):
    return DiTWithTimeAwareLORA(
        dit_model=base_model,
        target_modules=("attn.to_qkv", "attn.to_out"),
        r_early=r_early, r_late=r_late,
        lora_alpha_early=r_early, lora_alpha_late=r_late,
    )