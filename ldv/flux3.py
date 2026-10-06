"""FLUX 3 Action 7B as a pretrained base: weight loading, LoRA, and its flow-matching conventions.

The pretrained trunk (black-forest-labs/flux-3-action-base, FLUX Kommunity License) is loaded into
LoopedFluxT2V with the action / action_cond / video_cond streams dropped. Its conventions differ from
the Looped-DiT ones in diffusion.py and are implemented here:

    x_t = t * eps + (1 - t) * x0          t = 1 is noise, t = 0 is data
    model predicts the velocity eps - x0   (loss: MSE on the velocity, per exit for deep supervision)
    t ~ sigmoid(width * logit(u)), then the rational shift  t' = s t / (1 + (s - 1) t)
    text tokens are modulated with t = 0; RoPE time ids are in 10 ms units
    the pooled `vector` input is always zero (and has no bias), so it is dropped

Looping, deep supervision and the attention gate are applied exactly as for the WebVid runs: the
pretrained single blocks [7:16] become the shared looped core, run num_loops times.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import LoopedFluxT2V, ModelConfig

FLUX3_BASE = "black-forest-labs/flux-3-action-base"
CTX_DIM = 20480  # Qwen3-VL-4B, 8 layers x 2560
LATENT_CHANNELS = 96
TIME_ID_PER_SECOND = 100  # FLUX: times_to_ids = time * 1000 // 10


def flux3_model_config(**overrides) -> ModelConfig:
    """The FLUX 3 Action 7B layout with the Looped-DiT split used for the WebVid runs."""
    kw = dict(
        latent_channels=LATENT_CHANNELS, patch_size=(1, 1, 1), context_in_dim=CTX_DIM,
        hidden_size=3072, num_heads=24, mlp_ratio=3.0, depth=5, depth_single_blocks=28,
        axes_dim=[32, 32, 32, 32], loop_split=(12, 9, 12), num_loops=4, share_loop_weights=True,
        use_attn_gate=True, use_xsa=False, text_timestep_zero=True, attn_gate_init=4.0,
    )
    kw.update(overrides)
    return ModelConfig(**kw)


def time_id_stride(fps: float, temporal_downsample: int = 4) -> int:
    """RoPE time id spacing between latent frames: one latent frame spans temporal_downsample frames."""
    return int(round(temporal_downsample / fps * TIME_ID_PER_SECOND))


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_KEY_MAP = [
    (r"^emb_in\.video\.weight$", "emb_in.weight"),
    (r"^txt_in\.weight$", "txt_in.weight"),
    (r"^time_in\.(.*)$", r"time_in.\1"),
    (r"^early_stream_modulations\.(video|txt)\.lin\.weight$", r"early_mod.\1.lin.weight"),
    (r"^single_stream_modulations\.(video|txt)\.lin\.weight$", r"single_mod.\1.lin.weight"),
    (r"^content_mode_blocks\.video\.(\d+)\.(.*)$", r"video_mode_blocks.\1.\2"),
    (r"^txt_mode_blocks\.(\d+)\.(.*)$", r"txt_mode_blocks.\1.\2"),
    (r"^single_blocks\.(\d+)\.(.*)$", r"single_blocks.\1.\2"),
    (r"^final_layer\.video\.(.*)$", r"final_layer.\1"),
]


def map_flux3_state_dict(sd: dict[str, Tensor]) -> tuple[dict[str, Tensor], list[str]]:
    """Rename FLUX 3 Action keys to LoopedFluxT2V keys; returns (mapped, dropped keys)."""
    out, dropped = {}, []
    for k, v in sd.items():
        for pat, rep in _KEY_MAP:
            if re.match(pat, k):
                out[re.sub(pat, rep, k)] = v
                break
        else:
            dropped.append(k)
    return out, dropped


def load_flux3_weights(model: LoopedFluxT2V, path: str, strict_blocks: bool = True) -> dict:
    """Load the pretrained trunk into `model` (video + text streams only). New parameters (attention
    gates, mask token) keep their init. Returns a summary dict."""
    from safetensors.torch import load_file

    if not model.cfg.share_loop_weights:
        raise ValueError("the pretrained trunk has one copy of each block: use share_loop_weights=True")
    sd, dropped = map_flux3_state_dict(load_file(path, device="cpu"))
    own = model.state_dict()
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    if strict_blocks:
        bad = [k for k in missing if not (k.startswith("mask_token") or ".img_gate." in k or ".txt_gate." in k)]
        if bad or unexpected:
            raise RuntimeError(f"weight mapping mismatch: missing {bad[:5]}..., unexpected {unexpected[:5]}...")
    shape_bad = [k for k in sd if k in own and own[k].shape != sd[k].shape]
    if shape_bad:
        raise RuntimeError(f"shape mismatch for {shape_bad[:5]}")
    model.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items() if k in own}, strict=False)
    return {"loaded": len([k for k in sd if k in own]), "new_params": missing, "dropped_pretrained": len(dropped),
            "dropped_examples": [k for k in dropped if "action" not in k and "video_cond" not in k][:8]}


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with the base weight frozen. B starts at zero."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        return y + F.linear(F.linear(self.dropout(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scale

    def merge(self) -> nn.Linear:
        """Fold the adapter into a plain Linear (for export)."""
        with torch.no_grad():
            w = self.base.weight.float() + (self.lora_B.float() @ self.lora_A.float()) * self.scale
            lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
            lin.weight.copy_(w.to(lin.weight.dtype))
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")


def add_lora(model: nn.Module, rank: int = 64, alpha: float | None = None, targets=LORA_TARGETS,
             dropout: float = 0.0) -> int:
    """Wrap the target Linear layers of every block in LoRA; returns the number of wrapped layers."""
    alpha = rank if alpha is None else alpha
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                n += 1
    return n


def set_trainable(model: nn.Module, train_gates: bool = True, full_modules: tuple[str, ...] = ()) -> dict:
    """Freeze everything except LoRA params, attention gates and the modules named in full_modules
    (e.g. ("emb_in", "final_layer")). Returns parameter counts."""
    for n, p in model.named_parameters():
        trainable = ".lora_" in n or (train_gates and (".img_gate." in n or ".txt_gate." in n)) \
            or any(n.startswith(m + ".") for m in full_modules)
        p.requires_grad_(trainable)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": tot, "trainable": tr, "trainable_pct": 100.0 * tr / tot}


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# FLUX flow-matching objective and sampler
# ---------------------------------------------------------------------------


def rational_shift(t: Tensor, shift: float) -> Tensor:
    return shift * t / (1.0 + (shift - 1.0) * t)


def sample_timesteps(b: int, device, width: float = 1.0, shift: float = 42.0) -> Tensor:
    """FLUX 3 Action training timesteps: logit(t) ~ Logistic(0, width), then the rational shift."""
    eps = torch.finfo(torch.float32).eps
    u = torch.rand(b, device=device).clamp(eps, 1 - eps)
    t = torch.sigmoid(width * (torch.log(u) - torch.log1p(-u)))
    return rational_shift(t, shift)


def flux_training_loss(model: nn.Module, latents: Tensor, text: Tensor, null_text: Tensor, exit_weights,
                       t_width: float = 1.0, t_shift: float = 42.0, label_drop_rate: float = 0.1):
    """Velocity loss in the FLUX convention, on every loop exit (deep supervision) when exit_weights is given.
    `null_text` is the context of the empty caption, substituted for dropped captions (CFG)."""
    b = latents.shape[0]
    if label_drop_rate > 0:
        drop = torch.rand(b, device=latents.device) < label_drop_rate
        text = torch.where(drop[:, None, None], null_text.expand_as(text), text)
    mask = torch.ones(text.shape[:2], dtype=torch.long, device=text.device)
    t = sample_timesteps(b, latents.device, t_width, t_shift)
    t = t.to(latents.dtype).float()  # the rounded value the noising uses (reference behaviour)
    tb = t.view(b, *([1] * (latents.ndim - 1)))
    eps = torch.randn_like(latents)
    x_t = tb * eps + (1 - tb) * latents
    target = eps - latents
    dims = tuple(range(1, latents.ndim))

    def vloss(pred: Tensor) -> Tensor:
        return (pred.float() - target.float()).pow(2).mean(dim=dims).mean()

    if exit_weights is None:
        loss = vloss(model(x_t, t, text, mask))
        return loss, {"loss": loss.detach()}
    n = len(exit_weights)
    final, exits = model(x_t, t, text, mask, exit_loops=tuple(range(1, n)))
    loss_final = vloss(final)
    exit_losses = [vloss(exits[r]) for r in range(1, n)]
    loss = exit_weights[-1] * loss_final + sum(w * l for w, l in zip(exit_weights, exit_losses))
    metrics = {"loss": loss.detach(), "loss_final": loss_final.detach()}
    metrics.update({f"loss_exit{r}": l.detach() for r, l in enumerate(exit_losses, start=1)})
    return loss, metrics


def timeshift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1.0 + (alpha - 1.0) * t)


@torch.no_grad()
def flux_euler_sample(model: nn.Module, text: Tensor, null_text: Tensor, latent_shape, steps: int = 50,
                      cfg_scale: float = 4.0, shift: float = 5.0, num_loops: int | None = None,
                      generator: torch.Generator | None = None) -> Tensor:
    """Euler from t = 1 (noise) to 0 (data) on the shifted schedule, with classifier-free guidance."""
    was_training = model.training
    model.eval()
    b, device = text.shape[0], text.device
    x = torch.randn(b, *latent_shape, device=device, generator=generator)
    ts = timeshift(shift, torch.linspace(1.0, 0.0, steps + 1, device=device)).tolist()
    mask = torch.ones(text.shape[:2], dtype=torch.long, device=device)
    dtype = next(model.parameters()).dtype
    amp = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    for t_cur, t_prev in zip(ts[:-1], ts[1:]):
        t = torch.full((b,), t_cur, device=device)
        with torch.autocast("cuda", dtype=amp, enabled=x.is_cuda):
            if cfg_scale != 1.0:
                both = model(torch.cat([x, x]), torch.cat([t, t]),
                             torch.cat([text, null_text.expand_as(text)]), torch.cat([mask, mask]), num_loops=num_loops)
                cond, uncond = both.float().chunk(2)
                v = uncond + cfg_scale * (cond - uncond)
            else:
                v = model(x, t, text, mask, num_loops=num_loops).float()
        x = x + (t_prev - t_cur) * v
    model.train(was_training)
    return x
