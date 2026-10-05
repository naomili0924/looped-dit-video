"""Flow-matching objective with deep supervision, and the Euler sampler (Looped-DiT recipe,
adapted to video latents [B, C, T, H, W]).

The denoiser predicts the clean latents x0; the loss compares the implied velocity
(x0_hat - x_t) / (1 - t) with the true one, clamping 1 - t at 0.05 so the weight stays
bounded near t = 1. t = 0 is noise and t = 1 is data.
"""

from __future__ import annotations

import torch
from torch import nn

# Exit weights are rescaled to sum to this (as in Looped-DiT), so weightings differ only in shape.
WEIGHT_SUM = 2.0
EXPONENTIAL_RATIO = 0.5


def deep_supervision_weights(num_loops: int, weighting: str) -> list[float]:
    """Loss weights (w_1, ..., w_N) of the exits after each loop; w_N is the final one.

    At N = 4 (before rescaling to WEIGHT_SUM):
        uniform          1,   1,   1,   1
        final_plus_mean  1/3, 1/3, 1/3, 1
        exponential      1/8, 1/4, 1/2, 1
    """
    if num_loops < 2:
        raise ValueError("deep supervision needs at least two loops")
    if weighting == "uniform":
        raw = [1.0] * num_loops
    elif weighting == "final_plus_mean":
        raw = [1.0 / (num_loops - 1)] * (num_loops - 1) + [1.0]
    elif weighting == "exponential":
        raw = [EXPONENTIAL_RATIO ** (num_loops - r) for r in range(1, num_loops + 1)]
    else:
        raise ValueError(f"unknown deep supervision weighting: {weighting!r}")
    scale = WEIGHT_SUM / sum(raw)
    return [w * scale for w in raw]


def sample_t(b: int, device, mean: float, std: float) -> torch.Tensor:
    return torch.sigmoid(torch.randn(b, device=device) * std + mean).clamp(1e-5, 1 - 1e-5)


def training_loss(
    model: nn.Module,
    latents: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    exit_weights: list[float] | None = None,
    noise_scale: float = 1.0,
    t_logit_mean: float = -0.8,
    t_logit_std: float = 0.8,
    label_drop_rate: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Flow-matching loss. With exit_weights (one per loop) every loop exit is decoded and
    trained on the same target (deep supervision); without, only the final prediction is."""
    b = latents.shape[0]
    if label_drop_rate > 0:
        drop = torch.rand(b, device=latents.device) < label_drop_rate
        text_mask = torch.where(drop[:, None], torch.zeros_like(text_mask), text_mask)
    t = sample_t(b, latents.device, t_logit_mean, t_logit_std)
    tb = t.view(b, *([1] * (latents.ndim - 1)))
    noise = torch.randn_like(latents) * noise_scale
    x_t = latents * tb + noise * (1.0 - tb)
    velocity_scale = (1.0 - tb).clamp_min(0.05)
    target = (latents - x_t) / velocity_scale
    dims = tuple(range(1, latents.ndim))

    def flow_loss(pred_x0: torch.Tensor) -> torch.Tensor:
        return ((pred_x0.float() - x_t) / velocity_scale - target).pow(2).mean(dim=dims).mean()

    if exit_weights is None:
        loss = flow_loss(model(x_t, t, text, text_mask))
        return loss, {"loss": loss.detach()}

    num_loops = len(exit_weights)
    final, exits = model(x_t, t, text, text_mask, exit_loops=tuple(range(1, num_loops)))
    loss_final = flow_loss(final)
    exit_losses = [flow_loss(exits[r]) for r in range(1, num_loops)]
    loss = exit_weights[-1] * loss_final + sum(w * l for w, l in zip(exit_weights, exit_losses))
    metrics = {"loss": loss.detach(), "loss_final": loss_final.detach()}
    metrics.update({f"loss_exit{r}": l.detach() for r, l in enumerate(exit_losses, start=1)})
    return loss, metrics


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    latent_shape: tuple[int, int, int, int],
    steps: int = 100,
    cfg_scale: float = 6.0,
    noise_scale: float = 1.0,
    num_loops: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler integration from t = 0 (noise) to t = 1 (data) with classifier-free guidance.
    latent_shape = (C, T, H, W). Returns latents [B, C, T, H, W] (float32)."""
    was_training = model.training
    model.eval()
    b, device = text.shape[0], text.device
    x = torch.randn(b, *latent_shape, device=device, generator=generator) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    null_mask = torch.zeros_like(text_mask)
    dtype = next(model.parameters()).dtype
    amp = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        t = torch.full((b,), float(t0), device=device)
        with torch.autocast("cuda", dtype=amp, enabled=x.is_cuda):
            if cfg_scale != 1.0:
                both = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([text, text]),
                             torch.cat([text_mask, null_mask]), num_loops=num_loops)
                cond, uncond = both.float().chunk(2)
                pred = uncond + (cond - uncond) * cfg_scale
            else:
                pred = model(x, t, text, text_mask, num_loops=num_loops).float()
        v = (pred - x) / (1.0 - t0).clamp_min(0.05)
        x = x + v * (t1 - t0)
    model.train(was_training)
    return x
