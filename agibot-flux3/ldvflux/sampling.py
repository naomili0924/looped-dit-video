"""Generation with a video-only FLUX policy: k context frames + caption -> F future frames.

Works on a plain or an FSDP-sharded DiT (every rank must call ``generate`` the same number of times).
"""

from __future__ import annotations

import torch
from torch import Tensor

from flux_action.inference.sampling import timeshift
from flux_action.models.text_encoder import VEC_DIM
from flux_action.processing import packing, video_only


@torch.no_grad()
def encode_windows(policy, frames: Tensor, k: int) -> tuple[Tensor | None, Tensor]:
    """uint8 ``(B, k + F, 3, H, W)`` -> (context latents or None, future latents), as in training."""
    cfg = policy.config
    device = policy.device
    videos = torch.stack([
        packing.materialize_video(f[None], None, device, layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw)
        for f in frames
    ])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cond = video_only.encode_clip(policy.video_vae, videos[:, :, :k], cfg.latent_hw) if k else None
        future = video_only.encode_clip(policy.video_vae, videos[:, :, k:], cfg.latent_hw)
    return cond, future


@torch.no_grad()
def generate(policy, cond: Tensor | None, k: int, captions: list[str], *, n_future_latents: int, steps: int = 30,
             cfg_scale: float = 4.0, shift: float = 5.0, seed: int = 0, num_loops: int | None = None) -> Tensor:
    """Euler on the shifted schedule (t = 1 noise -> 0 data) with classifier-free guidance on the caption
    (the empty caption is the null). Returns future latents ``(B, 96, n_future_latents, h, w)`` float32."""
    pcfg = policy.config
    device = policy.device
    b = len(captions)
    h, w = pcfg.latent_hw
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(b, 96, n_future_latents, h, w, generator=g).to(device)
    video = video_only.pack(cond, noise, k, pcfg.fps)
    ctx_c = [c for c, _ in policy._contexts(captions, device)]
    ctx_u = [c for c, _ in policy._contexts([""] * b, device)]
    x = video["x_video"].float()
    ts = timeshift(shift, torch.linspace(1.0, 0.0, steps + 1)).tolist()
    extra = {"num_loops": num_loops} if num_loops is not None else {}

    def velocity(ctxs, xv: Tensor, t: float) -> Tensor:
        kw = {"x_video": xv.to(torch.bfloat16), "x_video_ids": video["x_video_ids"].to(device),
              "x_video_timesteps": torch.full(xv.shape[:2], t, device=device)}
        if cond is not None:
            kw.update(x_video_cond=video["x_video_cond"], x_video_cond_ids=video["x_video_cond_ids"].to(device),
                      x_video_cond_timesteps=torch.zeros(video["x_video_cond"].shape[:2], device=device))
        packed, seqlens = packing.pack_windows(kw, ctxs, VEC_DIM)
        return policy.dit(**policy._cast_inputs(packed), seqlens=seqlens, **extra)["x_video"].float().reshape(xv.shape)

    for t_cur, t_prev in zip(ts[:-1], ts[1:]):
        v_c = velocity(ctx_c, x, t_cur)
        if cfg_scale == 1.0:
            v = v_c
        else:
            v_u = velocity(ctx_u, x, t_cur)
            v = v_u + cfg_scale * (v_c - v_u)
        x = x + (t_prev - t_cur) * v
    # tokens (B, nf*h*w, 96) -> (B, 96, nf, h, w)
    return x.reshape(b, n_future_latents, h, w, 96).permute(0, 4, 1, 2, 3).contiguous()


@torch.no_grad()
def decode(policy, latents: Tensor) -> Tensor:
    """Latents ``(B, 96, n, h, w)`` -> video ``(B, 3, 1 + 4 (n - 1), H, W)`` in [-1, 1]."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.cat([policy.video_vae.decode(z[None].to(torch.bfloat16)).float() for z in latents])
