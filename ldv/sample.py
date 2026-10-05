"""Generate videos from a checkpoint at any loop depth.

    python -m ldv.sample --checkpoint /dev/shm/ldv/outputs/run1/ema_latest.pt \
        --prompt "Waves crashing on a rocky beach" --loops 1 4 8 --out samples/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import TrainConfig
from .diffusion import euler_sample
from .encoders import TextEncoder, WanVAE
from .model import LoopedFluxT2V
from .video import save_video_grid


def load_model(path: str, device, weights: str = "ema") -> tuple[LoopedFluxT2V, TrainConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(ckpt["config"])
    model = LoopedFluxT2V(cfg.model)
    model.load_state_dict({k: v for k, v in ckpt[weights].items()})
    return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False), cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", nargs="+", required=True)
    p.add_argument("--loops", type=int, nargs="+", default=[None])
    p.add_argument("--frames", type=int, default=5, help="latent frames (5 = 17 video frames)")
    p.add_argument("--size", type=int, default=32, help="latent height/width (32 = 256 px)")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="samples")
    args = p.parse_args()
    device = torch.device("cuda")
    model, cfg = load_model(args.checkpoint, device)
    text_encoder = TextEncoder(cfg.text_encoder, cfg.prompt_length, device)
    vae = WanVAE(device, repo=cfg.vae)
    text, mask = text_encoder(args.prompt)
    shape = (cfg.model.latent_channels, args.frames, args.size, args.size)
    for loops in args.loops:
        g = torch.Generator(device=device).manual_seed(args.seed)
        lat = euler_sample(model, text.to(torch.bfloat16), mask, shape, args.steps, args.cfg,
                           cfg.noise_scale, loops, generator=g)
        name = f"loops{loops or cfg.model.num_loops}.mp4"
        save_video_grid(vae.decode(lat), Path(args.out) / name)
        print("wrote", Path(args.out) / name)


if __name__ == "__main__":
    main()
