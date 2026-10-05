"""Write videos to mp4."""

from __future__ import annotations

import math
from pathlib import Path

import torch


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    """[B, 3, T, H, W] in [-1, 1] -> uint8 [B, T, H, W, 3]."""
    return ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 4, 1).cpu()


def save_video_grid(video: torch.Tensor, path: str | Path, fps: int = 8, nrow: int | None = None) -> None:
    import imageio.v3 as iio

    frames = to_uint8(video)  # [B, T, H, W, 3]
    b, t, h, w, c = frames.shape
    nrow = nrow or math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    grid = torch.zeros(t, ncol * h, nrow * w, c, dtype=torch.uint8)
    for i in range(b):
        r, q = divmod(i, nrow)
        grid[:, r * h : (r + 1) * h, q * w : (q + 1) * w] = frames[i]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, grid.numpy(), fps=fps, codec="libx264", macro_block_size=1)
