"""Motion-aware evaluation: compare the optical flow of a generated clip with the flow of the real continuation.

Pixel metrics (PSNR/SSIM/LPIPS) compare *where things are*; these compare *how things move*. RAFT (torchvision)
estimates the flow F_t: frame t -> t+1 for the real clip and for the generated clip; the comparison is restricted to
the pixels that really move (|F_gt| > ``tau``) and is tolerant to an object being drawn a few pixels off:

  dir_cos    mean cosine between F_gen and F_gt over the moving pixels (1 = same direction, 0 = unrelated or no
             motion in the generation, -1 = opposite)
  mag_ratio  mean |F_gen| / mean |F_gt| over the moving pixels (1 = moved the right amount, 0 = did not move)
  region_cos cosine between the *mean* flow vectors of the moving region (robust to position offsets)
  epe        mean end-point error |F_gen - F_gt| over the moving pixels (px/frame)
  spurious   mean |F_gen| outside the moving region / mean |F_gt| inside it (background that should not move)
  moving_frac fraction of pixels in the real clip that move (> tau), for reference

Reference rows: ``static`` (no motion: cos 0, ratio 0), ``reversed`` (real clip played backwards: cos -1, ratio 1)
and ``other`` (the real motion of another window: unrelated motion) calibrate the scale.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

EPS = 1e-6


def load_raft(device: str = "cpu", large: bool = False):
    from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

    if large:
        w = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=w)
    else:
        w = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=w)
    model = model.eval().to(device)
    model.transforms = w.transforms()
    return model


@torch.no_grad()
def flow(model, frames: np.ndarray, batch: int = 16, iters: int = 12) -> np.ndarray:
    """uint8 frames (T, H, W, 3) -> flow (T-1, 2, H, W) float32, frame t -> t+1, in pixels. H, W multiples of 8."""
    device = next(model.parameters()).device
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    out = []
    for s in range(0, len(x) - 1, batch):
        a, b = x[s : s + batch], x[s + 1 : s + 1 + batch]
        n = min(len(a), len(b))
        a, b = model.transforms(a[:n], b[:n])
        f = model(a.to(device), b.to(device), num_flow_updates=iters)[-1]
        out.append(f.cpu())
    return torch.cat(out).numpy()


def compensate_camera(f: np.ndarray) -> np.ndarray:
    """Subtract each frame's median flow vector (the head camera's own motion) so that only objects and arms that
    move relative to the scene count as motion."""
    med = np.median(f.reshape(f.shape[0], 2, -1), axis=2)  # (T-1, 2)
    return f - med[:, :, None, None]


def compare(f_gen: np.ndarray, f_gt: np.ndarray, tau: float = 1.0, compensate: bool = True) -> dict[str, float]:
    """Flow metrics between two flow sequences (T-1, 2, H, W) of the same clip length."""
    assert f_gen.shape == f_gt.shape, (f_gen.shape, f_gt.shape)
    if compensate:
        f_gen, f_gt = compensate_camera(f_gen), compensate_camera(f_gt)
    mag_gt = np.linalg.norm(f_gt, axis=1)  # (T-1, H, W)
    mag_gen = np.linalg.norm(f_gen, axis=1)
    mask = mag_gt > tau
    if mask.sum() == 0:
        return {"moving_frac": 0.0}
    dot = (f_gen * f_gt).sum(1)
    cos = dot / (mag_gen * mag_gt + EPS)  # ~0 where the generation does not move
    epe = np.linalg.norm(f_gen - f_gt, axis=1)
    # region-level: mean vector of the moving region per frame, then cosine, averaged over frames with motion
    rc = []
    for t in range(len(mask)):
        m = mask[t]
        if m.sum() < 20:
            continue
        vg = f_gen[t][:, m].mean(1)
        vr = f_gt[t][:, m].mean(1)
        rc.append(float(vg @ vr / (np.linalg.norm(vg) * np.linalg.norm(vr) + EPS)))
    bg = ~mask
    return {
        "moving_frac": float(mask.mean()),
        "dir_cos": float(cos[mask].mean()),
        "mag_ratio": float(mag_gen[mask].mean() / (mag_gt[mask].mean() + EPS)),
        "region_cos": float(np.mean(rc)) if rc else float("nan"),
        "epe": float(epe[mask].mean()),
        "spurious": float(mag_gen[bg].mean() / (mag_gt[mask].mean() + EPS)),
    }


def read_grid(path: str, cw: int = 512, ch: int = 384, rows: int = 8) -> np.ndarray:
    """eval_phase4 grid video -> uint8 (T, rows*ch, 2*cw, 3)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "image2pipe", "-vcodec", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, rows * ch, 2 * cw, 3)


def main() -> None:
    p = argparse.ArgumentParser(description="flow metrics on the 8-window eval grids (GT = left half of --gt)")
    p.add_argument("--gt", required=True, help="grid whose left half is the ground truth")
    p.add_argument("--models", nargs="+", required=True, help="name=path of grids whose right half is generated")
    p.add_argument("--k", type=int, required=True, help="context frames in the grid (skipped)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--large", action="store_true", help="raft_large instead of raft_small")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--no-compensate", action="store_true", help="keep the camera's own motion in the flows")
    a = p.parse_args()
    comp = not a.no_compensate
    model = load_raft(a.device, a.large)
    cw, ch = 512, 384
    gt_grid = read_grid(a.gt)[a.k :]  # future frames only
    gt = [gt_grid[:, i * ch : (i + 1) * ch, :cw] for i in range(8)]
    f_gt = [flow(model, g) for g in gt]
    print(f"GT flow done: raw moving fraction {np.mean([(np.linalg.norm(f, axis=1) > a.tau).mean() for f in f_gt]):.3f}", flush=True)
    results: dict[str, dict] = {}

    def add(name: str, flows: list[np.ndarray]) -> None:
        rows = [compare(fg, fr, a.tau, comp) for fg, fr in zip(flows, f_gt)]
        keys = [k for k in rows[0] if k != "moving_frac"]
        results[name] = {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}
        results[name]["moving_frac"] = float(np.mean([r["moving_frac"] for r in rows]))
        print(f"[{name}] " + " ".join(f"{k} {v:.3f}" for k, v in results[name].items()), flush=True)

    add("reference: real clip", f_gt)
    add("reference: static (no motion)", [np.zeros_like(f) for f in f_gt])
    add("reference: real clip reversed", [-f[::-1] for f in f_gt])
    add("reference: another window's real motion", [f_gt[(i + 1) % 8] for i in range(8)])
    for spec in a.models:
        name, path = spec.split("=", 1)
        grid = read_grid(path)[a.k :]
        gen = [grid[:, i * ch : (i + 1) * ch, cw:] for i in range(8)]
        add(name, [flow(model, g) for g in gen])
    Path(a.out).write_text(json.dumps({"k": a.k, "tau": a.tau, "compensate": comp, "raft": "large" if a.large else "small",
                                       "results": results}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
