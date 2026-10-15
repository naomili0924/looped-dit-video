"""Offline motion-aware analysis of saved generation clips (eval_phase4 --save-clips).

For a clips directory <out>/clips/<split>_k<k>/ (win<i>_gen.mp4, win<i>_gt.mp4, meta.json) it computes, per window
and averaged:

  pixel level (where things are)
    psnr, ssim, lpips          full frame, mean over the future frames (as in Phase 4)
    psnr_moving                PSNR restricted to the pixels that move in the real clip (|GT_t - GT_1| > 25/255)
    psnr_late                  PSNR over the last 8 frames (the horizon where motion has accumulated)
    psnr_curve                 per-frame PSNR (the error-vs-horizon curve)
    static baseline            the same numbers for "copy the first future frame" (no motion at all)
  motion level (how things move; ldvflux.motion_eval, RAFT flow, camera motion removed)
    dir_cos, mag_ratio, region_cos, epe, spurious
    reference rows             real clip, static, reversed real clip, another window's real motion

    PYTHONPATH=/workspace/loop-dit python -m ldvflux.clip_eval --clips results/ft_step6000/clips/test_k1 \
        --out results/motion/ft_step6000__test_k1.json [--device cuda:0] [--flow-scale 1.0] [--n 50]
    python -m ldvflux.clip_eval --summarize results/motion --md results/motion/MOTION.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .motion_eval import compare, flow, load_raft

MOVE_THRESH = 25.0  # /255, pixel change vs the first future frame that counts as "moving"


def read_clip(path: Path) -> np.ndarray:
    """mp4 -> uint8 (T, H, W, 3)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split(","))
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "image2pipe", "-vcodec", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        d = d[mask]
    return float(10 * np.log10(255.0 ** 2 / max(float(d.mean()), 1e-6)))


class PerceptualMetrics:
    def __init__(self, device: str):
        import lpips

        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, gen: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
        from torchmetrics.functional.image import structural_similarity_index_measure

        g = torch.from_numpy(gen).permute(0, 3, 1, 2).float().div(255).to(self.device)
        r = torch.from_numpy(gt).permute(0, 3, 1, 2).float().div(255).to(self.device)
        ssim = float(structural_similarity_index_measure(g, r, data_range=1.0))
        lp = float(np.mean([float(self.lpips(g[s : s + 16] * 2 - 1, r[s : s + 16] * 2 - 1).mean())
                            for s in range(0, len(g), 16)]))
        return ssim, lp


def pixel_metrics(gen: np.ndarray, gt: np.ndarray, perc: PerceptualMetrics | None) -> dict:
    """gen, gt uint8 (T, H, W, 3), the T future frames. Returns the model's numbers and the static baseline's."""
    T = len(gt)
    mov = np.abs(gt.astype(np.float32) - gt[0:1].astype(np.float32)).mean(-1) > MOVE_THRESH  # (T, H, W)
    mask = np.broadcast_to(mov[..., None], gt.shape)
    # frame 0 is excluded from every mean: the static baseline reproduces it exactly (PSNR -> inf), and the model's
    # frame 0 is one step from the last context frame; means over frames 1..T-1 make the two comparable
    curve = [psnr(gen[t], gt[t]) for t in range(T)]
    curve_static = [psnr(gt[0], gt[t]) for t in range(T)]
    res = {
        "moving_frac": float(mov[1:].mean()),
        "psnr": float(np.mean(curve[1:])),
        "psnr_moving": float(np.nanmean([psnr(gen[t], gt[t], mask[t]) for t in range(1, T)])),
        "psnr_late": float(np.mean(curve[-8:])),
        "psnr_curve": [round(c, 3) for c in curve],
        "static_psnr": float(np.mean(curve_static[1:])),
        "static_psnr_moving": float(np.nanmean([psnr(gt[0], gt[t], mask[t]) for t in range(1, T)])),
        "static_psnr_late": float(np.mean(curve_static[-8:])),
        "static_psnr_curve": [round(min(c, 99.0), 3) for c in curve_static],
    }
    if perc is not None:
        res["ssim"], res["lpips"] = perc(gen[1:], gt[1:])
        res["static_ssim"], res["static_lpips"] = perc(np.repeat(gt[0:1], T - 1, 0), gt[1:])
    return res


def downscale(frames: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frames
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    h, w = int(round(x.shape[2] * scale / 8)) * 8, int(round(x.shape[3] * scale / 8)) * 8
    y = torch.nn.functional.interpolate(x, size=(h, w), mode="area")
    return y.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def mean_rows(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if not k.endswith("_curve")]
    out = {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}
    for k in [k for k in rows[0] if k.endswith("_curve")]:
        out[k] = [float(x) for x in np.mean([r[k] for r in rows], axis=0)]
    return out


def analyze(clips: Path, device: str, n: int | None, flow_scale: float, large: bool, tau: float, skip_flow: bool) -> dict:
    meta = json.loads((clips / "meta.json").read_text())
    ids = sorted(int(p.name[3:6]) for p in clips.glob("win*_gen.mp4"))[: n or None]
    perc = PerceptualMetrics(device)
    gens, gts, pix = [], [], []
    for i in ids:
        g, r = read_clip(clips / f"win{i:03d}_gen.mp4"), read_clip(clips / f"win{i:03d}_gt.mp4")
        assert g.shape == r.shape, (i, g.shape, r.shape)
        pix.append(pixel_metrics(g, r, perc))
        gens.append(g)
        gts.append(r)
    result = {"clips": str(clips), "n": len(ids), "meta": {k: v for k, v in meta.items() if k != "windows"},
              "pixel": mean_rows(pix), "pixel_per_window": [{k: v for k, v in p.items() if not k.endswith("_curve")} for p in pix]}
    print(f"[{clips.name}] pixel: psnr {result['pixel']['psnr']:.2f} (static {result['pixel']['static_psnr']:.2f}) "
          f"moving {result['pixel']['psnr_moving']:.2f} (static {result['pixel']['static_psnr_moving']:.2f}) "
          f"lpips {result['pixel']['lpips']:.3f}", flush=True)
    if not skip_flow:
        raft = load_raft(device, large)
        f_gt = [flow(raft, downscale(r, flow_scale)) for r in gts]
        f_gen = [flow(raft, downscale(g, flow_scale)) for g in gens]
        rows = {
            "model": [compare(a, b, tau) for a, b in zip(f_gen, f_gt)],
            "ref_real": [compare(b, b, tau) for b in f_gt],
            "ref_static": [compare(np.zeros_like(b), b, tau) for b in f_gt],
            "ref_reversed": [compare(-b[::-1], b, tau) for b in f_gt],
            "ref_other": [compare(f_gt[(j + 1) % len(f_gt)], f_gt[j], tau) for j in range(len(f_gt))],
        }
        result["flow"] = {k: mean_rows(v) for k, v in rows.items()}
        result["flow"]["model_per_window"] = rows["model"]
        result["flow_settings"] = {"scale": flow_scale, "raft": "large" if large else "small", "tau": tau}
        m = result["flow"]["model"]
        print(f"[{clips.name}] flow: dir_cos {m['dir_cos']:.3f} mag_ratio {m['mag_ratio']:.3f} region_cos {m['region_cos']:.3f} "
              f"epe {m['epe']:.2f} spurious {m['spurious']:.3f} (other-window cos {result['flow']['ref_other']['dir_cos']:.3f})", flush=True)
    return result


def summarize(folder: Path, md: Path | None) -> None:
    files = sorted(folder.glob("*__*.json"))
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        model, cell = f.stem.split("__", 1)
        p, fl = d["pixel"], d.get("flow", {}).get("model", {})
        rows.append({"model": model, "cell": cell, "n": d["n"], "psnr": p["psnr"], "psnr_moving": p["psnr_moving"],
                     "psnr_late": p["psnr_late"], "ssim": p.get("ssim"), "lpips": p.get("lpips"),
                     "static_psnr": p["static_psnr"], "static_psnr_moving": p["static_psnr_moving"],
                     "dir_cos": fl.get("dir_cos"), "mag_ratio": fl.get("mag_ratio"), "region_cos": fl.get("region_cos"),
                     "epe": fl.get("epe"), "spurious": fl.get("spurious"),
                     "other_cos": d.get("flow", {}).get("ref_other", {}).get("dir_cos")})
    (folder / "summary.json").write_text(json.dumps(rows, indent=1))
    if md:
        def f(x, nd=3):
            return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
        lines = ["# Motion-aware evaluation", "",
                 "Pixel level: PSNR full frame / on moving pixels / last 8 frames, SSIM, LPIPS; `static` = copy the first future frame. "
                 "Motion level (RAFT flow, camera motion removed, moving pixels only): direction cosine (1 = same direction, 0 = no or "
                 "unrelated motion, -1 = opposite), magnitude ratio (1 = moved the right amount), region cosine (mean flow vector of the "
                 "moving region), end-point error (px/frame), spurious motion (background flow relative to real object motion). "
                 "`other` = direction cosine against another window's real motion (chance level).", ""]
        cells = sorted({r["cell"] for r in rows})
        for cell in cells:
            lines += [f"## {cell}", "", "| model | n | PSNR | PSNR moving | PSNR last 8 | SSIM | LPIPS | dir cos | mag ratio | region cos | EPE | spurious |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            first = next(r for r in rows if r["cell"] == cell)
            lines.append(f"| static (copy first frame) | {first['n']} | {f(first['static_psnr'], 2)} | {f(first['static_psnr_moving'], 2)} | – | – | – | 0 | 0 | 0 | – | 0 |")
            for r in [r for r in rows if r["cell"] == cell]:
                lines.append(f"| {r['model']} | {r['n']} | {f(r['psnr'], 2)} | {f(r['psnr_moving'], 2)} | {f(r['psnr_late'], 2)} | {f(r['ssim'])} | "
                             f"{f(r['lpips'])} | {f(r['dir_cos'])} | {f(r['mag_ratio'])} | {f(r['region_cos'])} | {f(r['epe'], 2)} | {f(r['spurious'])} |")
            lines.append(f"\nchance-level direction cosine (another window's real motion): {f(first['other_cos'])}\n")
        md.write_text("\n".join(lines))
        print("wrote", md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", help="clips directory of one (split, k) cell")
    p.add_argument("--out", help="result json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=None, help="first n windows only")
    p.add_argument("--flow-scale", type=float, default=1.0, help="downscale frames before RAFT (0.5 = 4x faster)")
    p.add_argument("--small", action="store_true", help="raft_small instead of raft_large")
    p.add_argument("--tau", type=float, default=1.0, help="moving-pixel flow threshold (px/frame at the flow scale)")
    p.add_argument("--skip-flow", action="store_true")
    p.add_argument("--summarize", help="folder of result jsons -> summary.json (+ --md)")
    p.add_argument("--md", default=None)
    a = p.parse_args()
    if a.summarize:
        summarize(Path(a.summarize), Path(a.md) if a.md else None)
        return
    res = analyze(Path(a.clips), a.device, a.n, a.flow_scale, not a.small, a.tau, a.skip_flow)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
