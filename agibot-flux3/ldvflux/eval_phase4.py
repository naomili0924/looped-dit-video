"""Phase 4 evaluation: a model x {val, test, test_unseen} x context lengths k -> PSNR / SSIM / LPIPS, FVD, prompt
adherence (VLM judge), qualitative grids. One GPU, no gradients.

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python -m ldvflux.eval_phase4 \
        --name base --splits val test test_unseen --k 0 1 5 9 17 --n-recon 200 --n-fvd 1000 --fvd-k 1 17 \
        --out ../results/base
    # a fine-tuned EMA: --ema /path/ema_step-N.safetensors ; a looped model: --loops 4 (Phase 5)

Every model is evaluated on the same fixed windows (seeded picks of each split, K_MAX context available), the
same noise seeds and sampler settings (Euler 30 steps, CFG 4, shift 5). Generations for a (split, k) are made
once and reused for every metric. Results: <out>/<split>_k<k>.json and <out>/summary.json; grids as mp4.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from flux_action.config import PolicyConfig
from flux_action.policy import FluxActionPolicy

from . import sampling
from .evaluate import _load_windows, _save_grid
from .stream import fixed_windows, load_splits, trim

DEFAULT_CONFIG = "/workspace/loop-dit/configs/video_ft_384.json"


# ---------------------------------------------------------------------------- model
def load_policy(config_path: str, ema: str | None, device: torch.device, loops: int | None = None):
    cfg = json.load(open(config_path))
    pol = dict(cfg["policy"])
    pol["torch_dtype"] = "bfloat16"
    retrofit = loops is not None and pol.get("loop") is None
    if retrofit:  # zero-shot loop-depth probe of a plain checkpoint: the gate-free Looped-DiT retrofit at depth N
        pol["loop"] = {"loop_split": [12, 9, 12], "num_loops": loops, "attn_gate": False, "xsa": False}
        pol["overlay_weights"] = ema
    policy = FluxActionPolicy(PolicyConfig(**pol))
    if ema and not retrofit:
        from safetensors.torch import load_file

        sd = {k.removeprefix("dit."): v for k, v in load_file(ema).items()}
        res = policy.dit.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        # the frozen conditioning head is not in the EMA; everything else must be
        missing = [k for k in res.missing_keys if not k.startswith("final_layer.video_cond")]
        assert not missing, missing[:5]
    policy.dit.to(device)
    for m in policy.frozen.modules():
        m.to(device)
    policy.eval()
    policy.loops = loops
    return policy, cfg


# ---------------------------------------------------------------------------- metrics
class Recon:
    """Per-frame PSNR / SSIM / LPIPS between generated and ground-truth future frames."""

    def __init__(self, device):
        import lpips

        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, gen: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
        """gen float (3, T, H, W) in [-1, 1]; gt uint8 (T, 3, H, W)."""
        from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

        g = ((gen.permute(1, 0, 2, 3).float().clamp(-1, 1) + 1) / 2).to(self.device)  # (T, 3, H, W) in [0, 1]
        r = (gt.float() / 255).to(self.device)
        psnr = float(peak_signal_noise_ratio(g, r, data_range=1.0, reduction="elementwise_mean", dim=(1, 2, 3)))
        ssim = float(structural_similarity_index_measure(g, r, data_range=1.0))
        lp = float(self.lpips(g * 2 - 1, r * 2 - 1).mean())
        return {"psnr": psnr, "ssim": ssim, "lpips": lp}


def to_uint8_video(gen: torch.Tensor) -> np.ndarray:
    """(3, T, H, W) in [-1, 1] -> (T, H, W, 3) uint8."""
    return ((gen.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()


def _patch_sqrtm() -> None:
    """cd-fvd calls scipy.linalg.sqrtm(..., disp=False) (returns (root, err)); scipy >= 1.16 dropped ``disp``."""
    import inspect

    import scipy.linalg

    if "disp" in inspect.signature(scipy.linalg.sqrtm).parameters:
        return
    orig = scipy.linalg.sqrtm

    def sqrtm(a, disp=True, **kw):
        root = orig(a, **kw)
        return root if disp else (root, 0.0)

    scipy.linalg.sqrtm = sqrtm
    import cdfvd.fvd as m

    if getattr(m, "scipy", None) is not None:
        m.scipy.linalg.sqrtm = sqrtm


class FVD:
    def __init__(self, device):
        _patch_sqrtm()
        from cdfvd import fvd

        self.ev = fvd.cdfvd("i3d", n_real="full", n_fake="full", device=str(device))

    def add(self, real: list[np.ndarray], fake: list[np.ndarray]) -> None:
        self.ev.add_real_stats(np.stack(real))
        self.ev.add_fake_stats(np.stack(fake))

    def compute(self) -> float:
        return float(self.ev.compute_fvd_from_stats())

    def reset(self) -> None:
        self.ev.empty_real_stats()
        self.ev.empty_fake_stats()


# ---------------------------------------------------------------------------- main
@torch.no_grad()
def generate_windows(policy, windows: list[dict], k: int, future: int, batch: int, seed: int, steps, cfg_scale, shift):
    """-> list of (gen (3, T, H, W) in [-1, 1], gt_future uint8 (T, 3, H, W)) in window order."""
    out = []
    n_lat = 1 + (future - 1) // 4
    for s in range(0, len(windows), batch):
        chunk = windows[s : s + batch]
        frames = torch.stack([trim(w, k, future) for w in chunk])  # (B, k + F, 3, H, W) uint8
        cond, _ = sampling.encode_windows(policy, frames, k)
        lat = sampling.generate(policy, cond, k, [w["caption"] for w in chunk], n_future_latents=n_lat,
                                steps=steps, cfg_scale=cfg_scale, shift=shift, seed=seed + s,
                                num_loops=getattr(policy, "loops", None))
        vids = sampling.decode(policy, lat)  # (B, 3, F, H, W)
        for v, f in zip(vids, frames):
            out.append((v.cpu(), f[k:].cpu()))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--ema", default=None, help="EMA safetensors (hub_backup.py); omit for the base trunk")
    p.add_argument("--loops", type=int, default=None, help="inference loop depth (looped models only)")
    p.add_argument("--splits", nargs="+", default=["val", "test", "test_unseen"])
    p.add_argument("--k", type=int, nargs="+", default=[0, 1, 5, 9, 17])
    p.add_argument("--n-recon", type=int, default=200)
    p.add_argument("--n-fvd", type=int, default=1000)
    p.add_argument("--fvd-k", type=int, nargs="*", default=[1, 17])
    p.add_argument("--n-judge", type=int, default=100)
    p.add_argument("--judge-k", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-model", default="/workspace/loop-dit/flux-action/outputs/weights/text_encoder",
                   help="Qwen3-VL directory for the prompt-adherence judge")
    args = p.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    policy, cfg = load_policy(args.config, args.ema, dev, args.loops)
    pcfg = policy.config
    splits_path = cfg["dataset_kwargs"]["splits"]
    hw = tuple(cfg["dataset_kwargs"].get("frame_hw") or ())
    meta = load_splits(splits_path)["meta"]
    future = pcfg.future_frames
    recon = Recon(dev)
    fvd = FVD(dev) if args.fvd_k else None
    # earlier invocations with the same --out (other splits / sample counts) stay in summary.json
    summary: dict[str, dict] = {f.stem: json.loads(f.read_text()) for f in sorted(out.glob("*_k*.json"))}
    for split in args.splits:
        n_max = max(args.n_recon, args.n_fvd if set(args.k) & set(args.fvd_k) else 0, args.n_judge)
        picks = fixed_windows(split, n_max, seed=args.seed, splits_path=splits_path)
        t0 = time.time()
        windows = _load_windows(picks, meta, hw or None)
        print(f"[{split}] {len(windows)} windows decoded in {time.time() - t0:.0f}s", flush=True)
        for k in args.k:
            name = f"{split}_k{k}"
            path = out / f"{name}.json"
            if path.exists():
                summary[name] = json.loads(path.read_text())
                print(f"[{name}] cached", flush=True)
                continue
            n_gen = args.n_fvd if k in args.fvd_k else args.n_recon
            if k == args.judge_k and not args.no_judge:
                n_gen = max(n_gen, args.n_judge)
            t0 = time.time()
            gens = generate_windows(policy, windows[:n_gen], k, future, args.batch, args.seed, args.steps, args.cfg, args.shift)
            gen_s = time.time() - t0
            rows = [recon(g, gt) for g, gt in gens[: args.n_recon]]
            res = {"split": split, "k": k, "n_recon": len(rows), "n_gen": len(gens), "gen_seconds": round(gen_s),
                   **{m: float(np.mean([r[m] for r in rows])) for m in ("psnr", "ssim", "lpips")},
                   **{m + "_std": float(np.std([r[m] for r in rows])) for m in ("psnr", "ssim", "lpips")}}
            if fvd is not None and k in args.fvd_k:
                fvd.reset()
                for s in range(0, len(gens), 64):
                    fvd.add([gt.permute(0, 2, 3, 1).numpy() for _, gt in gens[s : s + 64]],
                            [to_uint8_video(g) for g, _ in gens[s : s + 64]])
                res["fvd"] = fvd.compute()
                res["n_fvd"] = len(gens)
            if k == args.judge_k and not args.no_judge:
                from .judge import judge_prompt_adherence

                sub = windows[: args.n_judge]
                caps = [w["caption"] for w in sub]
                rng = random.Random(args.seed)
                swapped = []
                for w in sub:  # a caption from another task
                    others = [c["caption"] for c in windows if c["episode"] != w["episode"] and c["caption"] != w["caption"]]
                    swapped.append(rng.choice(others) if others else "")
                gens_sw = generate_windows(policy, [{**w, "caption": c} for w, c in zip(sub, swapped)], k, future,
                                           args.batch, args.seed, args.steps, args.cfg, args.shift)
                res["judge"] = judge_prompt_adherence(
                    [to_uint8_video(g) for g, _ in gens[: args.n_judge]], caps,
                    [to_uint8_video(g) for g, _ in gens_sw], swapped, fps=pcfg.fps, device=dev,
                    text_encoder_dir=args.judge_model)
            # qualitative grid: the first 8 windows, GT | generated
            gt8 = torch.stack([trim(w, k, future) for w in windows[:8]])
            gen8 = torch.stack([g for g, _ in gens[:8]])
            _save_grid(gt8, gen8, k, out / f"{name}_grid.mp4", pcfg.fps)
            path.write_text(json.dumps(res, indent=1))
            summary[name] = res
            print(f"[{name}] " + json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in res.items()
                                             if a not in ("split", "k")}), flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
