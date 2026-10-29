"""Autoregressive long rollout ("Long Rollout" page): k = 1, W windows of 33 frames chained.

Window 1 starts from the real frame before the evaluation window (the gallery clip). Each later window starts from the
last frame the model itself generated, with the same prompt and fresh noise. For the gallery cases every prompt (the
real AgiBot annotation + the alternative prompts of the cases file) and every loop depth is rolled out.

    python -m ldvflux.rollout --config configs/looped_ft_384_r2.json --ema /dev/shm/ema/ema_loop4500_step-4500.safetensors \
        --cases configs/eval_cases_gallery8.json --out results/rollout/loop4500 --depths 1 2 4 8 16 --windows 4 \
        --text-encoder flux-action/outputs/weights/text_encoder --gt-out results/rollout/gt --device cuda:0
    (plain checkpoints: --depths 1; the base model: no --ema)

Outputs: <out>/clips/<case>/p<i>_N<d>.mp4 — 1 + W*33 frames at 15 fps, 384x512: the real context frame, then the W
generated windows; <out>/prompts.json {case: [prompts]}; <out>/times.json {clip: seconds}. Ground truth (once, shared
by every model): <gt-out>/<case>.mp4 with the same frame layout (black after the episode ends) and <gt-out>/meta.json
{case: {"gt_frames": n, "gt_s": n/15, "gt_full": bool}}. Existing clips are skipped (resumable, several processes may
share one <out> with different --depths).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import sampling
from .depth_eval import FUTURE, HW, STRIDE, resolve_caption, write_clip
from .stream import EpisodeFetcher, decode_frames, load_splits

FPS = 15.0


def load_episode_frames(case: dict, n_total: int, splits: dict | None, mirror: Path | None, fetcher: EpisodeFetcher | None) -> dict[int, np.ndarray]:
    """Frames anchor - STRIDE (context) and anchor + j*STRIDE for j < n_total - 1, as far as the episode goes."""
    idx = [case["anchor"] - STRIDE] + [case["anchor"] + j * STRIDE for j in range(n_total - 1)]
    if mirror is not None:
        data = (mirror / "videos" / case["split"] / f"{case['episode']}.mp4").read_bytes()
    else:
        ep = next(e for e in splits["splits"][case["split"]] if e["episode"] == str(case["episode"]))
        data = fetcher.fetch(ep)
    frames = decode_frames(data, set(idx), HW)
    if idx[0] not in frames or idx[1] not in frames:
        raise ValueError(f"{case['id']}: context/first frame missing")
    return {j: frames[i] for j, i in enumerate(idx) if i in frames}  # position in the rollout -> uint8 (3, H, W)


def write_gt(gt_out: Path, case: dict, frames: dict[int, np.ndarray], n_total: int) -> dict:
    gt_out.mkdir(parents=True, exist_ok=True)
    h, w = HW
    vid = np.zeros((n_total, h, w, 3), np.uint8)
    n = 0
    for j in range(n_total):
        if j in frames:
            vid[j] = frames[j].transpose(1, 2, 0)
            n = j + 1
        else:
            break
    write_clip(gt_out / f"{case['id']}.mp4", vid, FPS)
    meta_p = gt_out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta[case["id"]] = {"gt_frames": n, "gt_s": round(n / FPS, 1), "gt_full": n == n_total}
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta[case["id"]]


@torch.no_grad()
def rollout(policy, ctx: np.ndarray, prompt: str, windows: int, seed: int, steps: int, cfg: float, shift: float,
            loops: int | None, neg_prompt: str | None = None, neg_scale: float = 0.0) -> np.ndarray:
    """ctx uint8 (3, H, W) -> uint8 (1 + windows*FUTURE, H, W, 3): the context frame then the chained windows.
    ``neg_prompt`` / ``neg_scale``: negative-prompt guidance (ldvflux.sampling.generate), off by default."""
    from .eval_phase4 import to_uint8_video

    out = [ctx.transpose(1, 2, 0)[None]]
    n_lat = 1 + (FUTURE - 1) // 4
    for wi in range(windows):
        # (B=1, k + F, 3, H, W) uint8; the "future" slots only feed the (unused) future-latent encode
        frames = torch.from_numpy(np.stack([ctx] * (1 + FUTURE)))[None]
        cond, _ = sampling.encode_windows(policy, frames, 1)
        lat = sampling.generate(policy, cond, 1, [prompt], n_future_latents=n_lat, steps=steps, cfg_scale=cfg, shift=shift,
                                seed=seed + 1000 * wi, num_loops=loops,
                                neg_captions=[neg_prompt] if neg_prompt is not None else None, neg_scale=neg_scale)
        vid = to_uint8_video(sampling.decode(policy, lat)[0])  # (F, H, W, 3)
        out.append(vid)
        ctx = np.ascontiguousarray(vid[-1].transpose(2, 0, 1))
    return np.concatenate(out, 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ema", default=None)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gt-out", default=None, help="write the ground-truth rollouts here (skipped when they exist)")
    p.add_argument("--depths", type=int, nargs="+", default=[1])
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--text-encoder", default=None)
    p.add_argument("--mirror", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cases-only", nargs="*", default=None, help="case ids to run (default all)")
    a = p.parse_args()

    from .eval_phase4 import load_policy

    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.cases).read_text())["cases"]
    cases = [(ci, c) for ci, c in enumerate(cases) if not a.cases_only or c["id"] in a.cases_only]  # ci: index in the file (noise seed)
    mirror = Path(a.mirror) if a.mirror else None
    dev = torch.device(a.device)
    cfg = json.load(open(a.config))
    looped = cfg["policy"].get("loop") is not None
    depths = a.depths if looped else [1]
    policy, cfg = load_policy(a.config, a.ema, dev, depths[0] if looped else None, a.text_encoder)
    splits = None if mirror else load_splits(cfg["dataset_kwargs"]["splits"])
    fetcher = None if mirror else EpisodeFetcher()
    n_total = 1 + a.windows * FUTURE
    gt_out = Path(a.gt_out) if a.gt_out else None
    prompts_p, times_p = out / "prompts.json", out / "times.json"
    prompts_all = json.loads(prompts_p.read_text()) if prompts_p.exists() else {}
    times = json.loads(times_p.read_text()) if times_p.exists() else {}
    for ci, case in cases:
        cid = case["id"]
        cdir = out / "clips" / cid
        cdir.mkdir(exist_ok=True)
        caption = resolve_caption(case, splits, mirror)
        prompts = [caption] + list(case.get("alt_prompts", []))
        todo = [(pi, d) for pi in range(len(prompts)) for d in depths if not (cdir / f"p{pi}_N{d}.mp4").exists()]
        need_gt = gt_out is not None and not (gt_out / f"{cid}.mp4").exists()
        if not todo and not need_gt:
            continue
        frames = load_episode_frames(case, n_total, splits, mirror, fetcher)
        if need_gt:
            m = write_gt(gt_out, case, frames, n_total)
            print(f"[{cid}] ground truth: {m['gt_frames']} / {n_total} frames", flush=True)
        prompts_all[cid] = prompts
        prompts_p.write_text(json.dumps(prompts_all, indent=1))
        ctx = frames[0]
        for pi, d in todo:
            policy.loops = d if looped else None
            t0 = time.time()
            vid = rollout(policy, ctx, prompts[pi], a.windows, a.seed + ci, a.steps, a.cfg, a.shift, d if looped else None)
            write_clip(cdir / f"p{pi}_N{d}.mp4", vid, FPS)
            # several processes (different --depths) share one times.json: merge on every write
            times = {**(json.loads(times_p.read_text()) if times_p.exists() else {}), **times}
            times[f"{cid}/p{pi}_N{d}"] = round(time.time() - t0, 1)
            times_p.write_text(json.dumps(times, indent=1))
            print(f"[{cid} p{pi} N{d}] {times[f'{cid}/p{pi}_N{d}']} s  {prompts[pi][:60]!r}", flush=True)
    print("rollout done:", out, flush=True)


if __name__ == "__main__":
    main()
