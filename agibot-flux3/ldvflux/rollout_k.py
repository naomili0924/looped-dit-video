"""Autoregressive rollout with k context frames: chain 33-frame windows, each conditioned on the last k frames the model
itself generated (from the eval-h100 branch; the long-rollout and seed-sweep pages use ldvflux.rollout).

Round 0 is the normal eval window (real context, same noise as ldvflux.depth_eval, so it reproduces the gallery clip);
round r > 0 takes the last k output frames of round r - 1 as context and the same prompt. With k = 1 each round adds
33 frames (2.2 s at 15 fps): 3 rounds = 6.6 s, 4 rounds = 8.8 s. The ground truth is read as far as the episode goes.

    python -m ldvflux.rollout_k --config <cfg> [--ema <EMA>] --cases configs/eval_cases_gallery8.json --mirror <eval-splits> \
        --text-encoder <Qwen3-VL dir> --depths 1 4 --k 1 --rounds 4 --out results/rollout/loop2500
Writes clips/<case>/k<k>/{gt.mp4, p<prompt>_N<d>.mp4} (context frame + all rounds) and manifest[_<tag>].json.
--prompts selects the real caption (0) and/or the cases file's alt_prompts (1, 2, ...); all prompts share the noise.
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
from .stream import decode_frames, window_frame_indices


def load_long(case: dict, k: int, rounds: int, mirror: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """-> (context uint8 (k, 3, H, W), ground truth uint8 (<= rounds * 33, 3, H, W), caption)."""
    idx = window_frame_indices(case["anchor"], k, STRIDE, FUTURE * rounds)
    data = (mirror / "videos" / case["split"] / f"{case['episode']}.mp4").read_bytes()
    frames = decode_frames(data, set(idx), HW)
    ctx = np.stack([frames[i] for i in idx[:k]])
    gt = []
    for i in idx[k:]:
        if i not in frames:  # episode ended
            break
        gt.append(frames[i])
    return ctx, np.stack(gt), resolve_caption(case, None, mirror)


def run(args: argparse.Namespace) -> None:
    from .eval_phase4 import load_policy, to_uint8_video

    out = Path(args.out)
    mirror = Path(args.mirror)
    dev = torch.device(args.device)
    cfg = json.load(open(args.config))
    policy, cfg = load_policy(args.config, args.ema, dev, args.depths[0] if cfg["policy"].get("loop") else None,
                              args.text_encoder)
    looped = cfg["policy"].get("loop") is not None
    depths = args.depths if looped else [1]
    n_lat = 1 + (FUTURE - 1) // 4
    pad = torch.zeros(FUTURE, 3, *HW, dtype=torch.uint8)
    cases = json.loads(Path(args.cases).read_text())["cases"]
    man = {"config": args.config, "ema": args.ema, "depths": depths, "k": args.k, "rounds": args.rounds, "seed": args.seed,
           "cases": []}
    for ci, case in enumerate(cases):
        cid = case.get("id", f"case{ci:02d}")
        if args.only and cid not in args.only:
            continue
        ctx0, gt, caption = load_long(case, args.k, args.rounds, mirror)
        cdir = out / "clips" / cid / f"k{args.k}"
        cdir.mkdir(parents=True, exist_ok=True)
        write_clip(cdir / "gt.mp4", np.concatenate([ctx0[-1:], gt]).transpose(0, 2, 3, 1))
        prompts = [caption] + list(case.get("alt_prompts", []))
        rec = {"id": cid, "prompts": prompts, "gt_frames": int(len(gt)), "seconds": {}}
        for pi, d in [(pi, d) for pi in args.prompts for d in depths]:
            path = cdir / f"p{pi}_N{d}.mp4"
            if path.exists():
                continue
            policy.loops = d if looped else None
            ctx, chunks, t0 = torch.from_numpy(ctx0), [], time.time()
            for r in range(args.rounds):
                frames = torch.cat([ctx, pad])[None]  # (1, k + 33, 3, H, W); the future part is unused
                cond, _ = sampling.encode_windows(policy, frames, args.k)
                # round 0 = depth_eval's noise (seed + case index); later rounds get fresh noise
                lat = sampling.generate(policy, cond, args.k, [prompts[pi]], n_future_latents=n_lat, steps=args.steps,
                                        cfg_scale=args.cfg, shift=args.shift, seed=args.seed + ci + 1000 * r,
                                        num_loops=getattr(policy, "loops", None))
                vid = to_uint8_video(sampling.decode(policy, lat)[0])  # (33, H, W, 3)
                chunks.append(vid)
                ctx = torch.from_numpy(vid[-args.k:].transpose(0, 3, 1, 2).copy())
                print(f"[{cid} p{pi} N{d}] round {r + 1}/{args.rounds} {time.time() - t0:.0f}s", flush=True)
            rec["seconds"][f"p{pi}_N{d}"] = round(time.time() - t0, 1)
            write_clip(path, np.concatenate([ctx0[-1:].transpose(0, 2, 3, 1)] + chunks))
        man["cases"].append(rec)
        (out / (f"manifest_{args.tag}.json" if args.tag else "manifest.json")).write_text(json.dumps(man, indent=1))
    print("rollout done:", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ema")
    p.add_argument("--cases", required=True)
    p.add_argument("--only", nargs="*", help="case ids to run (default all)")
    p.add_argument("--mirror", required=True)
    p.add_argument("--text-encoder", required=True)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 4])
    p.add_argument("--prompts", type=int, nargs="+", default=[0], help="0 = real caption, i > 0 = alt_prompts[i - 1]")
    p.add_argument("--tag", default="", help="manifest suffix, for parallel jobs writing into the same --out")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    run(p.parse_args())


if __name__ == "__main__":
    main()
