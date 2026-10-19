"""Seed sweep of the long rollout ("Random-Depth Seed Sweep" page): ldvflux.rollout's chained k = 1 rollout, repeated
for several noise seeds per (case, prompt, loop depth).

The rollout itself is ldvflux.rollout.rollout, unchanged. Seed s of case ci uses noise seed BASE(s) + ci (+ 1000 per
window inside rollout), with BASE(s) = 10000 s (SEED_STRIDE). Seed 0 is therefore the noise of the single-seed Long Rollout
pages; every checkpoint and depth shares the same seeds.

    cd flux-action && PYTHONPATH=.. .venv/bin/python -m ldvflux.seed_sweep --config ../configs/looped_ft_384_random_n1-8.json \
        --ema /dev/shm/ema/ema_rlong_step-9500.safetensors --cases ../configs/eval_cases_gallery8.json \
        --out ../results/seedsweep/rlong9500 --seeds 0 1 2 3 4 --depths 1 2 4 8 16 \
        --text-encoder outputs/weights/text_encoder --mirror /workspace/loop_dit_eval/mirror

Outputs: <out>/clips/<case>/p<i>_N<d>_s<seed>.mp4 (1 + 4*33 frames, 384x512, 15 fps), <out>/prompts.json,
<out>/times.json {"<case>/p<i>_N<d>_s<seed>": seconds}. Existing clips are skipped (resumable). Loop order: seed, then
case, prompt, depth, so a seed finishes for the whole gallery before the next one starts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .depth_eval import FUTURE, resolve_caption, write_clip
from .rollout import FPS, load_episode_frames, rollout

SEED_STRIDE = 10000  # recovered: seed 1 of step 3000 matches noise 10000 + ci (36 dB vs the published grid row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ema", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed-stride", type=int, default=SEED_STRIDE)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--prompts", type=int, nargs="+", default=None, help="prompt indices (default all)")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--text-encoder", required=True)
    p.add_argument("--mirror", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cases-only", nargs="*", default=None)
    a = p.parse_args()

    from .eval_phase4 import load_policy

    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.cases).read_text())["cases"]
    cases = [(ci, c) for ci, c in enumerate(cases) if not a.cases_only or c["id"] in a.cases_only]  # ci: index in the file (noise seed)
    mirror = Path(a.mirror)
    policy, _ = load_policy(a.config, a.ema, torch.device(a.device), a.depths[0], a.text_encoder)
    n_total = 1 + a.windows * FUTURE
    prompts_p, times_p = out / "prompts.json", out / "times.json"
    prompts_all = json.loads(prompts_p.read_text()) if prompts_p.exists() else {}
    times = json.loads(times_p.read_text()) if times_p.exists() else {}
    ctx_cache: dict[str, object] = {}
    for s in a.seeds:
        for ci, case in cases:
            cid = case["id"]
            cdir = out / "clips" / cid
            cdir.mkdir(exist_ok=True)
            prompts = [resolve_caption(case, None, mirror)] + list(case.get("alt_prompts", []))
            pis = a.prompts if a.prompts is not None else range(len(prompts))
            todo = [(pi, d) for pi in pis for d in a.depths if not (cdir / f"p{pi}_N{d}_s{s}.mp4").exists()]
            if not todo:
                continue
            prompts_all[cid] = prompts
            prompts_p.write_text(json.dumps(prompts_all, indent=1))
            if cid not in ctx_cache:
                ctx_cache[cid] = load_episode_frames(case, n_total, None, mirror, None)[0]
            for pi, d in todo:
                policy.loops = d
                t0 = time.time()
                vid = rollout(policy, ctx_cache[cid], prompts[pi], a.windows, s * a.seed_stride + ci, a.steps, a.cfg, a.shift, d)
                name = f"p{pi}_N{d}_s{s}"
                write_clip(cdir / f"{name}.mp4", vid, FPS)
                times[f"{cid}/{name}"] = round(time.time() - t0, 1)
                times_p.write_text(json.dumps(times, indent=1))
                print(f"[seed {s} {cid} p{pi} N{d}] {times[f'{cid}/{name}']} s", flush=True)
    print("seed sweep done:", out, flush=True)


if __name__ == "__main__":
    main()
