"""Capability eval: ldvflux.rollout's chained k = 1 rollout on never-trained cases (configs/eval_cases_capability.json),
three prompts per case (real / alt1 / alt2) × a few noise seeds.

A case is either **real** (an AgiBot World Beta episode in no split of the training data, or a test_unseen episode
of a held-out task: the context frame is the episode's frame before the anchor and the ground truth follows it) or
**generated** (``context_image``: an edited frame, labelled ``"source": "generated"`` in the cases file; no ground
truth). Seeds follow ldvflux.seed_sweep: seed s of case ci uses noise BASE(s) + ci with BASE(s) = 10000 s.

    cd flux-action && PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_eval --config ../configs/looped_ft_384_random_n1-8.json \
        --ema /dev/shm/w/looped/random_depth_n1-8_long/ema_step-9500.safetensors \
        --cases ../configs/eval_cases_capability.json --out ../results/capeval/rlong9500 --seeds 0 1 2 --depths 1 \
        --text-encoder outputs/weights/text_encoder --mirror /dev/shm/capeval/mirror --shard 0/2 --device cuda:0

Outputs: <out>/clips/<case>/<prompt>_N<d>_s<seed>.mp4 (1 + W*33 frames, 384x512, 15 fps), <out>/gt/<case>.mp4 and
<out>/gt/meta.json (real cases, ldvflux.rollout.write_gt), <out>/prompts.json, <out>/times_<shard>.json. Existing clips
are skipped (resumable); --shard i/n takes every n-th case so two GPUs can share one <out>.

Negative-prompt test: --prompts alt2 --neg-prompt real --neg-scale 3 adds 3 (v(alt2) - v(real)) to every step (see
ldvflux.sampling.generate); give each guidance setting its own --out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .depth_eval import FUTURE, HW, write_clip
from .rollout import FPS, load_episode_frames, rollout, write_gt
from .seed_sweep import SEED_STRIDE

PROMPTS = ("real", "alt1", "alt2")  # v1 cases; v2 cases use real / object / action / direction


def context_image(path: Path) -> np.ndarray:
    """An edited context frame -> uint8 (3, H, W) at the training resolution."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize(HW[::-1], Image.BICUBIC)
    return np.ascontiguousarray(np.asarray(img).transpose(2, 0, 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ema", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--depths", type=int, nargs="+", default=[1])
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--text-encoder", required=True)
    p.add_argument("--mirror", required=True, help="dir with videos/<split>/<episode>.mp4 (capeval fetch_eps.py)")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cases-only", nargs="*", default=None)
    p.add_argument("--prompts", nargs="+", default=None, help="prompt keys of the cases file (default: all of each case)")
    p.add_argument("--neg-prompt", default=None, help="the case's prompt key to use as the negative (e.g. real)")
    p.add_argument("--neg-scale", type=float, default=0.0)
    a = p.parse_args()

    from .eval_phase4 import load_policy

    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(a.cases).read_text())
    shard, n_shards = map(int, a.shard.split("/"))
    cases = [(ci, c) for ci, c in enumerate(spec["cases"])  # ci: index in the file (noise seed)
             if ci % n_shards == shard and (not a.cases_only or c["id"] in a.cases_only)]
    mirror = Path(a.mirror)
    policy, _ = load_policy(a.config, a.ema, torch.device(a.device), a.depths[0], a.text_encoder)
    n_total = 1 + a.windows * FUTURE
    prompts_p, times_p = out / "prompts.json", out / f"times_{shard}.json"
    times = json.loads(times_p.read_text()) if times_p.exists() else {}
    for s in a.seeds:
        for ci, case in cases:
            cid = case["id"]
            cdir = out / "clips" / cid
            cdir.mkdir(exist_ok=True)
            todo = [(pk, d) for pk in (a.prompts or case["prompts"]) for d in a.depths if not (cdir / f"{pk}_N{d}_s{s}.mp4").exists()]
            if not todo:
                continue
            if case["source"] == "generated":
                ctx = context_image(Path(a.cases).parent / case["context_image"])
            else:
                frames = load_episode_frames(case, n_total, None, mirror, None)
                if not (out / "gt" / f"{cid}.mp4").exists():
                    write_gt(out / "gt", case, frames, n_total)
                ctx = frames[0]
            allp = json.loads(prompts_p.read_text()) if prompts_p.exists() else {}
            prompts_p.write_text(json.dumps({**allp, cid: case["prompts"]}, indent=1))
            for pk, d in todo:
                policy.loops = d
                t0 = time.time()
                neg = case["prompts"][a.neg_prompt] if a.neg_prompt and a.neg_prompt != pk else None
                vid = rollout(policy, ctx, case["prompts"][pk], a.windows, s * SEED_STRIDE + ci, a.steps, a.cfg, a.shift, d,
                              neg, a.neg_scale)
                name = f"{pk}_N{d}_s{s}"
                write_clip(cdir / f"{name}.mp4", vid, FPS)
                times[f"{cid}/{name}"] = round(time.time() - t0, 1)
                times_p.write_text(json.dumps(times, indent=1))
                print(f"[seed {s} {cid} {pk} N{d}] {times[f'{cid}/{name}']} s", flush=True)
    print("capability eval done:", out, flush=True)


if __name__ == "__main__":
    main()
