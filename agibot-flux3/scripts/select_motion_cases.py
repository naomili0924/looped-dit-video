#!/usr/bin/env python3
"""Select "background-easy, motion-hard" evaluation cases from a split using only the real clips.

Criteria (RAFT flow of the real 33 future frames at k = 1, camera motion = per-frame median flow):
  camera        mean |median flow| <= --max-camera px/frame      (static head camera: background is a copy)
  moving_frac   fraction of pixels with compensated flow > 1 px >= --min-moving
  object        mean compensated flow over those pixels >= --min-object px/frame
  class         'hard' when the region-mean motion of the first and last third disagree (cosine < --hard-cos), else 'easy'

    python scripts/select_motion_cases.py --split val --n-scan 120 --n-keep 50 --out configs/eval_cases_val50.json \
        [--mirror <eval-splits dir>] [--device cuda:0]
Scans the first --n-scan fixed picks of the split (fixed_windows, seed 0), keeps up to --n-keep that pass, half hard / half
easy when possible. Captions are not written to the file (resolved at run time) unless --with-captions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldvflux.depth_eval import FUTURE, HW, STRIDE, load_case  # noqa: E402
from ldvflux.motion_eval import compensate_camera, flow, load_raft  # noqa: E402
from ldvflux.stream import EpisodeFetcher, fixed_windows, load_splits  # noqa: E402


def stats(raft, frames: np.ndarray) -> dict:
    """frames uint8 (T, 3, H, W) -> camera / moving_frac / object / direction cosine."""
    f = flow(raft, frames.transpose(0, 2, 3, 1))
    med = np.median(f.reshape(len(f), 2, -1), axis=2)
    cam = float(np.linalg.norm(med, axis=1).mean())
    fc = compensate_camera(f)
    mag = np.linalg.norm(fc, axis=1)
    mask = mag > 1.0
    obj = float(mag[mask].mean()) if mask.any() else 0.0
    v = [fc[t][:, mask[t]].mean(1) if mask[t].sum() > 20 else np.zeros(2) for t in range(len(fc))]
    a, b = np.mean(v[: len(v) // 3], axis=0), np.mean(v[-(len(v) // 3):], axis=0)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    return {"camera_px_per_frame": round(cam, 3), "moving_frac": round(float(mask.mean()), 4),
            "object_px_per_frame": round(obj, 3), "direction_change_cos": round(cos, 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--n-scan", type=int, default=120)
    p.add_argument("--n-keep", type=int, default=50)
    p.add_argument("--max-camera", type=float, default=0.3)
    p.add_argument("--min-moving", type=float, default=0.05)
    p.add_argument("--min-object", type=float, default=1.5)
    p.add_argument("--hard-cos", type=float, default=0.5)
    p.add_argument("--splits", default="/workspace/loop-dit/data/splits.json")
    p.add_argument("--mirror", default=None, help="local snapshot of Jinyan0924/agibot-flux3-eval-splits")
    p.add_argument("--device", default="cpu")
    p.add_argument("--small", action="store_true")
    p.add_argument("--with-captions", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    raft = load_raft(a.device, large=not a.small)
    mirror = Path(a.mirror) if a.mirror else None
    if mirror is not None:
        fixed = sorted((mirror / "fixed").glob(f"{a.split}_seed0_n*.json"), key=lambda q: -int(q.stem.split("_n")[-1]))[0]
        picks = [{"split": a.split, "episode": r["episode"], "task": r["task"], "anchor": r["anchor"], "caption": r["caption"]}
                 for r in json.loads(fixed.read_text())["picks"]][: a.n_scan]
        splits, fetcher = None, None
    else:
        splits = load_splits(a.splits)
        picks = [{"split": a.split, "episode": x["episode"]["episode"], "task": x["episode"]["task"], "anchor": x["window"]["anchor"],
                  "caption": x["window"]["caption"]} for x in fixed_windows(a.split, a.n_scan, seed=0, splits_path=a.splits)]
        fetcher = EpisodeFetcher()
    hard, easy, scanned = [], [], []
    for i, pk in enumerate(picks):
        try:
            w = load_case(pk, 1, splits, mirror, fetcher)
        except Exception as e:  # noqa: BLE001
            print(f"skip {pk['episode']}@{pk['anchor']}: {e}", flush=True)
            continue
        s = stats(raft, w["frames"][1 : 1 + FUTURE])
        rec = {"id": f"{a.split}{i:03d}_{pk['episode']}", "split": a.split, "task": pk["task"], "episode": pk["episode"],
               "anchor": pk["anchor"], "stats": s}
        if a.with_captions:
            rec["caption"] = pk["caption"]
        ok = s["camera_px_per_frame"] <= a.max_camera and s["moving_frac"] >= a.min_moving and s["object_px_per_frame"] >= a.min_object
        rec["class"] = ("hard" if s["direction_change_cos"] < a.hard_cos else "easy") if ok else "rejected"
        scanned.append(rec)
        (hard if rec["class"] == "hard" else easy if rec["class"] == "easy" else []).append(rec)
        print(f"[{i:3d}] {rec['class']:8s} cam {s['camera_px_per_frame']:.2f} mov {s['moving_frac']:.3f} obj {s['object_px_per_frame']:.2f} "
              f"dir {s['direction_change_cos']:+.2f}  {pk['caption'][:60]}", flush=True)
    half = a.n_keep // 2
    keep = hard[:half] + easy[: a.n_keep - min(half, len(hard))]
    if len(keep) < a.n_keep:
        keep = (hard + easy)[: a.n_keep]
    Path(a.out).write_text(json.dumps({"note": f"{a.split}: first {a.n_scan} fixed picks scanned, {len(hard)} hard / {len(easy)} easy pass "
                                                f"(camera <= {a.max_camera}, moving >= {a.min_moving}, object >= {a.min_object}, hard if dir cos < {a.hard_cos})",
                                        "cases": keep, "scanned": scanned}, indent=1))
    print(f"kept {len(keep)} ({sum(c['class'] == 'hard' for c in keep)} hard) -> {a.out}")


if __name__ == "__main__":
    main()
