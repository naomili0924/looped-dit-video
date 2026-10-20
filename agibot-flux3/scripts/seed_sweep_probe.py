"""Seed-sweep probe for results/seedsweep/README.md: ground-truth motion per window, and LPIPS between prompts at the same
seed (N 1, steps 3000 and 9500). Run from agibot-flux3 with PYTHONPATH=$PWD flux-action/.venv/bin/python scripts/seed_sweep_probe.py"""
import json, itertools
from pathlib import Path
import numpy as np, torch
from ldvflux.depth_eval import read_clip
from ldvflux.seed_metrics import windows, motion, Lpips
SW = Path("results/seedsweep"); W = windows(133)
cases = json.loads(Path("configs/eval_cases_gallery8.json").read_text())["cases"]
gm = []
for c in cases:
    g = np.load(SW / "gt_cache" / f"{c['id']}.npy")
    gm.append([motion(g, sl) if min(sl.stop, len(g)) - sl.start > 1 else np.nan for sl in W])
    print(c["id"], "gt frames", len(g))
print("GT motion per window", np.round(np.nanmean(gm, 0), 3).tolist())
lp = Lpips(torch.device("cuda:0"))
for st in (3000, 9500):
    root = SW / ("verify3000" if st == 3000 else f"rlong{st}")
    pr, sd = [], []
    for c in cases:
        for s in range(5):
            v = {p: read_clip(root / "clips" / c["id"] / f"p{p}_N1_s{s}.mp4") for p in range(3)}
            pr.append([np.mean([lp(v[0][sl], v[p][sl]) for p in (1, 2)]) for sl in W])
    print(st, "N1 LPIPS between prompts (same seed)", np.round(np.mean(pr, 0), 4).tolist(), flush=True)
