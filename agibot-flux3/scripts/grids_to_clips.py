#!/usr/bin/env python3
"""Split an eval_phase4 grid video (8 rows of windows x [GT | generated]) into the clips layout that
ldvflux.clip_eval reads (win<i>_gen.mp4 / win<i>_gt.mp4 + meta.json), for analysing the already-saved 8-window grids.

    scripts/grids_to_clips.py results/ft_step6000/test_k1_grid.mp4 --k 1 --out results/ft_step6000/clips/test_k1
"""

import argparse
import json
import subprocess
from pathlib import Path

CW, CH = 512, 384

p = argparse.ArgumentParser()
p.add_argument("grid")
p.add_argument("--k", type=int, required=True, help="context frames at the start of the grid (dropped)")
p.add_argument("--out", required=True)
p.add_argument("--fps", default="15")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for i in range(8):
    for side, name in ((0, "gt"), (1, "gen")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.grid, "-vf",
                        f"select=gte(n\\,{a.k}),setpts=N/FRAME_RATE/TB,crop={CW}:{CH}:{side * CW}:{i * CH}",
                        "-r", a.fps, "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv444p",
                        str(out / f"win{i:03d}_{name}.mp4")], check=True)
(out / "meta.json").write_text(json.dumps({"source_grid": a.grid, "k": a.k, "fps": float(a.fps), "note": "from the 8-window grid",
                                           "windows": [{"i": i} for i in range(8)]}, indent=1))
print(f"wrote 8 windows to {out}")
