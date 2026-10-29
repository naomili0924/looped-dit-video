"""Judge-free direction check for the capability eval's `direction` prompts: does the clip move the way the prompt says?

Farneback optical flow between consecutive frames of window 1 (the 33 frames after the context frame, at half
resolution). Each case's direction prompt is tagged by hand (EXPECT) as one of
  ego  +x / -x   the robot turns: the whole scene pans (median flow), "turn right" -> scene moves left (-x)
  obj  +x / -x / -y / +y   an object or arm moves that way: the moving pixels after removing the median (camera) flow
Prompts with no clear image-plane direction (push the oven back, turn on the faucet, push the cart forward) are left out.
Per clip: the summed displacement d (ego: median flow; obj: mean residual flow of the 5% fastest pixels per frame pair),
then `proj` = d . e / |d| in [-1, 1] (e = expected unit direction) and `along` = d . e in pixels. The control is the same
case and seed with the real prompt (cfg4 run): the direction prompt should score higher than the real prompt does.

    .venv/bin/python ../scripts/capeval/flow_dir.py ../results/capeval/negprompt cfg4 neg4 neg7 cfg4neg3 cfg7
Writes <root>/flow_dir.json {arm: {clip: {...}}} and prints per-arm means and paired gains over the real-prompt control.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

EXPECT = {  # case -> (kind, axis vector (x, y) in image coordinates, y down)
    "pour_dressing": ("obj", (0, -1)), "scoop_tapioca": ("obj", (0, -1)), "stir_coffee": ("obj", (0, -1)),
    "twist_cap": ("obj", (-1, 0)), "scan_juice": ("obj", (1, 0)), "stamp_form": ("obj", (-1, 0)),
    "press_ricecooker": ("obj", (-1, 0)), "carry_bag": ("obj", (-1, 0)), "move_guard": ("obj", (-1, 0)),
    "cut_dough": ("obj", (-1, 0)), "knead_dough": ("obj", (1, 0)), "shake_decanter": ("obj", (-1, 0)),
    "untie_curtain": ("obj", (1, 0)), "tie_tieback": ("obj", (-1, 0)), "heldout_toast": ("obj", (1, 0)),
    "heldout_water": ("obj", (1, 0)), "heldout_curtains": ("obj", (-1, 0)),
    "clap": ("obj", (0, 1)), "wave": ("obj", (0, 1)),
    "base_forward": ("ego", (-1, 0)), "base_turn": ("ego", (1, 0)), "heldout_menu": ("ego", (1, 0)),
}
FRAMES = 34  # context frame + window 1


def frames(p: Path) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(str(p)), []
    while len(out) < FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    return out


def displacement(p: Path, kind: str) -> np.ndarray:
    fs, d = frames(p), np.zeros(2)
    for a, b in zip(fs[1:-1], fs[2:]):  # skip the real context -> generated seam
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0).reshape(-1, 2)
        med = np.median(fl, 0)
        if kind == "ego":
            d += med
        else:
            r = fl - med
            m = np.linalg.norm(r, axis=1)
            d += r[m >= np.quantile(m, 0.95)].mean(0)
    return d


def one(job):
    arm, clip, path, kind, e = job
    d = displacement(path, kind)
    along = float(d @ np.array(e))
    return arm, clip, {"d": d.round(2).tolist(), "along": round(along, 2), "proj": round(along / (np.linalg.norm(d) + 1e-6), 3)}


def main():
    root, arms = Path(sys.argv[1]), sys.argv[2:]
    jobs = []
    for arm in arms + ["cfg4@real"]:
        a, pk = arm.split("@") if "@" in arm else (arm, "direction")
        for case, (kind, e) in EXPECT.items():
            for p in sorted((root / a / "clips" / case).glob(f"{pk}_N1_s*.mp4")):
                jobs.append((arm, f"{case}/{p.stem}", p, kind, e))
    res = {}
    with ProcessPoolExecutor(64) as ex:
        for arm, clip, r in ex.map(one, jobs):
            res.setdefault(arm, {})[clip] = r
    (root / "flow_dir.json").write_text(json.dumps(res, indent=1))
    ctrl = {k.replace("real_", "direction_"): v for k, v in res["cfg4@real"].items()}
    print(f"{'cfg4@real':10s} n={len(ctrl):3d}  proj {np.mean([v['proj'] for v in ctrl.values()]):+.3f}  (control)")
    for arm in arms:
        r = res.get(arm, {})
        proj = np.array([v["proj"] for v in r.values()])
        gain = np.array([v["proj"] - ctrl[k]["proj"] for k, v in r.items() if k in ctrl])
        boot = np.percentile([np.random.default_rng(i).choice(gain, len(gain)).mean() for i in range(4000)], [2.5, 97.5])
        print(f"{arm:10s} n={len(r):3d}  proj {proj.mean():+.3f}  right-way {np.mean(proj > 0):.2f}  "
              f"gain vs real prompt {gain.mean():+.3f} [{boot[0]:+.3f}, {boot[1]:+.3f}]")


if __name__ == "__main__":
    main()
