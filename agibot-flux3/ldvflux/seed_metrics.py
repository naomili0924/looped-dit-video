"""Per-depth numbers of the seed sweep (ldvflux.seed_sweep) and the 5 x 5 seed-by-depth grid videos of the page.

Per clip (case, prompt, seed, depth) and window w (frames 1 + 33 (w-1) ... 33 w of the 133-frame rollout):
  gt_psnr / gt_lpips  real prompt only, frames inside the episode, against the ground-truth clip as rollout.write_gt
                      stores it (mp4 round trip of the mirror frames); PSNR = 10 log10(255^2 / MSE) per frame, then
                      the frame mean; LPIPS (alex, [-1, 1]) per frame, then the plain frame mean
  motion              mean |frame_t - frame_t-1| (0-255) over the frame pairs inside the window
  div_lpips           mean LPIPS between the window's frames of two seeds, over all seed pairs (every prompt)
Checked on the regenerated step-3000 N1 clips against the numbers of the first (lost) runner: gt_psnr, gt_lpips, their
seed std and motion agree to <= 0.004; div_lpips w2-w4 to <= 0.0016, w1 is 0.007 lower (0.180 vs 0.187).
Aggregates per depth (summary.json): the mean over cases per seed, then the mean (and std, "_seed_std") over seeds;
div_lpips is the mean over every (case, prompt). Raw per-clip numbers are kept in raw.json.

    python -m ldvflux.seed_metrics --sweep results/seedsweep/rlong9500 --cases configs/eval_cases_gallery8.json \
        --mirror /workspace/loop_dit_eval/mirror [--grids results/seedsweep/grids --tag rlong9500]
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .depth_eval import FUTURE, read_clip, write_clip
from .rollout import load_episode_frames

CELL = (144, 192)  # grid cell (h, w)


def windows(n_total: int) -> list[slice]:
    return [slice(1 + w * FUTURE, 1 + (w + 1) * FUTURE) for w in range((n_total - 1) // FUTURE)]


def frame_psnr(a: np.ndarray, b: np.ndarray) -> float:
    d = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).reshape(len(a), -1).mean(1)
    return float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(d, 1e-6))))


def motion(clip: np.ndarray, sl: slice) -> float:
    return float(np.abs(np.diff(clip[sl].astype(np.float32), axis=0)).mean())


class Lpips:
    """LPIPS (alex) per frame pair, plain mean over the frames (depth_eval.Lpips averages chunk means instead)."""

    def __init__(self, device):
        import lpips

        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)
        return float(torch.cat([self.net(t(a[s : s + 16]), t(b[s : s + 16])).flatten() for s in range(0, len(a), 16)]).mean())


def gt_frames(cases: list[dict], mirror: Path, n_total: int, cache: Path) -> dict[str, np.ndarray]:
    out = {}
    for c in cases:
        p = cache / f"{c['id']}.npy"
        if not p.exists():
            fr = load_episode_frames(c, n_total, None, mirror, None)
            n = next(j for j in range(n_total + 1) if j not in fr)
            full = np.zeros((n_total,) + fr[0].transpose(1, 2, 0).shape, np.uint8)
            for j in range(n):
                full[j] = fr[j].transpose(1, 2, 0)
            mp4 = cache / f"{c['id']}.mp4"
            write_clip(mp4, full, 15.0)  # the stored ground truth of the rollout pages
            np.save(p, read_clip(mp4)[:n])
        out[c["id"]] = np.load(p)
    return out


def summarize(raw: dict, depths: list[int], seeds: list[int], n_win: int) -> dict:
    res = {}
    for d in depths:
        r = {"n_clips": 0, "seeds": seeds}
        rows = [v for v in raw["clips"] if v["N"] == d and v["seed"] in seeds]
        r["n_clips"] = len(rows)
        for key in ("gt_psnr", "gt_lpips"):
            per_seed = []
            for s in seeds:
                per_case = [[v[key][w] for v in rows if v["seed"] == s and v["p"] == 0 and v[key][w] is not None] for w in range(n_win)]
                per_seed.append([float(np.mean(x)) if x else np.nan for x in per_case])
            ps = np.array(per_seed)
            r[key] = [round(float(x), 4) for x in np.nanmean(ps, 0)]
            r[key + "_seed_std"] = [round(float(x), 4) for x in np.nanstd(ps, 0)]
        r["motion"] = [round(float(np.mean([v["motion"][w] for v in rows])), 3) for w in range(n_win)]
        div = [v for v in raw["div"] if v["N"] == d]
        r["div_lpips"] = [round(float(np.mean([v["lpips"][w] for v in div])), 4) for w in range(n_win)] if div else None
        res[str(d)] = r
    return res


def compose_grid(clips: dict[tuple[int, int], Path], seeds: list[int], depths: list[int], n_total: int, out: Path) -> None:
    """rows = seeds, columns = depths; a missing clip stays black."""
    h, w = CELL
    g = np.zeros((n_total, h * len(seeds), w * len(depths), 3), np.uint8)
    for (si, di), p in clips.items():
        small = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               capture_output=True, check=True).stdout
        g[:, si * h : (si + 1) * h, di * w : (di + 1) * w] = np.frombuffer(small, np.uint8).reshape(-1, h, w, 3)[:n_total]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{g.shape[2]}x{g.shape[1]}", "-r", "15",
                    "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                   input=g.tobytes(), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--mirror", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--grids", default=None, help="write <grids>/<tag>_<case>_p<i>.mp4")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    sweep = Path(a.sweep)
    cases = json.loads(Path(a.cases).read_text())["cases"]
    n_total = 1 + a.windows * FUTURE
    wins = windows(n_total)
    clip = lambda cid, pi, d, s: sweep / "clips" / cid / f"p{pi}_N{d}_s{s}.mp4"
    n_prompts = {c["id"]: 1 + len(c.get("alt_prompts", [])) for c in cases}

    if a.grids:
        for c in cases:
            for pi in range(n_prompts[c["id"]]):
                have = {(si, di): clip(c["id"], pi, d, s) for si, s in enumerate(a.seeds) for di, d in enumerate(a.depths)
                        if clip(c["id"], pi, d, s).exists()}
                if have:
                    compose_grid(have, a.seeds, a.depths, n_total, Path(a.grids) / f"{a.tag}_{c['id']}_p{pi}.mp4")
        print("grids done", flush=True)
    if a.no_metrics:
        return
    # metrics only for depths where every seed of every clip exists
    depths = [d for d in a.depths if all(clip(c["id"], pi, d, s).exists() for c in cases for pi in range(n_prompts[c["id"]]) for s in a.seeds)]
    if not depths:
        print("no complete depth yet")
        return
    cache = sweep.parent / "gt_cache"
    cache.mkdir(exist_ok=True)
    gts = gt_frames(cases, Path(a.mirror), n_total, cache)
    lp = Lpips(torch.device(a.device))
    raw_p = sweep / "raw.json"
    raw = json.loads(raw_p.read_text()) if raw_p.exists() else {"clips": [], "div": []}
    done_c = {(v["case"], v["p"], v["N"], v["seed"]) for v in raw["clips"]}
    done_d = {(v["case"], v["p"], v["N"]) for v in raw["div"]}
    for d in depths:
        for c in cases:
            cid, gt = c["id"], gts[c["id"]]
            for pi in range(n_prompts[cid]):
                vids = {}
                for s in a.seeds:
                    vids[s] = read_clip(clip(cid, pi, d, s))
                    if (cid, pi, d, s) in done_c:
                        continue
                    v = vids[s]
                    row = {"case": cid, "p": pi, "N": d, "seed": s, "motion": [motion(v, sl) for sl in wins], "gt_psnr": [], "gt_lpips": []}
                    for sl in wins:
                        hi = min(sl.stop, len(gt))
                        ok = pi == 0 and hi > sl.start
                        row["gt_psnr"].append(frame_psnr(v[sl.start : hi], gt[sl.start : hi]) if ok else None)
                        row["gt_lpips"].append(lp(v[sl.start : hi], gt[sl.start : hi]) if ok else None)
                    raw["clips"].append(row)
                if (cid, pi, d) not in done_d:
                    pairs = list(itertools.combinations(a.seeds, 2))
                    raw["div"].append({"case": cid, "p": pi, "N": d,
                                       "lpips": [float(np.mean([lp(vids[x][sl], vids[y][sl]) for x, y in pairs])) for sl in wins]})
                raw_p.write_text(json.dumps(raw))
            print(f"metrics N{d} {cid} done", flush=True)
    summary = summarize(raw, depths, a.seeds, len(wins))
    (sweep / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary)[:2000])


if __name__ == "__main__":
    main()
