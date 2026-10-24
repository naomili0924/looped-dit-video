"""Dynamics and video quality of the capability eval clips (ldvflux.cap_eval), per clip and for the ground truth.

The judge in ldvflux.cap_judge asks whether the instruction was carried out. This module measures what it does not:
how things move and how the frames look. All metrics are on the generated frames 1..132 (frame 0 is the real
context frame) and on the same frames of the ground truth, so every number has a real-footage reference.

Flow metrics (RAFT large at 256 x 192, camera motion removed with motion_eval.compensate_camera):
  motion     mean |flow| per frame pair, px at 512 x 384 (arms and objects moving relative to the scene)
  jerk       mean |flow_t+1 - flow_t| / motion: how abruptly motion changes (lower = smoother)
  warp_err   mean |frame_t - frame_t+1 warped back by the flow| (0-255): change that motion does not explain,
             i.e. flicker, popping and smearing (lower = more temporally consistent)
  seam       warp_err at the 3 joins between chained windows / median warp_err of the frames around them
             (1 = joins invisible)
  persist    motion in window 4 / motion in window 1 (does the action keep going or freeze)
  vs GT      real prompt, real cases: motion_eval.compare of the flow against the real continuation
             (dir_cos, region_cos, mag_ratio) over the frames the episode has

Frame quality (512 x 384, the model's native output; resolution is the same at every depth):
  sharp      variance of the Laplacian of the grey image / that of the context frame (1 = as sharp as the real
             first frame; blur lowers it, noise raises it)
  hf         share of spectral energy above 0.25 cycles/px / that of the context frame (fine detail)
  sharp_w4   sharp over window 4 only (detail lost along the chain)

Ratings (--rate): Qwen3-VL-32B sees 16 frames, no instruction, and rates 1-5 object dynamics, arm motion and
visual quality.

    cd flux-action && PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_quality --run ../results/capeval/rlong9500 \
        --device cuda:0 --shard 0/2            # metrics -> <run>/quality_0.json
    ... --rate --judge /dev/shm/w/qwen3vl32b --shard 0/2   # ratings -> <run>/rating_0.json
    ... --summary                                       # -> <run>/quality_summary.json
    ... --series --shard 0/2                            # per-frame curves -> <run>/series_0.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .depth_eval import read_clip
from .motion_eval import compare, compensate_camera, flow, load_raft
from .seed_metrics import windows

SEAMS = (33, 66, 99)  # flow index t: frame t -> t+1 crosses into the next window
RATE_Q = (
    "These {n} frames are evenly spaced over a {sec:.1f}-second video from the head camera of a two-armed robot. "
    "Ignore what task it does. Rate the video from 1 (very poor) to 5 (like real footage) on: "
    "object_dynamics = objects move only when a gripper touches them, keep their shape, size and identity, and fall, "
    "slide or rest in a physically believable way; arm_motion = the arms and grippers move along natural, "
    "continuous paths and keep their shape; visual_quality = sharp and detailed, with no blur, smearing, warping, "
    "noise or flicker. Answer with JSON only: {{\"object_dynamics\": <1-5>, \"arm_motion\": <1-5>, "
    "\"visual_quality\": <1-5>, \"reason\": \"<one sentence>\"}}")
RATE_KEYS = ("object_dynamics", "arm_motion", "visual_quality")


def jobs(run: Path) -> list[tuple[str, Path, int]]:
    gt_meta = json.loads((run / "gt" / "meta.json").read_text())
    out = [(f"{p.parent.name}/{p.stem}", p, 0) for p in sorted((run / "clips").glob("*/*.mp4"))]
    out += [(f"gt/{c}", run / "gt" / f"{c}.mp4", m["gt_frames"]) for c, m in sorted(gt_meta.items())]
    return out


def shard(items: list, spec: str) -> list:
    i, n = (int(x) for x in spec.split("/"))
    return items[i::n]


def grey(v: torch.Tensor) -> torch.Tensor:  # (T, 3, H, W) 0-255 -> (T, H, W)
    return (0.299 * v[:, 0] + 0.587 * v[:, 1] + 0.114 * v[:, 2])


def sharpness(g: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=g.dtype, device=g.device)[None, None]
    return F.conv2d(g[:, None], k).flatten(1).var(1)


def hf_share(g: torch.Tensor, cut: float = 0.25) -> torch.Tensor:
    p = torch.fft.rfft2(g - g.mean((1, 2), keepdim=True)).abs() ** 2
    fy = torch.fft.fftfreq(g.shape[1], device=g.device)[:, None].abs()
    fx = torch.fft.rfftfreq(g.shape[2], device=g.device)[None, :]
    hi = (torch.maximum(fy, fx) > cut).to(p.dtype)
    return (p * hi).flatten(1).sum(1) / p.flatten(1).sum(1)


def warp_back(nxt: torch.Tensor, fl: torch.Tensor) -> torch.Tensor:
    """frame t+1 (T, 3, H, W) sampled at x + flow_t(x) -> estimate of frame t."""
    t, _, h, w = nxt.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=nxt.device), torch.arange(w, device=nxt.device), indexing="ij")
    gx = (xs[None] + fl[:, 0]) / (w - 1) * 2 - 1
    gy = (ys[None] + fl[:, 1]) / (h - 1) * 2 - 1
    return F.grid_sample(nxt, torch.stack([gx, gy], -1), align_corners=True, padding_mode="border")


def mean(x) -> float | None:
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else None


@torch.no_grad()
def clip_metrics(raft, v: np.ndarray, dev) -> tuple[dict, np.ndarray]:
    full = torch.from_numpy(v).to(dev).permute(0, 3, 1, 2).float()
    half = F.interpolate(full, size=(192, 256), mode="bilinear", antialias=True, align_corners=False)
    fl = flow(raft, half.round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())  # (T-1, 2, h, w)
    flt = torch.from_numpy(fl).to(dev)
    fc = torch.from_numpy(compensate_camera(fl)).to(dev)
    mag = fc.norm(dim=1).flatten(1).mean(1) * 2  # px at full resolution
    werr = (warp_back(half[1:], flt) - half[:-1]).abs().flatten(1).mean(1)
    m = {"motion": mean(mag[1:].tolist()),
         "jerk": float((fc[2:] - fc[1:-1]).norm(dim=1).mean() * 2 / (mag[1:].mean() + 1e-6)),
         "warp_err": mean(werr[1:].tolist())}
    seams = []
    for s in SEAMS:
        if s + 6 <= len(werr):
            around = torch.cat([werr[s - 5:s - 1], werr[s + 2:s + 6]])
            seams.append(float(werr[s] / (around.median() + 1e-6)))
    m["seam"] = mean(seams)
    wins = windows(len(v))
    wm = [mag[w.start - 1:w.stop - 1].mean().item() for w in wins if w.stop - 1 <= len(mag)]
    m["persist"] = wm[-1] / (wm[0] + 1e-6) if len(wm) >= 2 else None
    g = grey(full)
    sh, hf = sharpness(g), hf_share(g)
    m["sharp"] = float(sh[1:].mean() / sh[0])
    m["hf"] = float(hf[1:].mean() / hf[0])
    last = wins[-1]
    m["sharp_w4"] = float(sh[last.start:last.stop].mean() / sh[0]) if last.stop <= len(v) else None
    return m, fl


def run_metrics(a) -> None:
    run, dev = Path(a.run), torch.device(a.device)
    out_p = run / f"quality_{a.shard.split('/')[0]}.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    raft = load_raft(str(dev), large=True)
    gt_flow = {}
    todo = shard(jobs(run), a.shard)
    gt_meta = json.loads((run / "gt" / "meta.json").read_text())
    for key, path, n in todo:
        if key in res:
            continue
        v = read_clip(path)
        if n:
            v = v[:n]
        m, fl = clip_metrics(raft, v, dev)
        cid, name = key.split("/")
        if cid != "gt" and name.startswith("real_"):
            if cid not in gt_flow:
                gt_flow[cid] = flow(raft, F.interpolate(torch.from_numpy(read_clip(run / "gt" / f"{cid}.mp4")[:gt_meta[cid]["gt_frames"]])
                                                        .permute(0, 3, 1, 2).float(), size=(192, 256), mode="bilinear",
                                                        antialias=True).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy())
            k = min(len(gt_flow[cid]), len(fl))
            m["vs_gt"] = compare(fl[1:k], gt_flow[cid][1:k])
        res[key] = m
        out_p.write_text(json.dumps(res, indent=1))
        print(f"[quality] {key} " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)), flush=True)


@torch.no_grad()
def run_series(a) -> None:
    """Per-frame warp_err and sharpness (relative to the context frame) -> <run>/series_<i>.json, for the plots."""
    run, dev = Path(a.run), torch.device(a.device)
    out_p = run / f"series_{a.shard.split('/')[0]}.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    raft = load_raft(str(dev), large=True)
    for key, path, n in shard(jobs(run), a.shard):
        if key in res:
            continue
        v = read_clip(path)
        if n:
            v = v[:n]
        full = torch.from_numpy(v).to(dev).permute(0, 3, 1, 2).float()
        half = F.interpolate(full, size=(192, 256), mode="bilinear", antialias=True, align_corners=False)
        fl = torch.from_numpy(flow(raft, half.round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())).to(dev)
        werr = (warp_back(half[1:], fl) - half[:-1]).abs().flatten(1).mean(1)
        sh = sharpness(grey(full))
        res[key] = {"warp_err": [round(x, 4) for x in werr.tolist()], "sharp": [round(x, 4) for x in (sh / sh[0]).tolist()]}
        out_p.write_text(json.dumps(res))
        print(f"[series] {key}", flush=True)


def run_rate(a) -> None:
    from .cap_judge import N_FRAMES, Judge

    run = Path(a.run)
    out_p = run / f"rating_{a.shard.split('/')[0]}.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    judge = Judge(a.judge, torch.device(a.device))
    for key, path, n in shard(jobs(run), a.shard):
        if key in res:
            continue
        v = read_clip(path)
        if n:
            v = v[:n]
        idx = np.linspace(0, len(v) - 1, N_FRAMES).round().astype(int)
        frames = [torch.from_numpy(v[i]).permute(2, 0, 1) for i in idx]
        msgs = [{"role": "user", "content": [{"type": "video", "video": frames},
                                             {"type": "text", "text": RATE_Q.format(n=N_FRAMES, sec=len(v) / 15)}]}]
        inp = judge.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                             return_tensors="pt", fps=N_FRAMES * 15 / len(v)).to(judge.device)
        with torch.no_grad():
            out = judge.model.generate(**inp, max_new_tokens=100, do_sample=False)
        raw = judge.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        mt = re.search(r"\{.*\}", raw, re.S)
        try:
            ans = json.loads(mt.group(0)) if mt else {}
        except json.JSONDecodeError:
            ans = {}
        res[key] = {**{k: (int(ans[k]) if str(ans.get(k, "")).strip().isdigit() else None) for k in RATE_KEYS},
                    "reason": str(ans.get("reason", "")), "raw": raw.strip()}
        out_p.write_text(json.dumps(res, indent=1))
        print(f"[rate] {key} " + " ".join(f"{k}={res[key][k]}" for k in RATE_KEYS), flush=True)


def run_summary(a) -> None:
    run = Path(a.run)
    q, r = {}, {}
    for f in run.glob("quality_[0-9]*.json"):
        q.update(json.loads(f.read_text()))
    for f in run.glob("rating_[0-9]*.json"):
        r.update(json.loads(f.read_text()))
    cases = sorted({k.split("/")[1] for k in q if k.startswith("gt/")})
    depths = sorted({k.split("_N")[1].split("_")[0] for k in q if "_N" in k}, key=int)
    keys = ("motion", "jerk", "warp_err", "seam", "persist", "sharp", "hf", "sharp_w4")
    summary = {"gt": {k: mean(q[f"gt/{c}"].get(k) for c in cases) for k in keys}}
    summary["gt"].update({k: mean(r.get(f"gt/{c}", {}).get(k) for c in cases) for k in RATE_KEYS})
    for d in depths:
        rows = {k: v for k, v in q.items() if f"_N{d}_" in k}
        s = {k: mean(v.get(k) for v in rows.values()) for k in keys}
        real = [v["vs_gt"] for k, v in rows.items() if k.split("/")[1].startswith("real_") and "vs_gt" in v]
        s.update({f"gt_{k}": mean(x.get(k) for x in real) for k in ("dir_cos", "region_cos", "mag_ratio")})
        # motion relative to the real footage, per case (real prompt), so busy and quiet scenes weigh the same
        s["motion_vs_gt"] = mean(rows[k]["motion"] / q[f"gt/{k.split('/')[0]}"]["motion"]
                                 for k in rows if k.split("/")[1].startswith("real_"))
        s.update({k: mean(v.get(k) for kk, v in r.items() if f"_N{d}_" in kk) for k in RATE_KEYS})
        s["n"] = len(rows)
        summary[f"N{d}"] = s
    (run / "quality_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--rate", action="store_true")
    p.add_argument("--judge")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--series", action="store_true", help="per-frame warp_err and sharpness for the plots")
    a = p.parse_args()
    run_summary(a) if a.summary else run_series(a) if a.series else run_rate(a) if a.rate else run_metrics(a)


if __name__ == "__main__":
    main()
