"""Zero-shot loop-depth probe of a *plain* (N = 1) checkpoint: run it as the Looped-DiT retrofit (pre 12 blocks |
core 9 single blocks x N | post 12 blocks, no gates) at several inference depths and measure the validation
velocity loss per context length. The core blocks were trained to be applied once, so this shows how much the
loop hurts before any looped training (Phase 5 trains it). CPU by default (the GPUs train).

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/probe_loops.py \
        --ema /dev/shm/ema/ema_step-1000.safetensors --k 1 9 --depths 1 2 4 8 10 --windows 8 --t-grid 2 \
        --out ../results/probe_loops_step1000.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flux_action.config import PolicyConfig
from flux_action.models.text_encoder import VEC_DIM
from flux_action.policy import FluxActionPolicy
from flux_action.processing import packing, video_only

from ldvflux import sampling
from ldvflux.evaluate import _load_windows, _t_grid
from ldvflux.stream import fixed_windows, load_splits, trim

LOOP = {"loop_split": [12, 9, 12], "num_loops": 1, "attn_gate": False, "xsa": False}


def build(cfg: dict, ema: str | None, device: str) -> FluxActionPolicy:
    pol = {**cfg["policy"], "torch_dtype": "bfloat16", "loop": LOOP, "overlay_weights": ema}
    policy = FluxActionPolicy(PolicyConfig(**pol))
    if device != "cpu":
        policy.dit.to(device)
        for m in policy.frozen.modules():
            m.to(device)
    policy.eval()
    return policy


@torch.no_grad()
def probe(policy, windows, k: int, depths: list[int], t_values: torch.Tensor, batch: int, log) -> dict:
    """{depth: {"loss": mean velocity MSE, "per_window": [...], "sec_per_forward": s}} at context length k."""
    future = policy.config.future_frames
    acc = {d: {"sum": 0.0, "n": 0, "per_window": [], "time": 0.0, "calls": 0} for d in depths}
    for s in range(0, len(windows), batch):
        chunk = windows[s : s + batch]
        frames = torch.stack([trim(w, k, future) for w in chunk])
        t0 = time.perf_counter()
        cond, fut = sampling.encode_windows(policy, frames, k)
        log(f"  k={k} windows {s}-{s + len(chunk) - 1}: VAE encode {time.perf_counter() - t0:.0f}s")
        ctxs = [c for c, _ in policy._contexts([w["caption"] for w in chunk], policy.device)]
        video = video_only.pack(cond, fut, k, policy.config.fps)
        for j, t in enumerate(t_values.tolist()):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1000 * s + j)
                kw, target = video_only.forward_kwargs(video, torch.full((len(chunk),), t))
            packed, seqlens = packing.pack_windows(kw, ctxs, VEC_DIM)
            tgt = packing.flatten_windows({"x": target})["x"]
            inputs = policy._cast_inputs(packed)
            for d in depths:
                t0 = time.perf_counter()
                pred = policy.dit(**inputs, seqlens=seqlens, num_loops=d)["x_video"]
                dt = time.perf_counter() - t0
                loss = float(video_only.flow_loss(pred, tgt)["loss"])
                a = acc[d]
                a["sum"] += loss * len(chunk)
                a["n"] += len(chunk)
                a["per_window"].append({"window": s, "t": round(t, 4), "loss": loss})
                a["time"] += dt
                a["calls"] += 1
                log(f"    t={t:.3f} depth {d:2d}: loss {loss:.4f}  ({dt:.0f}s)")
    return {d: {"loss": a["sum"] / a["n"], "per_window": a["per_window"], "sec_per_forward": a["time"] / a["calls"]}
            for d, a in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="/workspace/loop-dit/configs/video_ft_384_r2.json")
    p.add_argument("--ema", default=None, help="consolidated EMA safetensors to overlay (None = base trunk)")
    p.add_argument("--k", type=int, nargs="+", default=[1, 9])
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 10])
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--t-grid", type=int, default=2)
    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="/workspace/loop-dit/results/probe_loops_step1000.json")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    cfg = json.load(open(args.config))
    ds = cfg["dataset_kwargs"]
    meta = load_splits(ds["splits"])["meta"]
    hw = tuple(ds.get("frame_hw") or ()) or None
    picks = fixed_windows("val", args.windows, seed=0, splits_path=ds["splits"])  # the eval hook's first windows
    windows = _load_windows(picks, meta, hw)
    log(f"loaded {len(windows)} val windows; building policy on {args.device} (ema={args.ema})")
    t0 = time.perf_counter()
    policy = build(cfg, args.ema, args.device)
    log(f"policy built in {time.perf_counter() - t0:.0f}s: {type(policy.dit).__name__} loop={LOOP}")
    pc = policy.config
    t_values = _t_grid(args.t_grid, pc.video_logit_mean, pc.video_logit_std)
    report = {"ema": args.ema, "config": args.config, "loop": LOOP, "windows": args.windows, "t_values": t_values.tolist(),
              "captions": [w["caption"] for w in windows], "k": {}}
    for k in args.k:
        log(f"k={k}")
        report["k"][str(k)] = probe(policy, windows, k, args.depths, t_values, args.batch, log)
        out.write_text(json.dumps(report, indent=1))
        summary = {d: round(r["loss"], 4) for d, r in report["k"][str(k)].items()}
        log(f"k={k} val loss by depth: {summary}")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
