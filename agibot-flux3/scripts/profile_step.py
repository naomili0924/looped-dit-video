"""Where does a training step spend its time?  One GPU, no FSDP, real val windows, the training code paths.

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/profile_step.py \
        --config ../configs/video_ft_384.json --k 0 1 17 --windows 4 --repeats 3 --out ../logs/profile/step.json

Measures, per context length k, with CUDA synchronisation around each phase (micro-batch = ``--windows``):
  prepare   : policy.prepare  (VAE encode of context + future clips, caption lookup)
  forward   : DiT forward with activation checkpointing (as trained)
  backward  : DiT backward (includes the AC recompute)
  fwd_no_ac / bwd_no_ac : the same without activation checkpointing (skipped on OOM)
and, with --trace, a torch.profiler kernel breakdown of one forward+backward (attention / matmul / other).
bf16 parameters + bf16 grads stand in for FSDP's bf16 compute; FSDP communication is not part of this measurement,
so (31 s step - 4 micro-batches x (forward + backward)) is an upper bound on communication + optimizer time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flux_action.config import PolicyConfig
from flux_action.policy import FluxActionPolicy
from flux_action.training.data import collate_windows
from flux_action.training.distributed import apply_activation_checkpointing, apply_compile

from ldvflux.evaluate import _load_windows
from ldvflux.stream import fixed_windows, load_splits, trim


def build_policy(cfg: dict, device: torch.device, ac: bool, compile_blocks: bool = False) -> FluxActionPolicy:
    pol = dict(cfg["policy"])
    pol["torch_dtype"] = "bfloat16"
    policy = FluxActionPolicy(PolicyConfig(**pol))
    policy.dit.to(device)
    for m in policy.frozen.modules():
        m.to(device)
    if ac:
        apply_activation_checkpointing(policy.dit)
    if compile_blocks:
        apply_compile(policy.dit)
    policy.train()
    for p in policy.dit.parameters():
        p.requires_grad_(True)
    return policy


def make_batch(windows: list[dict], k: int, future: int) -> dict:
    items = []
    for i, w in enumerate(windows):
        items.append({"images.head": trim(w, k, future), "task": w["caption"], "context_frames": k,
                      "window_seed": 1000 + i, "episode": int(w["episode"]), "anchor": int(w["anchor"])})
    return collate_windows(items)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def profile_k(policy, batch, repeats: int) -> dict:
    res = {}
    # warm-up (cudnn autotune, allocator)
    prepared, _ = timed(lambda: policy.prepare(batch))
    (loss, _), _ = timed(lambda: policy(batch, prepared=prepared))
    _, _ = timed(lambda: loss.backward())
    policy.dit.zero_grad(set_to_none=True)
    ts = {"prepare": [], "forward": [], "backward": []}
    for _ in range(repeats):
        prepared, t = timed(lambda: policy.prepare(batch))
        ts["prepare"].append(t)
        (loss, info), t = timed(lambda: policy(batch, prepared=prepared))
        ts["forward"].append(t)
        _, t = timed(lambda: loss.backward())
        ts["backward"].append(t)
        policy.dit.zero_grad(set_to_none=True)
    for name, v in ts.items():
        res[name] = round(sum(v) / len(v), 3)
    res["n_valid_windows"] = int(info["n_valid_windows"])
    res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    return res


def trace_one(policy, batch, path: Path) -> dict:
    from torch.profiler import ProfilerActivity, profile

    prepared = policy.prepare(batch)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        loss, _ = policy(batch, prepared=prepared)
        loss.backward()
        torch.cuda.synchronize()
    policy.dit.zero_grad(set_to_none=True)
    prof.export_chrome_trace(str(path))
    buckets = {"attention": 0.0, "matmul": 0.0, "norm/act/elementwise": 0.0, "other": 0.0}
    total = 0.0
    for ev in prof.key_averages():
        if ev.device_type is None or "CUDA" not in str(ev.device_type):
            continue
        us = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0.0)
        if not us:
            continue
        n = ev.key.lower()
        if "attn" in n or "attention" in n or "flash" in n or "fmha" in n or "sdpa" in n:
            buckets["attention"] += us
        elif "gemm" in n or "matmul" in n or "cutlass" in n or "cublas" in n or "mm_" in n or "addmm" in n:
            buckets["matmul"] += us
        elif any(s in n for s in ("norm", "gelu", "silu", "elementwise", "vectorized", "reduce", "softmax", "copy", "cat")):
            buckets["norm/act/elementwise"] += us
        else:
            buckets["other"] += us
        total += us
    return {k: round(v / 1e6, 3) for k, v in {**buckets, "total_s": total}.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="/workspace/loop-dit/configs/video_ft_384.json")
    p.add_argument("--k", type=int, nargs="+", default=[0, 1, 17])
    p.add_argument("--windows", type=int, default=4, help="windows per micro-batch (training: windows_per_rank)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-ac-too", action="store_true", help="also time forward/backward without AC")
    p.add_argument("--trace", action="store_true", help="torch.profiler kernel breakdown at the largest k")
    p.add_argument("--compile", action="store_true", help="torch.compile every block (as compile_blocks=true)")
    p.add_argument("--out", default="/workspace/loop-dit/logs/profile/step.json")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    cfg = json.load(open(args.config))
    ds = cfg["dataset_kwargs"]
    meta = load_splits(ds["splits"])["meta"]
    hw = tuple(ds.get("frame_hw") or ()) or None
    future = cfg["policy"]["future_frames"]
    picks = fixed_windows("val", args.windows, seed=0, splits_path=ds["splits"])
    windows = _load_windows(picks, meta, hw)
    assert len(windows) == args.windows, len(windows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"config": args.config, "windows_per_micro_batch": args.windows, "frame_hw": hw, "future": future, "k": {}}

    policy = build_policy(cfg, dev, ac=True, compile_blocks=args.compile)
    report["compile"] = args.compile
    for k in args.k:
        batch = make_batch(windows, k, future)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        report["k"][str(k)] = profile_k(policy, batch, args.repeats)
        report["k"][str(k)]["wall_incl_warmup_s"] = round(time.time() - t0, 1)  # compile time shows up here
        print(f"[ac compile={args.compile}] k={k} {report['k'][str(k)]}", flush=True)
        out.write_text(json.dumps(report, indent=1))
    if args.trace:
        k = max(args.k)
        report["trace_k"] = k
        report["trace"] = trace_one(policy, make_batch(windows, k, future), out.with_suffix(".trace.json"))
        print(f"[trace] k={k} {report['trace']}", flush=True)
        out.write_text(json.dumps(report, indent=1))
    if args.no_ac_too:
        del policy
        torch.cuda.empty_cache()
        policy = build_policy(cfg, dev, ac=False)
        for k in args.k:
            batch = make_batch(windows, k, future)
            torch.cuda.reset_peak_memory_stats()
            try:
                r = profile_k(policy, batch, args.repeats)
                report["k"][str(k)].update({"forward_no_ac": r["forward"], "backward_no_ac": r["backward"],
                                            "peak_mem_gb_no_ac": r["peak_mem_gb"]})
            except torch.OutOfMemoryError:
                report["k"][str(k)]["no_ac"] = "OOM"
                policy.dit.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            print(f"[no-ac] k={k} {report['k'][str(k)]}", flush=True)
            out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
