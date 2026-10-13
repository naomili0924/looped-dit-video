"""Replay dumped DiT micro-batch inputs (FLUX_DUMP_DIR) through the video-only DiT and report non-finite grads.

    # one GPU, plain bf16 model
    PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/replay_dit.py --weights W.pt --dumps DIR/dit-inputs-rank0-call*.pt
    # same inputs under FSDP2 on two ranks (rank r replays its own rank-r dumps)
    PYTHONPATH=/workspace/loop-dit .venv/bin/torchrun --nproc_per_node 2 ../scripts/replay_dit.py --fsdp --weights W.pt --dumps DIR

Modes: --ac (activation checkpointing), --fsdp (fully_shard like the trainer; needs torchrun), --autocast,
--each (zero_grad between micro-batches: which one alone is bad) vs. accumulate like the trainer.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, "/workspace/loop-dit")
from flux_action.config import PolicyConfig  # noqa: E402
from flux_action.policy import FluxActionPolicy  # noqa: E402
from flux_action.processing import packing, video_only  # noqa: E402
from flux_action.training.distributed import (  # noqa: E402
    apply_activation_checkpointing,
    build_mesh,
    init_distributed,
    shard_dit,
)


def bad_grads(module):
    from torch.distributed.tensor import DTensor

    out = []
    for n, p in module.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        if not torch.isfinite(g).all():
            out.append(n.replace("_checkpoint_wrapped_module.", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/workspace/loop-dit/configs/anomaly.json")
    ap.add_argument("--weights", default="")
    ap.add_argument("--dumps", nargs="+", required=True, help="dump files, or one directory (per-rank files)")
    ap.add_argument("--ac", action="store_true")
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--autocast", action="store_true")
    ap.add_argument("--each", action="store_true")
    ap.add_argument("--txt-no-ac", action="store_true")
    ap.add_argument("--no-reshard", action="store_true", help="FSDP reshard_after_forward=False")
    ap.add_argument("--reduce-bf16", action="store_true", help="FSDP reduce_dtype=bfloat16")
    args = ap.parse_args()
    rank, world, dev = init_distributed()
    cfg = json.load(open(args.config))
    dtype = "float32" if args.fsdp else "bfloat16"
    policy = FluxActionPolicy(PolicyConfig(**{**cfg["policy"], "torch_dtype": dtype}),
                              video_vae=object(), text_encoder=torch.nn.Identity())
    if args.weights:
        sd = torch.load(args.weights, map_location="cpu", weights_only=False)["model"]
        sd = {k.removeprefix("dit.").replace("_checkpoint_wrapped_module.", ""): v for k, v in sd.items()}
        res = policy.dit.load_state_dict({k: v.to(getattr(torch, dtype)) for k, v in sd.items()}, strict=False)
        assert not res.missing_keys and not res.unexpected_keys, (res.missing_keys[:3], res.unexpected_keys[:3])
    policy.freeze_conditioning_heads()
    if args.txt_no_ac:
        os.environ["FLUX_TXT_NO_AC"] = "1"
    if args.fsdp:
        mesh = build_mesh(world, None, dev.type)
        shard_dit(policy.dit, mesh, param_dtype=torch.bfloat16,
                  reduce_dtype=torch.bfloat16 if args.reduce_bf16 else torch.float32,
                  reshard_after_forward=not args.no_reshard, activation_checkpointing=args.ac)
        policy.set_compute_dtype(torch.bfloat16)
    else:
        if args.ac:
            apply_activation_checkpointing(policy.dit)
        policy.dit.to(dev)
    policy.train()

    files = args.dumps
    if len(files) == 1 and os.path.isdir(files[0]):
        files = sorted(glob.glob(os.path.join(files[0], f"dit-inputs-rank{rank}-call*.pt")))
    accum = len(files)
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        packed = {k: v.to(dev) for k, v in d["packed"].items()}
        target = d["target"].to(dev)
        if args.fsdp:
            policy.dit.set_requires_gradient_sync(i == accum - 1 or args.each)
        if args.each:
            policy.dit.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.autocast):
            pred = policy.dit(**policy._cast_inputs(packed), seqlens=d["seqlens"])["x_video"]
            loss = video_only.flow_loss(pred, packing.flatten_windows({"x": target})["x"])["loss"]
        (loss / (1 if args.each else accum)).backward()
        bad = bad_grads(policy.dit)
        print(f"[rank {rank}] {os.path.basename(f)} k={d['k']} t={[round(x, 3) for x in d['timesteps'].tolist()]} "
              f"loss {float(loss):.4f} pred finite {bool(torch.isfinite(pred).all())} non-finite grads: {len(bad)} {bad[:4]}",
              flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
