"""Two-rank CPU reproduction of rank-asymmetric micro-batch k patterns under FSDP2 gradient accumulation.

Mimics the DiT: a root group with per-stream input projections, 5 ``video`` blocks, 5 ``video_cond`` blocks and
3 joint blocks, each ``fully_shard``-ed (bf16 params, fp32 reduce). A k = 0 micro-batch runs the cond blocks on a
2-token zero dummy whose output is discarded (as ``JointSingleSeq`` does for inactive modalities), so the cond
parameters get no gradient from it. Each rank runs its own micro-batch order; a mismatch in the collectives
issued shows up as a gloo timeout.

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/torchrun --nproc_per_node 2 --master_port 29812 \
        ../scripts/test_fsdp_rank_patterns.py --rank0 5 1 9 5 --rank1 0 1 9 0 [--connect-dummy]
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

D = 32


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class DiT(nn.Module):
    def __init__(self, connect_dummy: bool):
        super().__init__()
        self.video_in = nn.Linear(D, D)
        self.cond_in = nn.Linear(D, D)
        self.video_blocks = nn.ModuleList([Block() for _ in range(5)])
        self.cond_blocks = nn.ModuleList([Block() for _ in range(5)])
        self.joint = nn.ModuleList([Block() for _ in range(3)])
        self.connect_dummy = connect_dummy

    def forward(self, x, k):
        v = self.video_in(x)
        c = self.cond_in(x[:k] if k else torch.zeros(2, D, dtype=x.dtype))
        for b in self.video_blocks:
            v = b(v)
        for b in self.cond_blocks:
            c = b(c)
        if k:
            v = v + c.mean(0, keepdim=True)
        elif self.connect_dummy:
            v = v + 0.0 * c.sum()
        for b in self.joint:
            v = b(v)
        return v.float().pow(2).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank0", type=int, nargs="+", required=True)
    p.add_argument("--rank1", type=int, nargs="+", required=True)
    p.add_argument("--connect-dummy", action="store_true")
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--trace", action="store_true", help="print every FSDP post_backward (group, reduce?, grads?)")
    args = p.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout))
    rank = dist.get_rank()
    if args.trace:
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

        orig = pg.FSDPParamGroup.post_backward

        def traced(self, *a, **kw):
            has = any(
                fp.unsharded_accumulated_grad is not None
                or (hasattr(fp, "_unsharded_param") and fp._unsharded_param.grad is not None)
                for fp in self.fsdp_params
            )
            print(f"[r{rank}] post_backward {self._module_fqn or 'root':<16} reduce={self.reduce_grads} "
                  f"grads={has} state={self._training_state.name}", flush=True)
            return orig(self, *a, **kw)

        pg.FSDPParamGroup.post_backward = traced
    torch.manual_seed(0)
    model = DiT(args.connect_dummy)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blocks in (model.video_blocks, model.cond_blocks, model.joint):
        for b in blocks:
            fully_shard(b, mp_policy=mp, reshard_after_forward=False)
    fully_shard(model, mp_policy=mp, reshard_after_forward=False)
    order = args.rank0 if rank == 0 else args.rank1
    x = torch.randn(16, D)
    result = "ok"
    try:
        for i, k in enumerate(order):
            model.set_requires_gradient_sync(i == len(order) - 1)
            model(x, k).backward()
        n_grad = sum(1 for p_ in model.parameters() if p_.grad is not None)
        # the metrics all-reduce that follows the backward in the trainer
        t = torch.zeros(3, dtype=torch.float64)
        dist.all_reduce(t)
        result = f"ok (params with grad: {n_grad})"
    except Exception as e:  # noqa: BLE001
        result = f"{type(e).__name__}: {str(e)[:160]}"
    print(f"rank {rank} order {order} connect_dummy={args.connect_dummy}: {result}", flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
