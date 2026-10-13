"""Reproduce / verify the FSDP2 mixed-gradient-dtype assertion seen at step ~1080 of the Phase-3 run.

With ``reduce_dtype=fp32`` and gradient sync disabled for the first micro-batches, FSDP2 stores a parameter's
accumulated gradient in fp32; a parameter that first receives a gradient in the *last* (synced) micro-batch
still has a bf16 autograd gradient -> ``FSDP reduce-scatter expects uniform gradient dtype``. A k = 0 batch
(no context stream) followed by a k > 0 batch is exactly that pattern for the root parameter group.

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/torchrun --nproc_per_node 2 --master_port 29811 \
        ../scripts/test_fsdp_accum_dtype.py

Expected: the "unsorted" order (k0, k0, k0, k>0) raises, the sorted order (k>0 first, k0 last) trains.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


class TwoStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.video = nn.Linear(16, 16)
        self.cond = nn.Linear(16, 16)
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])

    def forward(self, x, k):
        h = self.video(x)
        if k:
            h = h + self.cond(x)
        for b in self.blocks:
            h = b(h)
        return h.float().pow(2).mean()


def run(order: list[int]) -> str:
    model = TwoStream()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for b in model.blocks:
        fully_shard(b, mp_policy=mp, reshard_after_forward=False)
    fully_shard(model, mp_policy=mp, reshard_after_forward=False)
    x = torch.randn(4, 16)
    try:
        for i, k in enumerate(order):
            model.set_requires_gradient_sync(i == len(order) - 1)
            model(x, k).backward()
        return "ok"
    except AssertionError as e:
        return f"AssertionError: {e}"


def main():
    dist.init_process_group("gloo")
    torch.manual_seed(0)
    unsorted = [0, 0, 0, 5]
    srt = [5, 0, 0, 0]
    r1, r2 = run(unsorted), run(srt)
    if dist.get_rank() == 0:
        print(f"order {unsorted}: {r1}")
        print(f"order {srt}: {r2}")
        assert "AssertionError" in r1 and r2 == "ok", "unexpected outcome"
        print("hazard reproduced and sorted order is clean")
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
