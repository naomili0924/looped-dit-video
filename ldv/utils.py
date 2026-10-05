"""Distributed helpers, seeding, LR schedule, EMA and atomic saves (after Looped-DiT utils.py)."""

from __future__ import annotations

import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def init_distributed() -> torch.device:
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(hours=1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def learning_rate(step: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup from 1e-6 to base_lr, then constant (as in Looped-DiT)."""
    if step < warmup_steps:
        return 1e-6 + (step + 1) / warmup_steps * (base_lr - 1e-6)
    return base_lr


class CPUEma:
    """fp32 EMA of the model kept in host memory, updated every `every` steps
    (decay compounded accordingly), so a 5-7B model's EMA costs no GPU memory."""

    def __init__(self, model: torch.nn.Module, decay: float, every: int = 1):
        self.decay, self.every = decay, max(1, every)
        self.params = {n: p.detach().float().cpu().pin_memory() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        if step % self.every:
            return
        d = self.decay**self.every
        for n, p in model.named_parameters():
            e = self.params[n]
            e.mul_(d).add_(p.detach().float().cpu(), alpha=1 - d)

    def state_dict(self, dtype=torch.bfloat16) -> dict:
        return {n: p.to(dtype) for n, p in self.params.items()}

    def load_state_dict(self, sd: dict) -> None:
        for n, p in sd.items():
            if n in self.params:
                self.params[n].copy_(p.float())


def atomic_save(obj: object, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
