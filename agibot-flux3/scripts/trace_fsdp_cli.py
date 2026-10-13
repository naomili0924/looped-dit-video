"""Run ``flux_action.cli`` with FSDP2's ``post_backward`` traced: one line per parameter group and micro-batch
(rank, group, reduce?, has grads?) so the reduce-scatter order of the two ranks can be compared.

    cd flux-action && FLUX_ACTION_PG_TIMEOUT_MINUTES=3 PYTHONPATH=/workspace/loop-dit .venv/bin/torchrun \
        --nproc_per_node 2 --master_port 29700 ../scripts/trace_fsdp_cli.py train --config <debug config>
"""

from __future__ import annotations

import os
import sys

import torch.distributed as dist
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

_orig = pg.FSDPParamGroup.post_backward


def _traced(self, *a, **kw):
    has = any(
        fp.unsharded_accumulated_grad is not None
        or (hasattr(fp, "_unsharded_param") and fp._unsharded_param.grad is not None)
        for fp in self.fsdp_params
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    fqn = (self._module_fqn or "root").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
    print(f"[trace r{rank}] post_backward {fqn:<40} reduce={self.reduce_grads} grads={has} "
          f"state={self._training_state.name}", flush=True)
    return _orig(self, *a, **kw)


pg.FSDPParamGroup.post_backward = _traced

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    from flux_action import cli

    sys.argv = [sys.argv[0], *sys.argv[1:]]
    cli.main()
