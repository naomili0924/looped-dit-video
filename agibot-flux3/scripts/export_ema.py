"""Consolidate a checkpoint's CPU-EMA shards into one bf16 safetensors file (the same file hub_backup.py uploads).

    PYTHONPATH=/workspace/loop-dit flux-action/.venv/bin/python scripts/export_ema.py /dev/shm/runs/ft384/step-5000 /dev/shm/ema/ema_ft384_step-5000.safetensors
"""
import sys
from pathlib import Path

from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from hub_backup import consolidate_ema  # noqa: E402

step_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
ema = consolidate_ema(step_dir)
save_file(ema, str(out), metadata={"step": step_dir.name, "run": str(step_dir.parent)})
print(f"wrote {out} ({sum(v.numel() for v in ema.values()) / 1e9:.3f}B params, {len(ema)} tensors)")
