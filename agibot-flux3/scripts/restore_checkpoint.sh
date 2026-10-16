#!/bin/bash
# Restore a full training checkpoint from the private Hub mirror into the run dir (local copies are deleted
# after upload to free /dev/shm).   usage: scripts/restore_checkpoint.sh 1500 [/dev/shm/runs/ft384]
set -e
step=${1:?step}; run=${2:-/dev/shm/runs/ft384}
cd /workspace/loop-dit/flux-action
PYTHONPATH=/workspace/loop-dit .venv/bin/python - "$step" "$run" <<'PY'
import sys, os
from pathlib import Path
sys.path.insert(0, "/workspace/loop-dit/scripts")
from hub_backup import token
from huggingface_hub import snapshot_download
step, run = int(sys.argv[1]), Path(sys.argv[2])
snapshot_download(os.environ.get("HUB_REPO", "Jinyan0924/flux3-agibot-videoft"), token=token(), allow_patterns=[f"checkpoints/step-{step}/*"],
                  local_dir="/dev/shm/hub_restore")
src = Path("/dev/shm/hub_restore/checkpoints") / f"step-{step}"
dst = run / f"step-{step}"
assert (src / "COMPLETE").exists(), f"incomplete download: {src}"
os.rename(src, dst)
print(f"restored {dst}")
PY
