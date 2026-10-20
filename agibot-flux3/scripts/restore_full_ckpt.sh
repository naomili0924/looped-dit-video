#!/bin/bash
# Restore a full resumable checkpoint mirrored by scripts/hub_full_ckpt.py, e.g. on a new machine:
#   scripts/restore_full_ckpt.sh 2000 /workspace/runs/looprand_long [random_depth_n1-8] [Jinyan0924/flux3-agibot-looped]
# then train with the same config, "resume": "<dst>/step-<N>" (or resume=auto with output_dir=<dst>).
set -e
step=${1:?step}; dst=${2:?destination run dir}; prefix=${3:-random_depth_n1-8}; repo=${4:-Jinyan0924/flux3-agibot-looped}
A=$(cd "$(dirname "$0")/.." && pwd)
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }
mkdir -p "$dst"
"$A/flux-action/.venv/bin/python" - "$step" "$dst" "$prefix" "$repo" <<'PY'
import os, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
step, dst, prefix, repo = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
tmp = dst / ".hub_restore"
snapshot_download(repo, token=os.environ.get("HF_TOKEN"), allow_patterns=[f"{prefix}/checkpoints/step-{step}/*"],
                  local_dir=str(tmp))
src = tmp / prefix / "checkpoints" / f"step-{step}"
assert (src / "COMPLETE").exists(), f"incomplete download: {src}"
os.rename(src, dst / f"step-{step}")
shutil.rmtree(tmp, ignore_errors=True)
print(f"restored {dst / f'step-{step}'}")
PY
