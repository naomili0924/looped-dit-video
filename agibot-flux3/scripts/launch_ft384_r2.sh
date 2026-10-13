#!/bin/bash
# Resume the Phase-3 run from its latest checkpoint with the r2 config (EMA every 10, compiled blocks, K_SET {1,5,9}).
set -e
cd /workspace/loop-dit/flux-action
cfg=${1:-../configs/video_ft_384_r2.json}
# resume: auto starts from step 0 if no local checkpoint exists (they are deleted after the Hub upload) — refuse.
if ! ls -d /dev/shm/runs/ft384/step-*/COMPLETE >/dev/null 2>&1; then
  echo "no complete local checkpoint in /dev/shm/runs/ft384 — run scripts/restore_checkpoint.sh <step> first"; exit 1
fi
cp "$cfg" ../logs/video_ft_384_r2.launched.json
cp "$cfg" /dev/shm/runs/ft384/config_r2.json
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "cd /workspace/loop-dit/flux-action && PYTHONPATH=/workspace/loop-dit FLUX_ACTION_PG_TIMEOUT_MINUTES=10 OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN .venv/bin/python .venv/bin/torchrun --nproc_per_node 2 --master_port 29700 -m flux_action.cli train --config $cfg 2>&1 | tee -a ../logs/ft384.log; echo TRAIN_EXITED; sleep 1d"
echo "launched tmux train with $cfg at $(date -u)"
