#!/bin/bash
# Phase 5: Looped-DiT retrofit training from the Phase-3 EMA (fresh start, or resume=auto from /dev/shm/runs/loop384).
#   scripts/launch_loop384.sh [config=configs/looped_ft_384_r1.json]
set -e
cd /workspace/loop-dit/flux-action
cfg=${1:-/workspace/loop-dit/configs/looped_ft_384_r1.json}
cp "$cfg" ../logs/looped_ft_384.launched.json
mkdir -p /dev/shm/runs/loop384 && cp "$cfg" /dev/shm/runs/loop384/config.json
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "cd /workspace/loop-dit/flux-action && PYTHONPATH=/workspace/loop-dit FLUX_ACTION_PG_TIMEOUT_MINUTES=10 OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN .venv/bin/python .venv/bin/torchrun --nproc_per_node 2 --master_port 29700 -m flux_action.cli train --config $cfg 2>&1 | tee -a /workspace/loop-dit/logs/loop384.log; echo TRAIN_EXITED; sleep 1d"
echo "launched tmux train (Phase 5) with $cfg at $(date -u)"
