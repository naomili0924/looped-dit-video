#!/bin/bash
# Random-depth (N 1-8) Looped-DiT training on the 2xH100 box (fresh start, or resume=auto from the config's output_dir).
#   scripts/launch_loop384.sh [config=configs/looped_ft_384_r1.json]
# The run dir comes from the config's output_dir; the log is logs/<basename of output_dir>.log.
set -e
cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3/flux-action
cfg=${1:-/workspace/loop-dit-eval/looped-dit-video/agibot-flux3/configs/looped_rand_384_r1.json}
run=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$cfg")
name=$(basename "$run"); log=/workspace/loop-dit-eval/looped-dit-video/agibot-flux3/logs/$name.log
cp "$cfg" "/workspace/loop-dit-eval/looped-dit-video/agibot-flux3/logs/$name.launched.json"
mkdir -p "$run" && cp "$cfg" "$run/config.json"
if pgrep -f '^[^ ]*python[^ ]* .*flux_action[.]cli train' >/dev/null; then echo "a trainer is still running; refusing to launch"; exit 1; fi
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3/flux-action && set -a; . /workspace/.env; set +a; PYTHONPATH=/workspace/loop-dit-eval/looped-dit-video/agibot-flux3 FLUX_ACTION_PG_TIMEOUT_MINUTES=10 OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN .venv/bin/python .venv/bin/torchrun --nproc_per_node 2 --master_port 29700 -m flux_action.cli train --config $cfg 2>&1 | tee -a $log; echo TRAIN_EXITED; sleep 1d"
echo "launched tmux train (Phase 5) with $cfg -> $run, log $log at $(date -u)"
