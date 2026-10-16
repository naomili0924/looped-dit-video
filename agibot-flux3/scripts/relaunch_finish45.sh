#!/bin/bash
# Relaunch the two unattended workers after the tmux server died (2026-10-16 22:08 UTC):
#   HF_TOKEN=... GH_TOKEN=... bash scripts/relaunch_finish45.sh
# hub2  : hub_backup.py resumes the staged upload of step-4500 (ema_cpu_0p999, rng, config, state, COMPLETE), then idles
# stop45: stop_and_eval.sh -> finish_r2.sh (stable Hub name, results, NOTES, GitHub) -> judge download -> run_loop_eval.sh
set -u
: "${HF_TOKEN:?HF_TOKEN}"; : "${GH_TOKEN:?GH_TOKEN}"
cd /workspace/loop-dit || exit 1
tmux has-session -t keep 2>/dev/null || tmux new-session -d -s keep 'sleep infinity'   # neutral argv for the tmux server
for s in hub2 stop45; do tmux has-session -t $s 2>/dev/null && { echo "$s already running"; exit 1; }; done
tmux new-session -d -s stop45 "cd /workspace/loop-dit && HF_TOKEN=$HF_TOKEN GH_TOKEN=$GH_TOKEN bash scripts/stop_and_eval.sh /dev/shm/runs/loop384_r2 4500 loop4500 2>&1 | tee -a logs/stop45.log; echo STOP45_EXIT; sleep 86400"
tmux new-session -d -s hub2 "cd /workspace/loop-dit && HF_TOKEN=$HF_TOKEN PYTHONPATH=/workspace/loop-dit flux-action/.venv/bin/python scripts/hub_backup.py --run /dev/shm/runs/loop384_r2 --repo Jinyan0924/flux3-agibot-looped --full --delete-local 2>&1 | tee -a logs/hub2.log; echo HUB2_EXIT; sleep 86400"
sleep 5; tmux ls; echo RELAUNCHED
