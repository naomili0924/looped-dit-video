#!/bin/bash
# Stop the training run (torchrun + ranks) and wait until the GPUs are free. Safe to call when nothing runs.
# The [.] keeps pgrep from matching this script's own command line.
# Anchored on the python binary so the tmux/sh wrappers carrying the same command line do not match.
pids=$(pgrep -f '^[^ ]*python[^ ]* .*flux_action[.]cli train'; pgrep -f '^[^ ]*python[^ ]* [^ ]*torchrun .*flux_action[.]cli')
if [ -z "$pids" ]; then echo "no trainer running"; exit 0; fi
echo "stopping: $pids"
kill $pids 2>/dev/null
for i in $(seq 1 60); do
  sleep 2
  pgrep -f '^[^ ]*python[^ ]* .*flux_action[.]cli train' >/dev/null || break
done
pgrep -f '^[^ ]*python[^ ]* .*flux_action[.]cli train' >/dev/null && { echo "still alive, SIGKILL"; kill -9 $(pgrep -f '^[^ ]*python[^ ]* .*flux_action[.]cli train') 2>/dev/null; sleep 3; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "trainer stopped at $(date -u)"
