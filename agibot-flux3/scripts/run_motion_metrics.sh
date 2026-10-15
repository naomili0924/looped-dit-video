#!/bin/bash
# Motion-aware evaluation, analysis stage: pixel + flow metrics on every saved clips cell -> results/motion/*.json,
# summary.json and MOTION.md.  Runs on a GPU when one is free (RAFT), else on CPU with --flow-scale 0.5.
#   scripts/run_motion_metrics.sh [DEVICE=cuda:0] [GEN_DIR=results/motion_gen] [OUT=results/motion]
set -e
cd /workspace/loop-dit
DEV=${1:-cuda:0}; GEN=${2:-results/motion_gen}; OUT=${3:-results/motion}
export PYTHONPATH=/workspace/loop-dit
PY=flux-action/.venv/bin/python
mkdir -p "$OUT"
for cell in "$GEN"/*/clips/*/; do
  model=$(basename "$(dirname "$(dirname "$cell")")"); name=$(basename "$cell")
  res="$OUT/${model}__${name}.json"
  [ -f "$res" ] && { echo "skip $res"; continue; }
  $PY -m ldvflux.clip_eval --clips "$cell" --out "$res" --device "$DEV" 2>&1 | grep -v Warning
done
$PY -m ldvflux.clip_eval --summarize "$OUT" --md "$OUT/MOTION.md"
