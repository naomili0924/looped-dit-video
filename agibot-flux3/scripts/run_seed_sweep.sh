#!/bin/bash
# Seed sweep queue of the random-depth long run (Hub Jinyan0924/flux3-agibot-looped/random_depth_n1-8_long), one GPU:
# for each step: fetch the EMA to /dev/shm/ema (the next one is prefetched), roll out seed by seed (ldvflux.seed_sweep),
# rebuild the 5 x 5 grids after every seed, the metrics after the last (ldvflux.seed_metrics), then drop the EMA.
#   HF_TOKEN=... scripts/run_seed_sweep.sh "9500 4000 5000 6000 7000 8000 9000" [cuda:0]
# Env: MIRROR (local snapshot of Jinyan0924/agibot-flux3-eval-splits), SEEDS ("0 1 2 3 4"), DEPTHS ("1 2 4 8 16").
# Log lines for watchers: SWEEP_SEED_DONE <step> <seed>, SWEEP_STEP_DONE <step>, SWEEP_ALL_DONE.
set -u
cd "$(dirname "$0")/.."
A=$PWD; STEPS=${1:?steps}; DEV=${2:-cuda:0}
MIRROR=${MIRROR:-/workspace/loop_dit_eval/mirror}; SEEDS=${SEEDS:-"0 1 2 3 4"}; DEPTHS=${DEPTHS:-"1 2 4 8 16"}
CFG=configs/looped_ft_384_random_n1-8.json; CASES=configs/eval_cases_gallery8.json; SW=$A/results/seedsweep
PY="env PYTHONPATH=$A $A/flux-action/.venv/bin/python"
ema() { echo /dev/shm/ema/ema_rlong_step-$1.safetensors; }
fetch() {  # resumable; a finished file is renamed into place
  [ -f "$(ema $1)" ] && return 0
  local url=https://huggingface.co/Jinyan0924/flux3-agibot-looped/resolve/main/random_depth_n1-8_long/ema_step-$1.safetensors
  for i in 1 2 3 4 5 6; do curl -sSfL -C - -H "Authorization: Bearer $HF_TOKEN" -o "$(ema $1).part" "$url" && mv "$(ema $1).part" "$(ema $1)" && return 0; sleep 20; done
  return 1
}
set -- $STEPS
while [ $# -gt 0 ]; do
  s=$1; shift; next=${1:-}
  fetch $s || { echo "SWEEP_FETCH_FAILED $s"; exit 1; }
  [ -n "$next" ] && { fetch $next & }
  for seed in $SEEDS; do
    (cd flux-action && $PY -m ldvflux.seed_sweep --config ../$CFG --ema "$(ema $s)" --cases ../$CASES --out $SW/rlong$s \
      --seeds $seed --depths $DEPTHS --text-encoder outputs/weights/text_encoder --mirror $MIRROR --device $DEV) || { echo "SWEEP_FAILED $s $seed"; exit 1; }
    $PY -m ldvflux.seed_metrics --sweep $SW/rlong$s --cases $CASES --mirror $MIRROR --grids $SW/grids --tag rlong$s --no-metrics
    echo "SWEEP_SEED_DONE $s $seed $(date -u +%H:%M)"
  done
  $PY -m ldvflux.seed_metrics --sweep $SW/rlong$s --cases $CASES --mirror $MIRROR --device $DEV | tail -n 2 || echo "SWEEP_METRICS_FAILED $s"
  echo "SWEEP_STEP_DONE $s $(date -u +%H:%M)"
  wait
  rm -f "$(ema $s)"
done
echo SWEEP_ALL_DONE
