#!/bin/bash
# "Where does extra loop depth go" evaluation (ldvflux.depth_eval) for one model: generation at N = 1/2/4/8 from the
# same noise for every case / k / prompt, then the background-vs-motion analysis and strips.
#   scripts/run_depth_eval.sh <name> <config> <ema|""> <cases.json> [DEVICE=cuda:0] [OUT=results/depth_eval]
# e.g. scripts/run_depth_eval.sh loop2500 configs/looped_ft_384_r1.json /dev/shm/ema/ema_loop2500_step-2500.safetensors \
#        configs/eval_cases_gallery8.json cuda:0
# Set MIRROR=<dir> to read the clips from a local snapshot of Jinyan0924/agibot-flux3-eval-splits instead of the
# streaming index (needed on a machine without data/splits.json + dataset access).
set -e
cd "$(dirname "$0")/.."
NAME=${1:?name}; CFG=${2:?config}; EMA=${3:-}; CASES=${4:?cases json}; DEV=${5:-cuda:0}; OUT=${6:-results/depth_eval}
TE=${TEXT_ENCODER:-$PWD/flux-action/outputs/weights/text_encoder}
export PYTHONPATH=$PWD HF_HUB_ENABLE_HF_TRANSFER=0
PY=$PWD/flux-action/.venv/bin/python
args="--config $PWD/$CFG --cases $PWD/$CASES --out $PWD/$OUT/$NAME --device $DEV --text-encoder $TE --depths 1 2 4 8 --k ${K:-1 9} --seed 0"
[ -n "$EMA" ] && args="$args --ema $EMA"
[ -n "$MIRROR" ] && args="$args --mirror $MIRROR"
mkdir -p "$OUT/$NAME" logs
( cd flux-action && $PY -m ldvflux.depth_eval $args ) 2>&1 | grep -v Warning | tee -a "logs/depth_eval_$NAME.log"
$PY -m ldvflux.depth_eval --analyze "$OUT/$NAME" --device "$DEV" 2>&1 | grep -v Warning | tee -a "logs/depth_eval_$NAME.log"
$PY -m ldvflux.depth_eval --summarize "$OUT" --md "$OUT/DEPTH.md" 2>&1 | tail -n 2
echo "DEPTH_EVAL_${NAME}_DONE"
