#!/bin/bash
# Motion-aware evaluation, generation stage (GPU): every model x splits x k {1, 9}, N windows, clips saved for the
# offline analysis (scripts/run_motion_metrics.sh). No FVD, no judge.
#   scripts/run_motion_gen.sh [N=50]
# Environment (all optional):
#   MIRROR=<dir>          local snapshot of Jinyan0924/agibot-flux3-eval-splits -> splits test test_unseen (no train off-box)
#   TEXT_ENCODER=<dir>    Qwen3-VL directory (default flux-action/outputs/weights/text_encoder); needed off the training box
#   SPLITS="..."          default "train test test_unseen" (or "test test_unseen" when MIRROR is set)
#   FT_EMA, LOOP2500_EMA, LOOP5000_EMA   EMA safetensors (defaults: /dev/shm/ema/...); a missing file skips that model
#   GPUS="0 1"            GPU indices to spread the models over (default: all visible, max 2)
#   MODELS_ONLY="base ft_step6000 loop2500_loops1 loop2500_loops4 loop5000_loops1 loop5000_loops4"  subset to run
set -e
cd "$(dirname "$0")/.."
ROOT=$PWD
N=${1:-50}
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }
export PYTHONPATH=$ROOT HF_HUB_ENABLE_HF_TRANSFER=0
TE=${TEXT_ENCODER:-$ROOT/flux-action/outputs/weights/text_encoder}
FT_EMA=${FT_EMA:-/dev/shm/ema/ema_ft384_step-6000.safetensors}
LOOP2500_EMA=${LOOP2500_EMA:-/dev/shm/ema/ema_loop2500_step-2500.safetensors}
LOOP5000_EMA=${LOOP5000_EMA:-/dev/shm/ema/ema_loop5000_step-5000.safetensors}
if [ -n "$MIRROR" ]; then SPLITS=${SPLITS:-"test test_unseen"}; else SPLITS=${SPLITS:-"train test test_unseen"}; fi
if [ -z "$GPUS" ]; then n=$(nvidia-smi -L | wc -l); [ "$n" -ge 2 ] && GPUS="0 1" || GPUS="0"; fi
E="cd $ROOT/flux-action && PYTHONPATH=$ROOT .venv/bin/python -m ldvflux.eval_phase4"
COMMON="--splits $SPLITS --k 1 9 --n-recon $N --n-fvd 0 --fvd-k --no-fvd --no-judge --n-judge 0 --batch 4 --seed 0 --save-clips"
[ -n "$MIRROR" ] && COMMON="$COMMON --mirror $MIRROR"
# off the training box the config's lookup: caption table does not exist -> real text encoder
[ -n "$MIRROR" ] || [ -n "$TEXT_ENCODER" ] && COMMON="$COMMON --text-encoder $TE"
MODELS=("base|configs/video_ft_384.json||")
[ -f "$FT_EMA" ] && MODELS+=("ft_step6000|configs/video_ft_384.json|$FT_EMA|")
[ -f "$LOOP2500_EMA" ] && MODELS+=("loop2500_loops1|configs/looped_ft_384_r1.json|$LOOP2500_EMA|1"
                                   "loop2500_loops4|configs/looped_ft_384_r1.json|$LOOP2500_EMA|4")
[ -f "$LOOP5000_EMA" ] && MODELS+=("loop5000_loops1|configs/looped_ft_384_r2.json|$LOOP5000_EMA|1"
                                   "loop5000_loops4|configs/looped_ft_384_r2.json|$LOOP5000_EMA|4")
# spread the model list over the GPUs
declare -A CMD; i=0; gpus=($GPUS)
for m in "${MODELS[@]}"; do
  IFS='|' read -r name cfg ema loops <<< "$m"
  if [ -n "$MODELS_ONLY" ] && ! grep -qw "$name" <<< "$MODELS_ONLY"; then continue; fi
  args="--name $name --config $ROOT/$cfg $COMMON --out $ROOT/results/motion_gen/$name"
  [ -n "$ema" ] && args="$args --ema $ema"
  [ -n "$loops" ] && args="$args --loops $loops"
  g=${gpus[$((i % ${#gpus[@]}))]}
  CMD[$g]="${CMD[$g]} $E $args --device cuda:$g ;"
  i=$((i+1))
done
mkdir -p results/motion_gen logs
for g in "${gpus[@]}"; do
  tmux kill-session -t mgen$g 2>/dev/null || true
  tmux new-session -d -s mgen$g "( ${CMD[$g]} ) 2>&1 | tee -a $ROOT/logs/motion_gen_gpu$g.log; echo MOTION_GEN_GPU${g}_DONE | tee -a $ROOT/logs/motion_gen_gpu$g.log; sleep 1d"
done
echo "motion-eval generation launched at $(date -u): $i models, $N windows x {$SPLITS} x k{1,9}; tmux mgen<gpu>, logs/motion_gen_gpu*.log"
echo "done when every log ends with MOTION_GEN_GPU<g>_DONE; then: scripts/run_motion_metrics.sh cuda:0"
