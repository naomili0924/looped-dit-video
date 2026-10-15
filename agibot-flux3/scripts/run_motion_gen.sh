#!/bin/bash
# Motion-aware evaluation, generation stage (GPU): every model x {train, test, test_unseen} x k {1, 9}, N windows,
# clips saved for the offline analysis (scripts/run_motion_metrics.sh). No FVD, no judge.
#   scripts/run_motion_gen.sh [N=50]
# Models (name|config|ema|loops); edit MODELS to add the 5000-step run once it exists.
set -e
cd /workspace/loop-dit
N=${1:-50}
set -a; . /workspace/.env; set +a
export PYTHONPATH=/workspace/loop-dit HF_HUB_ENABLE_HF_TRANSFER=0
E="cd /workspace/loop-dit/flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python -m ldvflux.eval_phase4"
COMMON="--splits train test test_unseen --k 1 9 --n-recon $N --n-fvd 0 --fvd-k --no-fvd --no-judge --n-judge 0 --batch 4 --seed 0 --save-clips"
MODELS=(
  "base|configs/video_ft_384.json||"
  "ft_step6000|configs/video_ft_384.json|/dev/shm/ema/ema_ft384_step-6000.safetensors|"
  "loop2500_loops1|configs/looped_ft_384_r1.json|/dev/shm/ema/ema_loop2500_step-2500.safetensors|1"
  "loop2500_loops4|configs/looped_ft_384_r1.json|/dev/shm/ema/ema_loop2500_step-2500.safetensors|4"
)
if [ -f /dev/shm/ema/ema_loop5000_step-5000.safetensors ]; then
  MODELS+=("loop5000_loops1|configs/looped_ft_384_r2.json|/dev/shm/ema/ema_loop5000_step-5000.safetensors|1"
           "loop5000_loops4|configs/looped_ft_384_r2.json|/dev/shm/ema/ema_loop5000_step-5000.safetensors|4")
fi
# split the model list over the two GPUs
cmd0=""; cmd1=""; i=0
for m in "${MODELS[@]}"; do
  IFS='|' read -r name cfg ema loops <<< "$m"
  args="--name $name --config /workspace/loop-dit/$cfg $COMMON --out ../results/motion_gen/$name"
  [ -n "$ema" ] && args="$args --ema $ema"
  [ -n "$loops" ] && args="$args --loops $loops"
  if [ $((i % 2)) -eq 0 ]; then cmd0="$cmd0 $E $args --device cuda:0 ;"; else cmd1="$cmd1 $E $args --device cuda:1 ;"; fi
  i=$((i+1))
done
mkdir -p results/motion_gen logs
for g in 0 1; do tmux kill-session -t mgen$g 2>/dev/null || true; done
tmux new-session -d -s mgen0 "( $cmd0 ) 2>&1 | tee -a /workspace/loop-dit/logs/motion_gen_gpu0.log; echo MOTION_GEN_GPU0_DONE | tee -a /workspace/loop-dit/logs/motion_gen_gpu0.log; sleep 1d"
tmux new-session -d -s mgen1 "( $cmd1 ) 2>&1 | tee -a /workspace/loop-dit/logs/motion_gen_gpu1.log; echo MOTION_GEN_GPU1_DONE | tee -a /workspace/loop-dit/logs/motion_gen_gpu1.log; sleep 1d"
echo "motion-eval generation launched at $(date -u): ${#MODELS[@]} models, $N windows x {train,test,test_unseen} x k{1,9}; tmux mgen0/mgen1, logs/motion_gen_gpu*.log"
