#!/bin/bash
# Light evaluation of a looped Phase-5 checkpoint at several loop depths, two GPUs in parallel.
#   scripts/run_loop_eval.sh <STEP> [RUN=/dev/shm/runs/loop384] [TAG=loop2500] [CONFIG=configs/looped_ft_384_r1.json]
# Protocol (test split only; the full Phase-4 protocol runs for both looped models at the very end):
#   k 1 and 9, n-recon 100 (PSNR/SSIM/LPIPS), FVD on 300 windows at k 1, VLM judge on 100 windows at k 1.
#   GPU 0: N = 8 then N = 1;  GPU 1: N = 4 then N = 2.  Outputs results/<TAG>_loops<N>/, logs/<TAG>_eval_gpu{0,1}.log.
set -e
cd /workspace/loop-dit
STEP=${1:?step}; RUN=${2:-/dev/shm/runs/loop384}; TAG=${3:-loop2500}; CFG=${4:-/workspace/loop-dit/configs/looped_ft_384_r1.json}
EMA=/dev/shm/ema/ema_${TAG}_step-$STEP.safetensors
PY=flux-action/.venv/bin/python
export PYTHONPATH=/workspace/loop-dit HF_HUB_ENABLE_HF_TRANSFER=0
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }  # tokens may instead come from the environment (HF_TOKEN)
mkdir -p /dev/shm/ema results logs
if [ ! -f "$EMA" ]; then
  [ -f "$RUN/step-$STEP/COMPLETE" ] || { echo "no complete checkpoint $RUN/step-$STEP and no EMA $EMA"; exit 1; }
  $PY scripts/export_ema.py "$RUN/step-$STEP" "$EMA"
fi
ls -la "$EMA"
ARGS="--config $CFG --ema $EMA --splits test --k 1 9 --n-recon 100 --n-fvd 300 --fvd-k 1 --n-judge 100 --judge-k 1 --batch 4 --seed 0"
E="cd /workspace/loop-dit/flux-action && HF_TOKEN=${HF_TOKEN:-} PYTHONPATH=/workspace/loop-dit .venv/bin/python -m ldvflux.eval_phase4"  # tmux sessions do not inherit the caller env
for s in eval_gpu0 eval_gpu1; do tmux kill-session -t $s 2>/dev/null || true; done
tmux new-session -d -s eval_gpu0 "( $E --name ${TAG}_loops8 --loops 8 $ARGS --out ../results/${TAG}_loops8 --device cuda:0 \
  && $E --name ${TAG}_loops1 --loops 1 $ARGS --out ../results/${TAG}_loops1 --device cuda:0 \
  ) 2>&1 | tee -a /workspace/loop-dit/logs/${TAG}_eval_gpu0.log; echo LOOP_EVAL_GPU0_DONE; sleep 1d"
tmux new-session -d -s eval_gpu1 "( $E --name ${TAG}_loops4 --loops 4 $ARGS --out ../results/${TAG}_loops4 --device cuda:1 \
  && $E --name ${TAG}_loops2 --loops 2 $ARGS --out ../results/${TAG}_loops2 --device cuda:1 \
  ) 2>&1 | tee -a /workspace/loop-dit/logs/${TAG}_eval_gpu1.log; echo LOOP_EVAL_GPU1_DONE; sleep 1d"
echo "loop eval of $TAG step $STEP launched at $(date -u): tmux eval_gpu0 (N 8, 1), eval_gpu1 (N 4, 2); logs/${TAG}_eval_gpu*.log"
