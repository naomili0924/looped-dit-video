#!/bin/bash
# GPU 1, part B. Pass 1: k=1 for all 4 models, metrics. Pass 2: k=1+9 (finished k=1 cells are cached), metrics.
cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3
set -a; . /workspace/.env; set +a
export MIRROR=$PWD/eval-splits TEXT_ENCODER=$PWD/flux-action/outputs/weights/text_encoder
export FT_EMA=$PWD/weights/ft/ema_step-6000.safetensors LOOP2500_EMA=$PWD/weights/loop/ema_loop2500_step-2500.safetensors
ts() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a logs/gpu1_chain.log; }
pass() {  # $1 K  $2 tag
  [ -f logs/motion_gen_gpu1.log ] && mv logs/motion_gen_gpu1.log "logs/motion_gen_gpu1.$(date -u +%H%M%S).log"
  ts "start motion_gen K=$1"
  K="$1" GPUS="1" scripts/run_motion_gen.sh 50 2>&1 | tee -a logs/gpu1_chain.log
  until grep -q MOTION_GEN_GPU1_DONE logs/motion_gen_gpu1.log 2>/dev/null; do sleep 60; done
  ts "end motion_gen K=$1"
  scripts/run_motion_metrics.sh cuda:1 2>&1 | tee -a logs/motion_metrics.log
  ts "$2"
}
pass 1 MOTION_K1_DONE
pass "1 9" GPU1_CHAIN_DONE
