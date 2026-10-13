#!/bin/bash
# Phase 4 after the Phase-3 run: fine-tuned EMA (GPU 0) and base zero-shot (GPU 1) in parallel, then the
# zero-shot loop-depth probe of the EMA (both GPUs), then results/RESULTS.md.
#
#   scripts/run_phase4.sh [STEP=6000]          # tmux sessions eval_ft, eval_base; logs/phase4_*.log
#
# Budget (A100, batch 4, Euler 30 steps, CFG 4 ≈ 8 s/clip): test + test_unseen with n-recon 100, n-fvd 300 (k 1, 17),
# judge 100 (k 1) ≈ 1,800 clips ≈ 4 h per model; val recon-only (50 windows, k 0 1 5 9 17) ≈ 35 min; loop probe
# (test, k 1 9, 50 windows, depths 2 4 8 10) ≈ 1 h. Whole phase ≈ 5.5 h wall.
set -e
cd /workspace/loop-dit
STEP=${1:-6000}
RUN=/dev/shm/runs/ft384
EMA=/dev/shm/ema/ema_ft384_step-$STEP.safetensors
PY=flux-action/.venv/bin/python
export PYTHONPATH=/workspace/loop-dit HF_HUB_ENABLE_HF_TRANSFER=0
set -a; . /workspace/.env; set +a

mkdir -p /dev/shm/ema results logs
if [ ! -f "$EMA" ]; then
  if [ -f "$RUN/step-$STEP/COMPLETE" ]; then
    $PY scripts/export_ema.py "$RUN/step-$STEP" "$EMA"
  else  # the sidecar already deleted the local checkpoint: take its consolidated upload
    $PY - <<EOF
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download("Jinyan0924/flux3-agibot-videoft", "ema_step-$STEP.safetensors", token=os.environ["HF_TOKEN"], cache_dir="/dev/shm/hub_restore")
shutil.copy(p, "$EMA"); print("downloaded", p)
EOF
  fi
fi
ls -la "$EMA"

HEAVY="--splits test test_unseen --k 0 1 5 9 17 --n-recon 100 --n-fvd 300 --fvd-k 1 17 --n-judge 100 --judge-k 1 --batch 4 --seed 0"
LIGHT="--splits val --k 0 1 5 9 17 --n-recon 50 --n-fvd 0 --fvd-k --n-judge 0 --no-judge --batch 4 --seed 0"
PROBE="--splits test --k 1 9 --n-recon 50 --n-fvd 0 --fvd-k --n-judge 0 --no-judge --batch 4 --seed 0"
E="cd /workspace/loop-dit/flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python -m ldvflux.eval_phase4"

tmux kill-session -t eval_ft 2>/dev/null || true
tmux kill-session -t eval_base 2>/dev/null || true
# GPU 0: fine-tuned EMA, heavy then val-light, then loop depths 2 and 8
tmux new-session -d -s eval_ft "( $E --name ft_step$STEP --ema $EMA $HEAVY --out ../results/ft_step$STEP --device cuda:0 \
  && $E --name ft_step$STEP --ema $EMA $LIGHT --out ../results/ft_step$STEP --device cuda:0 \
  && $E --name ft_step${STEP}_loops2 --ema $EMA --loops 2 $PROBE --out ../results/ft_step${STEP}_loops2 --device cuda:0 \
  && $E --name ft_step${STEP}_loops8 --ema $EMA --loops 8 $PROBE --out ../results/ft_step${STEP}_loops8 --device cuda:0 \
  ) 2>&1 | tee -a /workspace/loop-dit/logs/phase4_ft.log; echo PHASE4_FT_DONE; sleep 1d"
# GPU 1: base zero-shot, heavy then val-light, then loop depths 4 and 10
tmux new-session -d -s eval_base "( $E --name base $HEAVY --out ../results/base --device cuda:1 \
  && $E --name base $LIGHT --out ../results/base --device cuda:1 \
  && $E --name ft_step${STEP}_loops4 --ema $EMA --loops 4 $PROBE --out ../results/ft_step${STEP}_loops4 --device cuda:1 \
  && $E --name ft_step${STEP}_loops10 --ema $EMA --loops 10 $PROBE --out ../results/ft_step${STEP}_loops10 --device cuda:1 \
  && PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/probe_loops.py --ema $EMA --k 1 9 --depths 1 2 4 8 10 \
       --windows 16 --t-grid 2 --device cuda:1 --out ../results/probe_loops_step$STEP.json \
  ) 2>&1 | tee -a /workspace/loop-dit/logs/phase4_base.log; echo PHASE4_BASE_DONE; sleep 1d"
echo "phase 4 launched at $(date -u): tmux eval_ft (cuda:0), eval_base (cuda:1); logs/phase4_*.log"
echo "when both print PHASE4_*_DONE: $PY scripts/build_results.py --results results --out results/RESULTS.md"
