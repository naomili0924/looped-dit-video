#!/bin/bash
# GPU 0, part A. Pass 1: k=1 only (gallery x3 models, then val set x3). Pass 2: k=1+9 (k=1 clips exist -> skipped).
cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3
set -a; . /workspace/.env; set +a
export MIRROR=$PWD/eval-splits TEXT_ENCODER=$PWD/flux-action/outputs/weights/text_encoder
LOOP=$PWD/weights/loop/ema_loop2500_step-2500.safetensors
FT=$PWD/weights/ft/ema_step-6000.safetensors
ts() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a logs/gpu0_chain.log; }
one() {  # $1 name $2 cfg $3 ema $4 cases $5 out ; K from env
  ts "start $1 $4 K=$K"; touch logs/.gpu0_mark
  scripts/run_depth_eval.sh "$1" "$2" "$3" "$4" cuda:0 "$5"
  [ "$5/$1/analysis.json" -nt logs/.gpu0_mark ] || { ts "FAIL $1 $4 K=$K (analysis.json not rewritten) -> abort chain"; exit 1; }
  ts "end $1 $4 K=$K"
}
run3() {  # $1 cases  $2 out
  one loop2500 configs/looped_ft_384_r1.json "$LOOP" "$1" "$2"
  one ft6000 configs/video_ft_384_r3.json "$FT" "$1" "$2"
  one base configs/video_ft_384_r3.json "" "$1" "$2"
}
if [ ! -f configs/eval_cases_val50.json ]; then
  ts "start select_motion_cases"
  PYTHONPATH=$PWD flux-action/.venv/bin/python scripts/select_motion_cases.py --split val --n-scan 120 --n-keep 50 \
    --mirror $MIRROR --device cuda:0 --out configs/eval_cases_val50.json 2>&1 | grep -v Warning | tee -a logs/select_motion_cases.log
fi
export K=1
run3 configs/eval_cases_gallery8.json results/depth_eval
ts "GALLERY_K1_DONE"
run3 configs/eval_cases_val50.json results/depth_eval_val50
ts "VAL_K1_DONE"
export K="1 9"
run3 configs/eval_cases_gallery8.json results/depth_eval
ts "GALLERY_K9_DONE"
run3 configs/eval_cases_val50.json results/depth_eval_val50
ts "GPU0_CHAIN_DONE"
