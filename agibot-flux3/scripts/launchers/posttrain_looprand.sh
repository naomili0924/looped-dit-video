#!/bin/bash
# After the random-depth run (configs/looped_rand_384_r1.json): wait for step-2500 + free GPUs, consolidate the EMA,
# copy the final checkpoint off /dev/shm, then rollouts (N=1,2,4,8,16) and motion gen (k=1, test + test_unseen,
# 50 windows) on both GPUs, then motion metrics.
cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3
A=$PWD; set -a; . /workspace/.env; set +a
export PYTHONPATH=$A HF_HUB_ENABLE_HF_TRANSFER=0
RUN=/dev/shm/runs/looprand; STEP=${STEP:-2500}
EMA=$A/weights/looprand/ema_step-$STEP.safetensors
CFG=configs/looped_rand_384_r1.json
L2500=$A/weights/loop/ema_loop2500_step-2500.safetensors
LOG=$A/logs/posttrain.log
ts() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a $LOG; }

ts "waiting for $RUN/step-$STEP/COMPLETE and idle GPUs"
until [ -f $RUN/step-$STEP/COMPLETE ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
ts "training finished"

mkdir -p weights/looprand
if [ ! -f $EMA ]; then
  flux-action/.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
from pathlib import Path; from safetensors.torch import save_file; from hub_backup import consolidate_ema
save_file(consolidate_ema(Path('$RUN/step-$STEP')), '$EMA.tmp', metadata={'step': '$STEP', 'run': '$RUN'})" && mv $EMA.tmp $EMA
fi
[ -f $EMA ] || { ts "EMA consolidation FAILED"; exit 1; }
ts "EMA -> $EMA"
# /dev/shm is RAM: keep the resumable checkpoint on disk too (CPU, in the background)
( mkdir -p $A/ckpt_backup/looprand && cp -r $RUN/step-$STEP $RUN/config.json $RUN/metrics.jsonl $RUN/eval $A/ckpt_backup/looprand/ \
  && ts "checkpoint copied to ckpt_backup/looprand" ) &
CP=$!

# rollouts, balanced (~45 min per GPU)
gpu_jobs() {  # $1 gpu, rest: "depths|tag" ...
  g=$1; shift
  for j in "$@"; do IFS='|' read -r d t <<< "$j"; bash logs/rollout2.sh $g looprand $CFG $EMA "$d" "0 1 2" $t; done
}
gpu_jobs 0 "16|n16" "1|n1" & R0=$!
gpu_jobs 1 "8|n8" "4|n4" "2|n2" & R1=$!
wait $R0 $R1
ts "rollouts done"

# motion gen, k=1, same windows + seed as results/motion_gen (run_motion_gen.sh settings)
E="cd $A/flux-action && .venv/bin/python -m ldvflux.eval_phase4"
COMMON="--splits test test_unseen --k 1 --n-recon 50 --n-fvd 0 --fvd-k --no-fvd --no-judge --n-judge 0 --batch 4 --seed 0 --save-clips --mirror $A/eval-splits --text-encoder $A/flux-action/outputs/weights/text_encoder"
mgen() {  # $1 gpu, rest: "name|ema|loops|config" ...
  g=$1; shift
  for j in "$@"; do IFS='|' read -r n e l c <<< "$j"
    ts "motion_gen $n cuda:$g"
    eval "$E --name $n --config $A/$c $COMMON --out $A/results/motion_gen/$n --ema $e --loops $l --device cuda:$g" 2>&1 \
      | grep --line-buffered -v Warning >> $A/logs/motion_gen_post_gpu$g.log
    ts "motion_gen $n done (${PIPESTATUS[0]})"
  done
}
mgen 0 "looprand_loops8|$EMA|8|$CFG" "looprand_loops1|$EMA|1|$CFG" "loop2500_loops2|$L2500|2|configs/looped_ft_384_r1.json" & M0=$!
mgen 1 "looprand_loops4|$EMA|4|$CFG" "looprand_loops2|$EMA|2|$CFG" "loop2500_loops8|$L2500|8|configs/looped_ft_384_r1.json" & M1=$!
wait $M0 $M1 $CP
ts "motion gen done"
scripts/run_motion_metrics.sh cuda:0 2>&1 | tee -a logs/motion_metrics.log
ts "POSTTRAIN_DONE"
