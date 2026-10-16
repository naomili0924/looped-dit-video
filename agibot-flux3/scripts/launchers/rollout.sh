#!/bin/bash
# rollout.sh <gpu> <name> <cfg> <ema|""> [depths]
cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3
A=$PWD; export PYTHONPATH=$A HF_HUB_ENABLE_HF_TRANSFER=0
G=$1 NAME=$2 CFG=$3 EMA=$4 D=${5:-1}
ea=""; [ -n "$EMA" ] && ea="--ema $EMA"
echo "[$(date -u '+%F %T UTC')] start rollout $NAME cuda:$G" | tee -a logs/rollout.log
( cd flux-action && $A/flux-action/.venv/bin/python -m ldvflux.rollout --config $A/$CFG $ea --cases $A/configs/eval_cases_gallery8.json \
   --mirror $A/eval-splits --text-encoder $A/flux-action/outputs/weights/text_encoder --depths $D --k 1 --rounds 4 \
   --out $A/results/rollout/$NAME --device cuda:$G ) 2>&1 | grep -v Warning | tee -a logs/rollout_$NAME.log
echo "[$(date -u '+%F %T UTC')] end rollout $NAME (${PIPESTATUS[0]})" | tee -a logs/rollout.log
