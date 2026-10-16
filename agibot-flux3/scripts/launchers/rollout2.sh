#!/bin/bash
# rollout2.sh <gpu> <name> <cfg> <ema|""> <depths> <prompts> <tag>
cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3
A=$PWD; export PYTHONPATH=$A HF_HUB_ENABLE_HF_TRANSFER=0
ea=""; [ -n "$4" ] && ea="--ema $4"
echo "[$(date -u '+%F %T UTC')] start rollout $2 cuda:$1 N=$5 p=$6" | tee -a logs/rollout.log
( cd flux-action && $A/flux-action/.venv/bin/python -m ldvflux.rollout --config $A/$3 $ea --cases $A/configs/eval_cases_gallery8.json \
   --mirror $A/eval-splits --text-encoder $A/flux-action/outputs/weights/text_encoder --depths $5 --prompts $6 --tag $7 --k 1 --rounds 4 ${ONLY:+--only $ONLY} \
   --out $A/results/rollout/$2 --device cuda:$1 ) 2>&1 | grep --line-buffered -v Warning | tee -a logs/rollout_$2_$7.log
echo "[$(date -u '+%F %T UTC')] end rollout $2 N=$5 p=$6 (${PIPESTATUS[0]})" | tee -a logs/rollout.log
