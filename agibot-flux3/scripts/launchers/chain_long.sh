#!/bin/bash
# Long random-depth fine-tune (configs/looped_rand_384_long.json): resume from the pre-cooldown step-2000 of
# /dev/shm/runs/looprand at constant LR 1e-5 until the val-loss early stop (3 consecutive rises) or 20000 steps.
# Waits for the 2500-step run to exit, then launches it and its Hub sidecar (EMA latest+best, full latest).
cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3
A=$PWD; set -a; . /workspace/.env; set +a
RUN=$A/runs/looprand_long; LOG=$A/logs/chain_long.log
ts() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a $LOG; }

mkdir -p $RUN
# hard link (same filesystem, no extra space); keep_checkpoints=1 later removes only this link, not ckpt_backup
[ -d $RUN/step-2000 ] || cp -al ckpt_backup/looprand/step-2000 $RUN/step-2000
# step 2000 is already on the Hub (random_depth_n1-8/): the sidecar must not upload it again
[ -f $RUN/hub_backup.json ] || echo '{"uploaded": [2000], "full_uploaded": [2000]}' > $RUN/hub_backup.json

ts "waiting for the 2500-step run to exit"
idle() { ! pgrep -f '^[^ ]*python[^ ]* .*flux_action[.]cli train' >/dev/null && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
until idle && sleep 60 && idle; do sleep 60; done
[ -f /dev/shm/runs/looprand/step-2500/COMPLETE ] || ts "WARNING: the 2500-step run exited without step-2500"
ts "launching the long run"
bash scripts/launch_looprand.sh $A/configs/looped_rand_384_long.json 2>&1 | tee -a $LOG
tmux kill-session -t hublong 2>/dev/null
tmux new-session -d -s hublong "cd $A && set -a && . /workspace/.env && set +a && HF_HUB_ENABLE_HF_TRANSFER=0 \
  flux-action/.venv/bin/python scripts/hub_backup.py --run $RUN --repo Jinyan0924/flux3-agibot-looped \
  --prefix random_depth_n1-8_long --full --poll 300 2>&1 | tee -a logs/hub_looprand_long.log; sleep 1d"
ts "hub sidecar started (tmux hublong -> random_depth_n1-8_long/)"
