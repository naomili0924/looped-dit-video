#!/bin/bash
# Long random-depth fine-tune (configs/looped_rand_384_long.json): resume from the pre-cooldown step-2000 of
# /dev/shm/runs/looprand, LR 1e-5 constant to 9000, linear decay to 10000; early stop when the mean k9 val loss over
# N=1..8 has no new best for 4 evals. Waits for the 2500-step run to exit, smoke-tests the eval hook (2 updates + one
# eval, /dev/shm/runs/evalsmoke.json), falls back to the old eval settings if the smoke fails, launches, then the Hub
# sidecar (EMA latest+best, full latest, step 9000 always kept).
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
ts "eval smoke test"
( cd flux-action && PYTHONPATH=$A FLUX_ACTION_PG_TIMEOUT_MINUTES=10 OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True \
  NCCL_DEBUG=WARN timeout 45m .venv/bin/python .venv/bin/torchrun --nproc_per_node 2 --master_port 29701 -m flux_action.cli train \
  --config /dev/shm/runs/evalsmoke.json ) > logs/evalsmoke.log 2>&1
CFG=$A/configs/looped_rand_384_long.json
if grep '"event": "eval"' logs/evalsmoke.log | grep -q '"train_ref_loss"' && grep '"event": "eval"' logs/evalsmoke.log | grep -q '"val_loss_k9_loops7"'; then
  ts "smoke OK: $(grep '"event": "eval"' logs/evalsmoke.log | cut -c1-400)"
else
  ts "SMOKE FAILED (tail: $(grep -v post_backward logs/evalsmoke.log | tail -3 | tr '\n' ' ' | cut -c1-600)); falling back to the old eval settings"
  CFG=$A/configs/looped_rand_384_long_oldeval.json
  python3 -c "
import json; c=json.load(open('$A/configs/looped_rand_384_long.json'))
for k in ('train_ref_windows', 'stop_rule', 'stop_metric', 'eval_loops'): c['dataset_kwargs']['eval'].pop(k, None)
c['dataset_kwargs']['eval']['patience'] = 3
json.dump(c, open('$CFG', 'w'), indent=2)"
fi
rm -rf /dev/shm/runs/evalsmoke
idle_wait() { until idle; do sleep 10; done; }; idle_wait
ts "launching the long run with $CFG"
bash scripts/launch_looprand.sh $CFG 2>&1 | tee -a $LOG
tmux kill-session -t hublong 2>/dev/null
tmux new-session -d -s hublong "cd $A && set -a && . /workspace/.env && set +a && HF_HUB_ENABLE_HF_TRANSFER=0 \
  flux-action/.venv/bin/python scripts/hub_backup.py --run $RUN --repo Jinyan0924/flux3-agibot-looped \
  --prefix random_depth_n1-8_long --full --keep-steps 9000 --poll 300 2>&1 | tee -a logs/hub_looprand_long.log; sleep 1d"
ts "hub sidecar started (tmux hublong -> random_depth_n1-8_long/)"
