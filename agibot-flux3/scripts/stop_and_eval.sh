#!/bin/bash
# Stop the r2 run right after the step-STEP checkpoint + eval (user decision 2026-10-16 20:05 UTC: stop at 4500, no
# cooldown, evaluate), then finish (Hub stable name, GitHub), fetch the Qwen3-VL judge if missing, launch the loop eval.
#   HF_TOKEN=... GH_TOKEN=... scripts/stop_and_eval.sh [RUN=/dev/shm/runs/loop384_r2] [STEP=4500] [TAG=loop4500] [CFG]
set -u
RUN=${1:-/dev/shm/runs/loop384_r2}; STEP=${2:-4500}; TAG=${3:-loop4500}; CFG=${4:-/workspace/loop-dit/configs/looped_ft_384_r2.json}
LOG=/workspace/loop-dit/logs/$(basename "$RUN").log
EMA=/dev/shm/ema/ema_${TAG}_step-$STEP.safetensors
PY=/workspace/loop-dit/flux-action/.venv/bin/python
: "${HF_TOKEN:?HF_TOKEN}"; : "${GH_TOKEN:?GH_TOKEN}"
export HF_TOKEN GH_TOKEN PYTHONPATH=/workspace/loop-dit
log() { echo "[stop $(date -u +%H:%M:%S)] $*"; }
cd /workspace/loop-dit
log "waiting for the step-$STEP eval event in $LOG"
# done when the trainer logged the step-STEP eval, or when it wrote the step-STEP checkpoint and then died before the
# eval (happened 2026-10-16 21:50 UTC: rank 1 SIGKILLed 3 min after the checkpoint; the val point is then missing)
has_eval() { grep -q "\"event\": \"eval\", \"step\": $STEP," "$LOG"; }
has_ckpt() { grep -q "\"event\": \"checkpoint\", \"step\": $STEP," "$LOG"; }
alive() { pgrep -f 'flux_action[.]cli train' | xargs -r ps -o comm= -p 2>/dev/null | grep -q python; }  # the tmux wrapper shells also match the pattern
until has_eval || { has_ckpt && ! alive; }; do sleep 20; done
has_eval && log "step $STEP checkpoint + eval done; stopping the trainer" || log "step $STEP checkpoint written but the trainer died before its eval; finishing without the step-$STEP val point"
# never `pkill -f` on this pattern: the tmux server's argv contains the launch command too (it killed every tmux session on
# 2026-10-16 22:08 UTC); only signal python/torch processes
kill_train() { pgrep -f 'flux_action[.]cli train' | xargs -r ps -o pid=,comm= -p 2>/dev/null | awk '$2 ~ /python|pt_main/ {print $1}' | xargs -r kill "$@"; }
kill_train; sleep 15
kill_train -9 2>/dev/null; pkill -f 'multiprocessing[.]forkserver'; pkill -f 'multiprocessing[.]resource_tracker'
tmux kill-session -t train 2>/dev/null
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ] && break; sleep 5; done
echo TRAIN_EXITED >> "$LOG"   # finish_r2.sh keys on this marker
log "trainer stopped (GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '))"
bash scripts/finish_r2.sh "$RUN" "$STEP" "$TAG"; rc=$?
log "finish_r2.sh rc $rc"
if [ ! -f "$EMA" ]; then
  log "local EMA missing; downloading ema_step-$STEP from the Hub"
  $PY - "$STEP" "$EMA" <<'PY'
import os, sys, shutil
from huggingface_hub import hf_hub_download
step, out = int(sys.argv[1]), sys.argv[2]
p = hf_hub_download("Jinyan0924/flux3-agibot-looped", f"ema_step-{step}.safetensors", token=os.environ["HF_TOKEN"], local_dir="/dev/shm/ema/dl")
shutil.move(p, out); print("ema ->", out)
PY
fi
JUDGE=/workspace/loop-dit/flux-action/outputs/weights/text_encoder
if [ ! -f "$JUDGE/config.json" ]; then
  log "fetching the Qwen3-VL judge/text encoder into $JUDGE"
  $PY - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("black-forest-labs/flux-3-action-base", token=os.environ["HF_TOKEN"], allow_patterns=["text_encoder/*"],
                  local_dir="/dev/shm/flux3/weights")
PY
  ls "$JUDGE" | head -3; du -sh "$JUDGE"
fi
df -h /dev/shm | tail -1
log "launching the loop-depth eval"
bash scripts/run_loop_eval.sh "$STEP" "$RUN" "$TAG" "$CFG" && echo STOP_EVAL_LAUNCHED || echo STOP_EVAL_LAUNCH_FAILED
