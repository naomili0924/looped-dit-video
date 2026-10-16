#!/bin/bash
# After the r2 (5000-step looped) run ends: make sure the final weights are on the Hub under their final names and
# push the run's logs/notes to GitHub. Runs unattended in tmux next to the trainer and the hub_backup sidecar.
#   HF_TOKEN=... GH_TOKEN=... scripts/finish_r2.sh [RUN=/dev/shm/runs/loop384_r2] [STEP=5000] [TAG=loop5000]
# Steps: wait for step-STEP/COMPLETE (or an early trainer exit) -> export the EMA locally while the step dir still
# exists (the sidecar deletes it after its upload) -> wait for the sidecar's uploads (ema_step-N + checkpoints/step-N)
# -> server-side copy ema_step-N -> ema_<TAG>_step-N (the name EVAL_MOTION.md / run_loop_eval.sh expect) -> commit
# metrics + val curve + an auto note to the private repo and mirror NOTES/results into the public one.
set -u
RUN=${1:-/dev/shm/runs/loop384_r2}; STEP=${2:-5000}; TAG=${3:-loop5000}
REPO=Jinyan0924/flux3-agibot-looped
LOG=/workspace/loop-dit/logs/$(basename "$RUN").log
EMA=/dev/shm/ema/ema_${TAG}_step-$STEP.safetensors
PY=/workspace/loop-dit/flux-action/.venv/bin/python
export PYTHONPATH=/workspace/loop-dit
: "${HF_TOKEN:?HF_TOKEN}"; : "${GH_TOKEN:?GH_TOKEN}"
log() { echo "[finish $(date -u +%H:%M:%S)] $*"; }

log "waiting for $RUN/step-$STEP/COMPLETE or trainer exit"
while [ ! -f "$RUN/step-$STEP/COMPLETE" ] && ! grep -q TRAIN_EXITED "$LOG" 2>/dev/null; do sleep 60; done
if [ ! -f "$RUN/step-$STEP/COMPLETE" ]; then
  last=$(grep '"step"' "$LOG" | tail -1 | $PY -c "import sys,json; print(json.loads(sys.stdin.read()).get('step'))")
  log "trainer exited before step $STEP (last logged step $last)"; echo FINISH_TRAIN_EXITED_EARLY; exit 1
fi
log "step $STEP complete"
if [ ! -f "$EMA" ] && [ -d "$RUN/step-$STEP" ]; then
  $PY /workspace/loop-dit/scripts/export_ema.py "$RUN/step-$STEP" "$EMA" && log "exported $EMA" || log "local EMA export failed (will fall back to the Hub copy)"
fi
for i in $(seq 1 90); do grep -q TRAIN_EXITED "$LOG" && break; sleep 60; done
grep -q TRAIN_EXITED "$LOG" && log "trainer exited" || log "trainer still alive after 90 min; continuing anyway"

log "waiting for the sidecar's Hub uploads"
$PY - "$REPO" "$STEP" "$TAG" <<'PY'
import os, sys, time
from huggingface_hub import HfApi, CommitOperationCopy
repo, step, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
want = {f"ema_step-{step}.safetensors", f"checkpoints/step-{step}/COMPLETE"}
for _ in range(180):  # up to 3 h
    files = set(api.list_repo_files(repo))
    if want <= files:
        break
    time.sleep(60)
else:
    print("hub uploads not complete:", want - files, flush=True); sys.exit(2)
final = f"ema_{tag}_step-{step}.safetensors"
if final not in files:
    api.create_commit(repo, operations=[CommitOperationCopy(src_path_in_repo=f"ema_step-{step}.safetensors", path_in_repo=final)],
                      commit_message=f"{tag}: final EMA (step {step}) under its stable name")
    print("copied ->", final, flush=True)
else:
    print("already on the Hub:", final, flush=True)
print("hub ok:", sorted(f for f in api.list_repo_files(repo) if "step-%d" % step in f), flush=True)
PY
hub_rc=$?
[ $hub_rc -eq 0 ] && log "Hub: final weights in place" || log "Hub step failed (rc $hub_rc); continuing with GitHub"

cd /workspace/loop-dit || exit 1
mkdir -p results
cp "$RUN/metrics.jsonl" results/${TAG}_metrics.jsonl 2>/dev/null
cp "$RUN/eval/val.jsonl" results/${TAG}_val.jsonl 2>/dev/null
final_val=$(tail -1 "$RUN/eval/val.jsonl" 2>/dev/null)
steps_here=$(grep -c '"step_time"' "$LOG")
{
  echo
  echo "## $(date -u '+%Y-%m-%d %H:%M UTC') — r2 ($TAG) finished at step $STEP (auto entry, scripts/finish_r2.sh)"
  echo "- Final val line (EMA): \`$final_val\`"
  echo "- Hub: \`$REPO\` has \`ema_${TAG}_step-$STEP.safetensors\` (= \`ema_step-$STEP.safetensors\`) and the full checkpoint under \`checkpoints/step-$STEP\` (hub step rc $hub_rc)."
  echo "- Logs: results/${TAG}_metrics.jsonl, results/${TAG}_val.jsonl ($steps_here steps logged on the H100 box). Next: loop-depth eval (scripts/run_loop_eval.sh $STEP $RUN $TAG configs/looped_ft_384_r2.json), gallery/RESULTS update."
} >> NOTES.md
git add -A NOTES.md results/${TAG}_metrics.jsonl results/${TAG}_val.jsonl
git commit -qm "r2 ($TAG) finished at step $STEP: metrics, val curve, auto note" && \
git push -q "https://naomili0924:${GH_TOKEN}@github.com/naomili0924/loop-dit.git" HEAD:main && log "pushed private repo" || log "private push failed"
cd /workspace/loop_dit_video && cp /workspace/loop-dit/NOTES.md agibot-flux3/ && \
cp /workspace/loop-dit/results/${TAG}_val.jsonl /workspace/loop-dit/results/${TAG}_metrics.jsonl agibot-flux3/results/ 2>/dev/null; \
git add -A agibot-flux3/NOTES.md agibot-flux3/results/${TAG}_val.jsonl agibot-flux3/results/${TAG}_metrics.jsonl && \
git commit -qm "agibot-flux3: r2 ($TAG) finished at step $STEP: metrics, val curve, auto note" && \
git push -q "https://naomili0924:${GH_TOKEN}@github.com/naomili0924/looped-dit-video.git" HEAD:main && log "pushed public repo" || log "public push failed"
echo FINISH_DONE
