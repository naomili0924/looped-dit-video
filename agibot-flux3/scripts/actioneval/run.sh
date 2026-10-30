#!/bin/bash
# Action eval, end to end (from agibot-flux3/; README: results/actioneval/README.md):
#   index -> pool (+ episode mp4s) -> VLM scene -> candidates -> VLM feasibility -> balanced cases
#   -> per depth N: rollouts (all seeds) on both GPUs -> VLM judge on both GPUs -> report
# Every step skips work already done, so the script can be re-run after an interruption.
#   scripts/actioneval/run.sh
#   STEPS="rollout judge report" DEPTHS="2 4 8" scripts/actioneval/run.sh
# Needs: weights in /dev/shm/w (scripts/capeval/get_weights.sh, Qwen3-VL-32B in /dev/shm/w/qwen3vl32b), the
# flux-action venv (scripts/capeval/setup_env.sh), HF_TOKEN (or /workspace/.hf_home/token) for the episode download.
set -eu
root=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
cd "$root"
STEPS=${STEPS:-"index pool scene propose feasible select rollout judge report"}
PER_ACTION=${PER_ACTION:-3}
SEEDS=${SEEDS:-"0 1 2"}
DEPTHS=${DEPTHS:-"1 2 4 8"}
EMA=${EMA:-/dev/shm/w/looped/random_depth_n1-8_long/ema_step-9500.safetensors}
DATA=${DATA:-/dev/shm/actioneval}  # splits (windows/*.jsonl, episodes/*.json) and beta/task_info
MIRROR=${MIRROR:-/workspace/loop_dit_eval/data/actioneval/mirror}
VLM=${VLM:-/dev/shm/w/qwen3vl32b}
A=results/actioneval
RUN=${RUN:-$A/rlong9500}
CASES=${CASES:-configs/eval_cases_action.json}
LOG=${LOG:-$A/logs}
NGPU=${NGPU:-2}
export HF_TOKEN=${HF_TOKEN:-$(cat /workspace/.hf_home/token 2>/dev/null || true)}
export PYTHONPATH=$root
mkdir -p "$LOG" "$RUN"
# One driver per run dir: the lock fd is inherited by every rollout / judge process, so a second run.sh (a resume)
# waits here until the previous one *and its orphaned workers* have exited. (A pgrep-based waiter can match its own
# shell's command line and wait forever.)
exec 9> "$RUN/.lock"
flock -n 9 || { echo "waiting for the previous driver on $RUN to finish" >&2; flock 9; }
py=flux-action/.venv/bin/python
has() { [[ " $STEPS " == *" $1 "* ]]; }

on_gpus() {  # on_gpus <log name> <module> <args...>: one shard per GPU, wait for all
  local name=$1 mod=$2
  shift 2
  local pids=()
  for ((i = 0; i < NGPU; i++)); do
    (cd flux-action && PYTHONPATH=$root .venv/bin/python -m "$mod" "$@" --shard "$i/$NGPU" --device "cuda:$i") \
      > "$LOG/${name}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

if has index; then
  [ -d "$DATA/splits/windows" ] || /venv/main/bin/hf download Jinyan0924/agibot-flux3-eval-splits --repo-type dataset \
    windows/train.jsonl windows/val.jsonl windows/test.jsonl windows/test_unseen.jsonl episodes/val.json episodes/test.json \
    episodes/test_unseen.json --local-dir "$DATA/splits"
  [ -d "$DATA/beta/task_info" ] || /venv/main/bin/hf download agibot-world/AgiBotWorld-Beta --repo-type dataset \
    --include 'task_info/*' --local-dir "$DATA/beta"
  python3 -I scripts/actioneval/build_index.py "$DATA/splits/windows" "$DATA/beta/task_info" $A/action_index.json
fi
has pool && python3 -I scripts/actioneval/pool.py "$DATA/splits" "$DATA/beta/task_info" $A/pool.json --per-task 6 --download "$MIRROR"
has scene && on_gpus scene ldvflux.act_vlm --mode scene --pool ../$A/pool.json --mirror "$MIRROR" --vlm "$VLM" --out ../$A
has propose && python3 -I scripts/actioneval/propose.py $A
has feasible && on_gpus feasible ldvflux.act_vlm --mode feasible --pool ../$A/candidates.json --mirror "$MIRROR" --vlm "$VLM" --out ../$A
has select && python3 -I scripts/actioneval/select.py $A "$CASES" --per-action "$PER_ACTION"
for d in $DEPTHS; do
  # shellcheck disable=SC2086
  has rollout && on_gpus "rollout_N$d" ldvflux.cap_eval --config ../configs/looped_ft_384_random_n1-8.json --ema "$EMA" \
    --cases "../$CASES" --out "../$RUN" --seeds $SEEDS --depths "$d" --text-encoder outputs/weights/text_encoder --mirror "$MIRROR"
  has judge && on_gpus "judge_N$d" ldvflux.act_judge --run "../$RUN" --cases "../$CASES" --judge "$VLM"
  has report && $py -I scripts/actioneval/report.py "$CASES" "$RUN" > "$RUN/report.md"
done
echo ACTION_EVAL_DONE
