#!/bin/bash
# Negative-prompt test on the v2 capability cases (object / action / direction prompts, the real caption as the negative)
# at N = 1, seeds 0-2 (the capeval noise), four guidance arms (cfg4 also renders the real prompt as a reference),
# each in results/capeval/negprompt/<arm>. Generation is sharded over both GPUs; then each GPU judges two arms.
#   scripts/capeval/run_negprompt.sh            (from agibot-flux3/; weights in /dev/shm/w, mirror in /dev/shm/capeval)
#   scripts/capeval/run_negprompt.sh cfg7       (only the named arms, judged one after another on GPU 0)
set -eu
root=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
cd "$root/flux-action"
out=$root/results/capeval/negprompt
py=.venv/bin/python
common=(--config ../configs/looped_ft_384_random_n1-8.json
        --ema /dev/shm/w/looped/random_depth_n1-8_long/ema_step-9500.safetensors
        --cases ../configs/eval_cases_capability_v2.json --seeds 0 1 2 --depths 1 --prompts object action direction
        --text-encoder outputs/weights/text_encoder --mirror /dev/shm/capeval/mirror)
declare -A arm=(
  [cfg4]="--cfg 4 --prompts real object action direction"
  [neg4]="--cfg 1 --neg-prompt real --neg-scale 3"
  [neg7]="--cfg 1 --neg-prompt real --neg-scale 6"
  [cfg4neg3]="--cfg 4 --neg-prompt real --neg-scale 3"
  [cfg7]="--cfg 7"  # cfg4neg3's total guidance (4 + 3) without the negative: is its gain the negative or the strength?
)
order=(cfg4 neg4 neg7 cfg4neg3)
(($#)) && order=("$@")
export PYTHONPATH=$root

gen() {  # gen <gpu>: every arm, shard <gpu>/2
  for a in "${order[@]}"; do
    # shellcheck disable=SC2086
    $py -m ldvflux.cap_eval "${common[@]}" ${arm[$a]} --out "$out/$a" --shard "$1/2" --device "cuda:$1"
  done
}
judge() {  # judge <gpu> <arm>...
  local g=$1; shift
  for a in "$@"; do
    $py -m ldvflux.cap_judge --run "$out/$a" --cases ../configs/eval_cases_capability_v2.json --judge /dev/shm/w/qwen3vl32b \
        --device "cuda:$g"
  done
}

gen 0 > "$root/logs/negprompt_gen0.log" 2>&1 &
gen 1 > "$root/logs/negprompt_gen1.log" 2>&1 &
wait
echo GEN_DONE
if (($#)); then
  judge 0 "${order[@]}" > "$root/logs/negprompt_judge0.log" 2>&1
else
  judge 0 cfg4 neg4 > "$root/logs/negprompt_judge0.log" 2>&1 &
  judge 1 neg7 cfg4neg3 > "$root/logs/negprompt_judge1.log" 2>&1 &
  wait
fi
echo NEGPROMPT_DONE
