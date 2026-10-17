#!/bin/bash
# "normal eval" of a looped checkpoint = the "Trained Loop Depth" page: test split, k 1 and 9, loop depths N = 1 2 4 8
# (PSNR/SSIM/LPIPS on 100 windows, FVD on 300, VLM judge on 100), two GPUs (~2.5 h on H100s), then the comparison strips
# and the GitHub Pages site (docs/loop-depth) are rebuilt.
#   HF_TOKEN=... scripts/normal_eval.sh <tag> <step> <config> <ema> [label] [trained depths "4"]
#   e.g. HF_TOKEN=... scripts/normal_eval.sh rand2500 2500 configs/looped_ft_384_random_n1-8.json \
#            hf://Jinyan0924/flux3-agibot-looped/random_depth_n1-8/ema_step-2500.safetensors "random depth N 1-8, step 2500" 1,2,3,4,5,6,7,8
# <ema>: local .safetensors or hf://<user>/<repo>/<path>. Prints NORMAL_EVAL_<tag>_DONE in tmux normal_finish_<tag> when the
# pages are rebuilt; then publish with GH_TOKEN=... scripts/push_all.sh "...".
set -eu
cd "$(dirname "$0")/.."
TAG=${1:?tag}; STEP=${2:?step}; CFG=${3:?config}; SRC=${4:?ema}; LABEL=${5:-"looped, $STEP steps"}; TRAINED=${6:-4}
PY=flux-action/.venv/bin/python
EMA=/dev/shm/ema/ema_${TAG}_step-$STEP.safetensors
HF_TOKEN=${HF_TOKEN:-} $PY scripts/hf_get.py "$SRC" "$EMA"
$PY scripts/register_model.py "$TAG" "$STEP" "$LABEL" "$TRAINED" "$CFG" "$SRC"
[ -f /dev/shm/ema/ema_ft384_step-6000.safetensors ] || echo "note: the Phase-3 reference rows come from results/ft_step6000 (already in the repo)"
HF_TOKEN=${HF_TOKEN:-} bash scripts/run_loop_eval.sh "$STEP" /nonexistent "$TAG" "$PWD/$CFG"
tmux kill-session -t normal_finish_$TAG 2>/dev/null || true
tmux new-session -d -s normal_finish_$TAG "cd $PWD; until grep -q LOOP_EVAL_GPU0_DONE logs/${TAG}_eval_gpu0.log && grep -q LOOP_EVAL_GPU1_DONE logs/${TAG}_eval_gpu1.log; do sleep 60; done; bash scripts/make_looped_strips.sh $TAG && bash scripts/build_pages.sh 2>&1 | tail -n 5; echo NORMAL_EVAL_${TAG}_DONE; sleep 86400"
echo "normal eval of $TAG launched: tmux eval_gpu0 / eval_gpu1 (logs/${TAG}_eval_gpu*.log); finisher tmux normal_finish_$TAG prints NORMAL_EVAL_${TAG}_DONE"
