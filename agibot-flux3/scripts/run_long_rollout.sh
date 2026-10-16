#!/bin/bash
# Long-rollout eval of one checkpoint (the "Long Rollout" page): k = 1, 4 chained windows of 33 frames (8.8 s at 15 fps),
# the 8 gallery cases, real + 2 alternative prompts, loop depths 1 2 4 8 16 (plain checkpoints: depth 1). The reference
# columns (ground truth, base FLUX 3 Action zero-shot, Phase-3 fine-tune step 6000) are generated once into
# results/rollout/{gt,base,ft6000} and reused by every later checkpoint.
#   HF_TOKEN=... scripts/run_long_rollout.sh <tag> <config> <ema> [GPUS="0 1"]
# e.g. HF_TOKEN=... scripts/run_long_rollout.sh loop4500 configs/looped_ft_384_r2.json /dev/shm/ema/ema_loop4500_step-4500.safetensors
# Env: TEXT_ENCODER (default flux-action/outputs/weights/text_encoder = Qwen3-VL), FT_EMA (default
# /dev/shm/ema/ema_ft384_step-6000.safetensors; Hub: Jinyan0924/flux3-agibot-videoft ema_step-6000.safetensors),
# MIRROR (local snapshot of Jinyan0924/agibot-flux3-eval-splits on a box without the streaming index), DEPTHS.
# Two GPUs: tmux roll_gpu0 = depths 16 1, roll_gpu1 = base, ft6000, depths 2 4 8 (about 1 h on H100s); one GPU: everything
# in sequence in roll_gpu<g>. Logs: logs/rollout_<tag>_gpu<g>.log, each ending with ROLLOUT_<tag>_GPU<g>_DONE.
# Afterwards: scripts/build_rollout_page.py --models <tag>=<step> [...] (composes the 8-column strips and the page).
set -eu
cd "$(dirname "$0")/.."
TAG=${1:?tag}; CFG=${2:?config}; EMA=${3:?ema}; GPUS=${4:-"0 1"}
TE=${TEXT_ENCODER:-$PWD/flux-action/outputs/weights/text_encoder}
FT_EMA=${FT_EMA:-/dev/shm/ema/ema_ft384_step-6000.safetensors}
DEPTHS=${DEPTHS:-"1 2 4 8 16"}
CASES=$PWD/configs/eval_cases_gallery8.json
OUT=$PWD/results/rollout
[ -f "$EMA" ] || { echo "missing EMA $EMA"; exit 1; }
[ -f "$FT_EMA" ] || { echo "missing Phase-3 EMA $FT_EMA (hf download Jinyan0924/flux3-agibot-videoft ema_step-6000.safetensors)"; exit 1; }
[ -f "$TE/config.json" ] || { echo "missing text encoder $TE"; exit 1; }
mkdir -p logs "$OUT"
common="--cases $CASES --text-encoder $TE --gt-out $OUT/gt --windows 4 --seed 0"
[ -n "${MIRROR:-}" ] && common="$common --mirror $MIRROR"
E="cd $PWD/flux-action && HF_TOKEN=${HF_TOKEN:-} PYTHONPATH=$PWD HF_HUB_ENABLE_HF_TRANSFER=0 .venv/bin/python -m ldvflux.rollout"
refs() {  # base + Phase-3 fine-tune, depth 1, on device $1
  echo "$E --config $PWD/configs/video_ft_384_r3.json --out $OUT/base --depths 1 --device $1 $common && \
        $E --config $PWD/configs/video_ft_384_r3.json --ema $FT_EMA --out $OUT/ft6000 --depths 1 --device $1 $common"
}
model() { echo "$E --config $PWD/$CFG --ema $EMA --out $OUT/$TAG --depths $2 --device $1 $common"; }
set -- $GPUS
if [ $# -ge 2 ]; then
  d_lo=$(echo $DEPTHS | tr ' ' '\n' | sort -n | awk 'NR%2==0' | tr '\n' ' ')   # 2 8  (+ refs)
  d_hi=$(echo $DEPTHS | tr ' ' '\n' | sort -n | awk 'NR%2==1' | tr '\n' ' ')   # 1 4 16
  cmd0="( $(model cuda:$1 "$d_hi") ) 2>&1 | grep --line-buffered -v Warning | tee -a logs/rollout_${TAG}_gpu$1.log; echo ROLLOUT_${TAG}_GPU$1_DONE | tee -a logs/rollout_${TAG}_gpu$1.log; sleep 86400"
  cmd1="( $(refs cuda:$2) && $(model cuda:$2 "$d_lo") ) 2>&1 | grep --line-buffered -v Warning | tee -a logs/rollout_${TAG}_gpu$2.log; echo ROLLOUT_${TAG}_GPU$2_DONE | tee -a logs/rollout_${TAG}_gpu$2.log; sleep 86400"
  tmux kill-session -t roll_gpu$1 2>/dev/null || true; tmux kill-session -t roll_gpu$2 2>/dev/null || true
  tmux new-session -d -s roll_gpu$1 "$cmd0"
  tmux new-session -d -s roll_gpu$2 "$cmd1"
  echo "long rollout of $TAG launched $(date -u): tmux roll_gpu$1 (depths $d_hi), roll_gpu$2 (refs, depths $d_lo); logs/rollout_${TAG}_gpu*.log"
else
  cmd="( $(refs cuda:$1) && $(model cuda:$1 "$DEPTHS") ) 2>&1 | grep --line-buffered -v Warning | tee -a logs/rollout_${TAG}_gpu$1.log; echo ROLLOUT_${TAG}_GPU$1_DONE | tee -a logs/rollout_${TAG}_gpu$1.log; sleep 86400"
  tmux kill-session -t roll_gpu$1 2>/dev/null || true
  tmux new-session -d -s roll_gpu$1 "$cmd"
  echo "long rollout of $TAG launched $(date -u): tmux roll_gpu$1 (refs + depths $DEPTHS); logs/rollout_${TAG}_gpu$1.log"
fi
