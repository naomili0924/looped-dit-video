#!/bin/bash
# At the step-1000 checkpoint: stop training, profile a micro-batch (eager, then compiled) on GPU 0 and smoke-test the
# Phase-4 evaluator on GPU 1, both logged under logs/profile/. Prints DONE when both finish. Relaunch is manual.
set -u
cd /workspace/loop-dit
bash scripts/stop_train.sh
mkdir -p logs/profile results/_smoke
cd flux-action
export PYTHONPATH=/workspace/loop-dit
(
  .venv/bin/python ../scripts/profile_step.py --k 0 1 9 --repeats 2 --device cuda:0 --out ../logs/profile/eager.json \
      > ../logs/profile/eager.log 2>&1
  .venv/bin/python ../scripts/profile_step.py --k 0 1 9 --repeats 2 --device cuda:0 --compile --out ../logs/profile/compiled.json \
      > ../logs/profile/compiled.log 2>&1
  echo "PROFILE DONE $(date -u)"
) &
p1=$!
(
  .venv/bin/python -m ldvflux.eval_phase4 --name smoke --splits val --k 0 9 --n-recon 4 --n-fvd 4 --fvd-k 9 \
      --n-judge 4 --judge-k 9 --batch 4 --steps 10 --device cuda:1 --out ../results/_smoke > ../logs/profile/eval_smoke.log 2>&1
  echo "EVAL SMOKE DONE $(date -u) exit=$?"
) &
p2=$!
wait $p1 $p2
echo "DONE $(date -u)"
