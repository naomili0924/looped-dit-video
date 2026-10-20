#!/bin/bash
# 20-step smoke test of configs/looped_rand_384_r1.json on both GPUs (separate output dir, no eval, no checkpoint)
cd /workspace/loop-dit-eval/looped-dit-video/agibot-flux3/flux-action
A=/workspace/loop-dit-eval/looped-dit-video/agibot-flux3
set -a; . /workspace/.env; set +a
rm -rf /dev/shm/runs/looprand_stress; mkdir -p /dev/shm/runs/looprand_stress
PYTHONPATH=$A FLUX_ACTION_PG_TIMEOUT_MINUTES=10 OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN \
  .venv/bin/python .venv/bin/torchrun --nproc_per_node 2 --master_port 29700 -m flux_action.cli train \
  --config /dev/shm/runs/stress_k9_n8.json --override output_dir=/dev/shm/runs/looprand_stress --override steps=6 \
  --override cooldown_start=null --override cooldown_steps=0 --override eval_every=100000 --override checkpoint_every=100000 \
  --override log_every=1 2>&1 | grep -v Warning | tee $A/logs/stress_k9_n8.log
echo "STRESS_EXITED ${PIPESTATUS[0]}" | tee -a $A/logs/stress_k9_n8.log
