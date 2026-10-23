#!/bin/bash
# Weights for the capability eval, into RAM disk (root disk is 32 GB): FLUX 3 Action base (DiT, VAE, Qwen3-VL text
# encoder) and the random-depth long-run EMA at step 9500.
set -eux
W=/dev/shm/w
/venv/main/bin/hf download black-forest-labs/flux-3-action-base --local-dir $W/base --include 'flux-3-action-base.safetensors' 'video_vae.safetensors' 'text_encoder/*' 'config.json'
/venv/main/bin/hf download Jinyan0924/flux3-agibot-looped random_depth_n1-8_long/ema_step-9500.safetensors random_depth_n1-8_long/config.json --local-dir $W/looped
echo WEIGHTS_DONE
