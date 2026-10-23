#!/bin/bash
# flux-action checkout (pinned commit + our patch, as EVAL_MOTION.md §3) and its venv; weights already in /dev/shm/w.
set -eux
cd "$(dirname "$(readlink -f "$0")")/../.."
[ -d flux-action ] || git clone -q https://github.com/black-forest-labs/flux-action
cd flux-action && git checkout -q 830f1ce178d7aea52f18fde31d8e8e36ce6bae21
git apply --check ../patches/flux-action.patch && git apply ../patches/flux-action.patch
[ -d .venv ] || uv venv -q -p 3.12 .venv
uv pip install -q -p .venv/bin/python -e '.[encoders]' lpips torchmetrics av scipy
# NATTEN built for torch 2.11 + CUDA 12.8 (the PyPI build does not match), requests for fetch_eps.py
uv pip install -q -p .venv/bin/python 'natten==0.21.6+torch2110cu128' -f https://whl.natten.org/ requests
mkdir -p outputs && ln -sfn /dev/shm/w/base outputs/weights
echo ENV_DONE
