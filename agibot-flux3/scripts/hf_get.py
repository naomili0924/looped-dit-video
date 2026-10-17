#!/usr/bin/env python3
"""Fetch a file given as hf://<user>/<repo>/<path> (or copy a local path) to <dest>. Token from HF_TOKEN.
    scripts/hf_get.py hf://Jinyan0924/flux3-agibot-looped/random_depth_n1-8/ema_step-2500.safetensors /dev/shm/ema/ema_rand2500_step-2500.safetensors
"""
import os, shutil, sys
from pathlib import Path

src, dest = sys.argv[1], Path(sys.argv[2])
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
    print("exists", dest); sys.exit(0)
if src.startswith("hf://"):
    from huggingface_hub import hf_hub_download
    parts = src[5:].split("/", 2)
    repo, fn = parts[0] + "/" + parts[1], parts[2]
    p = hf_hub_download(repo, fn, token=os.environ.get("HF_TOKEN") or None, local_dir=str(dest.parent / "dl"))
    shutil.move(p, dest)
else:
    shutil.copy(src, dest)
print("fetched", dest, dest.stat().st_size)
