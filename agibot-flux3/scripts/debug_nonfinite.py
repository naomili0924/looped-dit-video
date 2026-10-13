"""Replay the micro-batches of a non-finite update on one GPU to find which window / timestep breaks backward.

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/debug_nonfinite.py \
        /dev/shm/runs/smoke/nonfinite-step11-rank0.json /dev/shm/runs/smoke/nonfinite-step11-rank1.json
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/loop-dit")
from flux_action.config import PolicyConfig  # noqa: E402
from flux_action.models.text_encoder import VEC_DIM  # noqa: E402
from flux_action.policy import FluxActionPolicy  # noqa: E402
from flux_action.processing import packing, video_only  # noqa: E402
from ldvflux.stream import EpisodeFetcher, decode_frames, window_frame_indices  # noqa: E402

SPLITS = "/tmp/claude-0/-workspace-loop-dit/eff958a6-3a23-4ecc-b209-f087c571a774/scratchpad/splits_test.json"
cfg_json = json.load(open("/workspace/loop-dit/configs/smoke_video_ft.json"))
splits = json.load(open(SPLITS))
meta = splits["meta"]
episodes = {e["episode"]: e for v in splits["splits"].values() for e in v}

dev = torch.device("cuda:0")
pcfg = PolicyConfig(**{**cfg_json["policy"], "torch_dtype": "bfloat16"})
policy = FluxActionPolicy(pcfg)
from flux_action.training.distributed import apply_activation_checkpointing  # noqa: E402
apply_activation_checkpointing(policy.dit)
policy.dit.to(dev)
for m in policy.frozen.modules():
    m.to(dev)
policy.freeze_conditioning_heads()
policy.train()
fetcher = EpisodeFetcher()


def batch_from(mb):
    k = mb["context_frames"][0]
    frames = []
    for ep_id, anchor in zip(mb["episode"], mb["anchor"]):
        ep = episodes[str(ep_id)]
        idx = window_frame_indices(anchor, k, meta["stride"], meta["future_frames"])
        fr = decode_frames(fetcher.fetch(ep), set(idx))
        frames.append(torch.from_numpy(np.stack([fr[i] for i in idx])))
    return {"images.head": torch.stack(frames), "task": mb["task"], "context_frames": torch.tensor(mb["context_frames"]),
            "window_seed": torch.tensor(mb["window_seed"])}


def grads_finite():
    bad = [n for n, p in policy.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    return bad


t_values = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.995, 0.999]
for path in sys.argv[1:]:
    dump = json.load(open(path))
    for mi, mb in enumerate(dump["micro_batches"]):
        batch = batch_from(mb)
        prepared = policy.prepare(batch)
        fin = {"cond": None if prepared.cond_latents is None else bool(torch.isfinite(prepared.cond_latents).all()),
               "future": bool(torch.isfinite(prepared.latents).all()),
               "ctx": all(bool(torch.isfinite(c).all()) for c in prepared.ctxs)}
        k = prepared.context_frames
        video = video_only.pack(prepared.cond_latents, prepared.latents, k, pcfg.fps)
        res = []
        for t in t_values:
            policy.dit.zero_grad(set_to_none=True)
            torch.manual_seed(0)
            kw, target = video_only.forward_kwargs(video, torch.full((len(prepared.idx),), t))
            packed, seqlens = packing.pack_windows(kw, prepared.ctxs, VEC_DIM)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = policy.dit(**policy._cast_inputs(packed), seqlens=seqlens)["x_video"]
                loss = video_only.flow_loss(pred, packing.flatten_windows({"x": target})["x"])["loss"]
            loss.backward()
            bad = grads_finite()
            res.append(f"t={t}:{'OK' if not bad else 'NaN(' + str(len(bad)) + ')'}")
        print(f"{path.split('/')[-1]} mb{mi} k={k} captions={[c[:25] for c in mb['task']]} inputs_finite={fin}")
        print("   ", " ".join(res), flush=True)
