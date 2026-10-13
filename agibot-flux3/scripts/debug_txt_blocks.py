"""Magnitudes through the text-stream mode blocks for one micro-batch, with base weights and with a checkpoint.

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/debug_txt_blocks.py <model.pt or ''>
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
cfg = json.load(open("/workspace/loop-dit/configs/anomaly.json"))
sp = json.load(open(SPLITS))
eps = {e["episode"]: e for v in sp["splits"].values() for e in v}
mb = json.load(open("/dev/shm/runs/smoke384/nonfinite-step11-rank0.json"))["micro_batches"][3]
dev = torch.device("cuda:0")
import os
AUTOCAST = os.environ.get("AUTOCAST", "1") == "1"
policy = FluxActionPolicy(PolicyConfig(**{**cfg["policy"], "torch_dtype": "bfloat16"}))
if len(sys.argv) > 1 and sys.argv[1]:
    sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
    sd = {k.removeprefix("dit.").replace("_checkpoint_wrapped_module.", ""): v for k, v in sd.items()}
    res = policy.dit.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    print("loaded checkpoint; missing", len(res.missing_keys), "unexpected", len(res.unexpected_keys))
policy.dit.to(dev)
for m in policy.frozen.modules():
    m.to(dev)
policy.train()

k = mb["context_frames"][0]
fetcher = EpisodeFetcher()
frames = []
for ep_id, anchor in zip(mb["episode"], mb["anchor"]):
    idx = window_frame_indices(anchor, k, 2, 33)
    fr = decode_frames(fetcher.fetch(eps[str(ep_id)]), set(idx), (384, 512))
    frames.append(torch.from_numpy(np.stack([fr[i] for i in idx])))
batch = {"images.head": torch.stack(frames), "task": mb["task"], "context_frames": torch.tensor(mb["context_frames"]),
         "window_seed": torch.tensor(mb["window_seed"])}
prepared = policy.prepare(batch)
print("captions after dropout:", [len(c[0]) for c in prepared.ctxs], "ctx absmax per window:",
      [round(float(c.float().abs().max()), 1) for c in prepared.ctxs])

stats = []


def hook(name):
    def fn(mod, inp, out):
        x = inp[0].float()
        stats.append((name, float(x.abs().max()), float(out.float().abs().max()), bool(torch.isfinite(out).all())))
    return fn


for d, blk in enumerate(policy.dit.txt_mode_blocks):
    blk.register_forward_hook(hook(f"txt_mode_block{d}"))
video = video_only.pack(prepared.cond_latents, prepared.latents, k, policy.config.fps)
for t in (0.3, 0.7, 0.95):
    stats.clear()
    policy.dit.zero_grad(set_to_none=True)
    kw, target = video_only.forward_kwargs(video, torch.full((len(prepared.idx),), t))
    packed, seqlens = packing.pack_windows(kw, prepared.ctxs, VEC_DIM)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AUTOCAST):
        pred = policy.dit(**policy._cast_inputs(packed), seqlens=seqlens)["x_video"]
        loss = video_only.flow_loss(pred, packing.flatten_windows({"x": target})["x"])["loss"]
    loss.backward()
    bad = [n for n, p in policy.dit.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"t={t} loss {float(loss):.4f} non-finite grads: {len(bad)} {bad[:3]}")
    for s in stats:
        print(f"   {s[0]}: |in|max {s[1]:.3e}  |out|max {s[2]:.3e}  finite {s[3]}")
