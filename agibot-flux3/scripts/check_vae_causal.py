"""Is the FLUX video VAE encoder causal in time? Compare latents of prefixes of one real clip.

If latent j depends only on frames <= 4j, then encode(clip[:k])[:, :, :n(k)] == encode(clip)[:, :, :n(k)].
"""
import json, sys
import torch
sys.path.insert(0, "/workspace/loop-dit")
from ldvflux.stream import EpisodeFetcher, decode_frames
from flux_action.models.video_vae import load_video_vae
from flux_action.processing import packing

splits = json.load(open(sys.argv[1]))["splits"]["train"]
ep = splits[0]
w = ep["windows"][0]
idx = [w["anchor"] - 34 + 2 * j for j in range(50)]  # 50 frames at 15 fps
frames = decode_frames(EpisodeFetcher().fetch(ep), set(idx))
clip = torch.stack([torch.from_numpy(frames[i]) for i in idx])  # (50, 3, 480, 640) uint8
dev = torch.device("cuda:0")
vae = load_video_vae("outputs/weights/video_vae.safetensors", dev)
video = packing.materialize_video(clip[None], None, dev, layout="single", canvas_hw=(480, 640))  # (3, T, H, W)
rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    full = vae.encode(video[None, :, :49].to(torch.bfloat16)).float()  # 49 frames -> 13 latents (chunked path)
    print("full 49 frames ->", tuple(full.shape))
    for k in (1, 5, 9, 17, 33):
        n = 1 + (k - 1) // 4
        pre = vae.encode(video[None, :, :k].to(torch.bfloat16)).float()
        print(f"chunked encode(k={k:2d}) -> {pre.shape[2]} latents; rel diff vs prefix of 49-frame encode: {rel(pre[:, :, :n], full[:, :, :n]):.4f}")
        if k in (1, 17, 33):
            task = vae.encode_task(video[None, :, :k].to(torch.bfloat16)).float()
            print(f"   encode_task(k={k}) vs chunked: {rel(task, pre):.4f}")
    # The future clip alone (frames 17..49) vs the same frames inside the joint encode: not expected to match
    fut = vae.encode(video[None, :, 17:50].to(torch.bfloat16)).float()
    print("future-alone latents", fut.shape[2], "| std ctx/full/future:", round(full.std().item(), 3), round(fut.std().item(), 3))
