"""Phase 0 smoke test: the FLUX 3 Action base trunk, video + text streams only, predicts future frames
from one observed frame and a caption; the latents are decoded to mp4 with the FLUX video VAE.

    cd /workspace/loop-dit/flux-action && .venv/bin/python ../scripts/smoke_base.py --out ../logs/smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from flux_action.inference.sampling import timeshift
from flux_action.models.positional import batched_prc_txt, batched_prc_vid
from flux_action.models.text_encoder import load_text_encoder, text_context
from flux_action.models.transformer import JointSingleSeq, JointSingleSeqParams
from flux_action.models.video_vae import load_video_vae
from flux_action.models.wiring import default_dtype, stream_of_key
from flux_action.processing.packing import video_time_ids

W = "outputs/weights"
VIDEO_STREAMS = {"video": 96, "video_cond": 96}


def build_video_dit(device) -> tuple[JointSingleSeq, dict]:
    params = JointSingleSeqParams(in_channels=dict(VIDEO_STREAMS),
                                  sequence={"x_video": "video", "x_video_cond": "video_cond"}, attn_mode="torch")
    with torch.device(device), default_dtype(torch.bfloat16):
        model = JointSingleSeq(params)
    keep = {None, "video", "video_cond", "txt"}
    with safe_open(f"{W}/flux-3-action-base.safetensors", framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in f.keys() if stream_of_key(k) in keep}
    res = model.load_state_dict(sd, strict=False)
    info = {"loaded": len(sd) - len(res.unexpected_keys), "missing": res.missing_keys,
            "unexpected": res.unexpected_keys}
    return model.eval().requires_grad_(False), info


def first_frame(path: str, hw: tuple[int, int]) -> torch.Tensor:
    """(3, H, W) in [-1, 1]: the first frame of a gif/mp4, center-cropped to the canvas aspect and resized."""
    frame = np.asarray(iio.imread(path, index=0))[..., :3]
    x = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    h, w = x.shape[1:]
    target = hw[0] / hw[1]
    if h / w > target:
        ch = int(round(w * target))
        x = x[:, (h - ch) // 2 : (h - ch) // 2 + ch]
    else:
        cw = int(round(h / target))
        x = x[:, :, (w - cw) // 2 : (w - cw) // 2 + cw]
    x = F.interpolate(x[None], size=hw, mode="bilinear", antialias=True, align_corners=False)[0]
    return x * 2 - 1


def save_mp4(video: torch.Tensor, path: Path, fps: float) -> None:
    """(3, T, H, W) in [-1, 1] -> mp4."""
    frames = ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="docs/assets/so101/pick-from-cup.gif")
    p.add_argument("--caption", default="The robot arm picks up the object from the cup.")
    p.add_argument("--hw", type=int, nargs=2, default=[480, 640])
    p.add_argument("--future-latents", type=int, default=8, help="8 latents = 32 frames at 4x")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../logs/smoke")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    report: dict = {"args": vars(args)}

    t0 = time.time()
    model, info = build_video_dit(dev)
    report["dit"] = {"params_B": sum(p.numel() for p in model.parameters()) / 1e9, "loaded": info["loaded"],
                     "missing": info["missing"], "unexpected": info["unexpected"][:10]}
    vae = load_video_vae(f"{W}/video_vae.safetensors", dev)
    txt = load_text_encoder(f"{W}/text_encoder", dev)
    report["load_s"] = round(time.time() - t0, 1)

    frame = first_frame(args.image, tuple(args.hw)).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cond_lat = vae.encode_task(frame[None, :, None].to(torch.bfloat16)).clone()  # (1, 96, 1, h, w)
        ctx_c = text_context(txt, args.caption, dev)
        ctx_u = text_context(txt, "", dev)
    _, c, _, lh, lw = cond_lat.shape
    report["latent_hw"] = [lh, lw]

    cond_tok, cond_ids = batched_prc_vid(cond_lat.to(torch.bfloat16), video_time_ids(1, 0, 1, fps=args.fps))
    n = args.future_latents
    g = torch.Generator(device=dev).manual_seed(args.seed)
    noise = torch.randn(1, c, n, lh, lw, generator=g, device=dev)
    x, x_ids = batched_prc_vid(noise.to(torch.bfloat16), video_time_ids(n, 1, 1, fps=args.fps))
    fixed = {"x_video_ids": x_ids.to(dev), "x_video_cond": cond_tok.to(dev), "x_video_cond_ids": cond_ids.to(dev),
             "x_video_cond_timesteps": torch.zeros(cond_tok.shape[:2], device=dev),
             "vector": torch.zeros(1, 768, device=dev, dtype=torch.bfloat16)}

    def velocity(xv: torch.Tensor, t: float) -> torch.Tensor:
        ts = torch.full(xv.shape[:2], t, device=dev)
        preds = []
        for ctx in (ctx_u, ctx_c):
            _, ids = batched_prc_txt(ctx)
            o = model(ctx=ctx, ctx_ids=ids.to(dev), timesteps_ctx=torch.zeros(ctx.shape[:2], device=dev),
                      x_video=xv, x_video_timesteps=ts, **fixed)
            preds.append(o["x_video"].float())
        return preds[0] + args.cfg * (preds[1] - preds[0])

    t1 = time.time()
    ts = timeshift(args.shift, torch.linspace(1.0, 0.0, args.steps + 1)).tolist()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t_cur, t_prev in zip(ts[:-1], ts[1:]):
            x = (x.float() + (t_prev - t_cur) * velocity(x, t_cur)).to(torch.bfloat16)
    torch.cuda.synchronize()
    report["sample_s"] = round(time.time() - t1, 1)

    future = x[0].float().reshape(n, lh, lw, c).permute(3, 0, 1, 2)[None]  # (1, 96, n, h, w)
    report["future_latent_std"] = round(future.std().item(), 4)
    report["cond_latent_std"] = round(cond_lat.float().std().item(), 4)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vid_future = vae.decode(future.to(torch.bfloat16))[0]  # future clip alone
        vid_joint = vae.decode(torch.cat([cond_lat.float(), future], 2).to(torch.bfloat16))[0]
    report["decoded_frames"] = {"future": vid_future.shape[1], "cond+future": vid_joint.shape[1]}
    # Motion check: mean absolute frame-to-frame change (0 would be a static video).
    report["mean_abs_frame_delta"] = round((vid_joint[:, 1:] - vid_joint[:, :-1]).abs().mean().item(), 4)
    report["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    save_mp4(vid_joint, out / "cond_plus_future.mp4", args.fps)
    save_mp4(vid_future, out / "future_only.mp4", args.fps)
    iio.imwrite(out / "cond_frame.png", ((frame.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy())
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
