"""Train the looped text-to-video DiT.

    python -m ldv.train --config configs/t2v_7b_webvid50k.yml --output-dir /dev/shm/ldv/outputs/run1
    torchrun --nproc_per_node=8 -m ldv.train ...      # multi-GPU (DDP)

A run resumes automatically from the newest checkpoint in its output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import TrainConfig
from .data import list_shards, make_loader
from .diffusion import deep_supervision_weights, euler_sample, training_loss
from .encoders import TextEncoder
from .model import LoopedFluxT2V
from .utils import CPUEma, atomic_save, init_distributed, is_main, learning_rate, rank, seed_everything, world_size


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the looped text-to-video DiT.")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-from", help="checkpoint to start from")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values (model.x=...)")
    p.add_argument("--max-steps", type=int, help="stop after this many steps in this invocation (smoke tests)")
    return p.parse_args()


def make_optimizer(cfg: TrainConfig, model: torch.nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    kw = dict(lr=cfg.learning_rate, betas=(0.9, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, **kw)
    return torch.optim.AdamW(params, fused=True, **kw)


def checkpoints(directory: Path) -> list[Path]:
    return sorted(directory.glob("checkpoint_*.pt"))


def save_checkpoint(directory: Path, step: int, model, ema: CPUEma, optimizer, cfg: TrainConfig) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"checkpoint_{step:07d}.pt"
    for old in checkpoints(directory)[: max(0, len(checkpoints(directory)) - cfg.keep_last + 1)]:
        old.unlink()  # free tmpfs space before writing the new one
    atomic_save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                 "optimizer": optimizer.state_dict(), "config": cfg.to_dict()}, path)
    atomic_save({"step": step, "ema": ema.state_dict(), "config": cfg.to_dict()}, directory.parent / "ema_latest.pt")
    return path


def load_checkpoint(path: Path, model, ema: CPUEma, optimizer, cfg: TrainConfig) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"] != cfg.model.to_dict():
        diff = {k: (ckpt["config"]["model"].get(k), v) for k, v in cfg.model.to_dict().items()
                if ckpt["config"]["model"].get(k) != v and k != "grad_checkpointing"}
        if diff:
            raise ValueError(f"{path} has a different architecture: {diff}")
    model.load_state_dict({k: v.float() for k, v in ckpt.get("model", ckpt["ema"]).items()})
    ema.load_state_dict(ckpt["ema"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if is_main():
        print(f"loaded {path} (step {ckpt['step']})", flush=True)
    return int(ckpt["step"])


def push_to_hub(cfg: TrainConfig, out_dir: Path, step: int) -> None:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(cfg.hub_repo, exist_ok=True, private=True)
        api.upload_file(path_or_fileobj=str(out_dir / "ema_latest.pt"), path_in_repo=f"ema_{step:07d}.pt",
                        repo_id=cfg.hub_repo)
        api.upload_file(path_or_fileobj=str(out_dir / "config.yaml"), path_in_repo="config.yaml", repo_id=cfg.hub_repo)
        for name in ("train_log.jsonl", "probe.jsonl"):
            if (out_dir / name).exists():
                api.upload_file(path_or_fileobj=str(out_dir / name), path_in_repo=name, repo_id=cfg.hub_repo)
        print(f"[hub] pushed step {step} to {cfg.hub_repo}", flush=True)
    except Exception as e:  # never kill training for an upload
        print(f"[hub] upload failed: {e}", flush=True)


@torch.no_grad()
def ema_model(model: LoopedFluxT2V, ema: CPUEma, device) -> LoopedFluxT2V:
    with torch.device(device):
        m = LoopedFluxT2V(model.cfg).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    m.load_state_dict(ema.state_dict(torch.bfloat16))
    return m


@torch.no_grad()
def write_samples(cfg: TrainConfig, model, ema, text_encoder, latent_shape, out_dir: Path, step: int, device) -> None:
    from .encoders import WanVAE
    from .video import save_video_grid

    m = ema_model(model, ema, device)
    text, mask = text_encoder(cfg.sample_prompts)
    g = torch.Generator(device=device).manual_seed(0)
    vae = WanVAE(device, repo=cfg.vae)
    for loops in sorted({1, cfg.model.num_loops}):
        lat = euler_sample(m, text.to(torch.bfloat16), mask, latent_shape, steps=cfg.sample_steps,
                           cfg_scale=cfg.sample_cfg, noise_scale=cfg.noise_scale, num_loops=loops, generator=g)
        videos = vae.decode(lat)
        save_video_grid(videos, out_dir / "samples" / f"{step:07d}_loops{loops}.mp4")
    del m, vae
    torch.cuda.empty_cache()


@torch.no_grad()
def probe(cfg: TrainConfig, model, ema, text_encoder, probe_batch, out_dir: Path, step: int, device) -> None:
    from .probe import run_probe

    m = ema_model(model, ema, device)
    latents = probe_batch["latents"].to(device, torch.float32)
    text, mask = text_encoder(probe_batch["caption"])
    rows = run_probe(m, latents, text.to(torch.bfloat16), mask, num_loops=cfg.probe_max_loops,
                     noise_scale=cfg.noise_scale)
    with open(out_dir / "probe.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps({"step": step, **row}) + "\n")
    summary = {f"t{r['t']}_L{r['loop']}": round(r["r2_joint"], 3) for r in rows if r["t"] == 0.5}
    print(json.dumps({"step": step, "probe_r2_joint@t0.5": summary}), flush=True)
    del m
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
    cfg = TrainConfig.from_yaml(args.config, overrides)
    device = init_distributed()
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    per_step = cfg.micro_batch_size * world_size()
    if cfg.batch_size % per_step:
        raise ValueError(f"batch_size {cfg.batch_size} not a multiple of micro_batch x GPUs = {per_step}")
    accum = cfg.batch_size // per_step
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(cfg.seed)
    with torch.device(device):  # build on the GPU: CPU init of ~5B params is slow
        model = LoopedFluxT2V(cfg.model)  # fp32 weights; bf16 autocast compute
    ema = CPUEma(model, cfg.ema_decay, cfg.ema_every)
    optimizer = make_optimizer(cfg, model)
    step = 0
    existing = checkpoints(ckpt_dir)
    resume = existing[-1] if existing else Path(args.init_from) if args.init_from else None
    if resume is not None:
        step = load_checkpoint(resume, model, ema, optimizer, cfg)
    seed_everything(cfg.seed + rank() + step)
    net = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True) if world_size() > 1 else model
    fwd = torch.compile(net) if cfg.compile else net

    text_encoder = TextEncoder(cfg.text_encoder, cfg.prompt_length, device)
    loader = make_loader(cfg.train_shards, cfg.micro_batch_size, cfg.num_workers, cfg.shuffle_buffer, cfg.seed + step)
    batches = iter(loader)
    probe_batch = None
    if cfg.probe_every and is_main():
        src = cfg.probe_shards or list_shards(cfg.train_shards)[:1]  # a full shard, no repeated clips
        probe_batch = next(iter(make_loader(src, cfg.probe_clips, 0, 0, seed=1234)))
    exit_weights = deep_supervision_weights(cfg.model.num_loops, cfg.deep_supervision_weighting) if cfg.deep_supervision else None
    if is_main():
        print(f"{model.num_params() / 1e9:.3f}B parameters; {world_size()} GPUs x micro-batch {cfg.micro_batch_size} "
              f"x {accum} accum = batch {cfg.batch_size}; exit weights {exit_weights}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    net.train()
    metric_sums: dict[str, torch.Tensor] = {}
    n_micro, last_time, last_step, latent_shape = 0, time.time(), step, None
    stop_at = cfg.num_steps if args.max_steps is None else min(cfg.num_steps, step + args.max_steps)
    while step < stop_at:
        for micro in range(accum):
            batch = next(batches)
            latents = batch["latents"].to(device, non_blocking=True).float()
            latent_shape = tuple(latents.shape[1:])
            text, text_mask = text_encoder(batch["caption"])
            no_sync = world_size() > 1 and micro < accum - 1
            with net.no_sync() if no_sync else contextlib.nullcontext():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = training_loss(
                        fwd, latents, text.to(torch.bfloat16), text_mask, exit_weights,
                        noise_scale=cfg.noise_scale, t_logit_mean=cfg.t_logit_mean,
                        t_logit_std=cfg.t_logit_std, label_drop_rate=cfg.label_drop_rate,
                    )
                (loss / accum).backward()
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0) + v
            n_micro += 1

        lr = learning_rate(step, cfg.learning_rate, cfg.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None],
            cfg.max_grad_norm if cfg.max_grad_norm > 0 else math.inf)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        ema.update(model, step)

        if step % cfg.log_every == 0 and is_main():
            now = time.time()
            log = {"step": step, **{k: round(v.item() / n_micro, 5) for k, v in metric_sums.items()},
                   "lr": lr, "grad_norm": round(grad_norm.item(), 4),
                   "sec_per_step": round((now - last_time) / (step - last_step), 2),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            print(json.dumps(log), flush=True)
            with open(out_dir / "train_log.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")
            metric_sums, n_micro, last_time, last_step = {}, 0, now, step
        saved = is_main() and (step % cfg.ckpt_every == 0 or step == cfg.num_steps)
        if saved:
            save_checkpoint(ckpt_dir, step, model, ema, optimizer, cfg)
        if is_main() and cfg.sample_every and step % cfg.sample_every == 0:
            write_samples(cfg, model, ema, text_encoder, latent_shape, out_dir, step, device)
        if is_main() and probe_batch is not None and step % cfg.probe_every == 0:
            probe(cfg, model, ema, text_encoder, probe_batch, out_dir, step, device)
        # Push last, so the probe results of this step go up with its checkpoint.
        if saved and cfg.hub_repo and (step % cfg.hub_every == 0 or step == cfg.num_steps):
            push_to_hub(cfg, out_dir, step)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
