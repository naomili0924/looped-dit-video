"""Periodic evaluation inside flux-action's trainer (``eval_fn = "ldvflux.evaluate:eval_hook"``).

Every ``eval_every`` updates, on all ranks (the DiT is FSDP-sharded, so every rank runs the same forwards):

* **val loss** on a fixed set of val windows, for every context length k in K_SET and k = 0, with fixed timesteps
  (a stratified grid) and fixed noise, so evals are comparable across steps; raw weights.
* **8 fixed val samples** with the EMA weights swapped in (fixed windows, k = K_MAX, fixed seeds), saved as mp4
  grids: top row ground truth (context + future), bottom row context + generated future.
* **train reference** (``train_ref_windows`` > 0): the same loss on fixed *train* windows, so the train/val gap
  (over- vs underfitting) is measurable; the per-update training loss is too noisy for that.
* **early stop**: returns ``stop`` when the mean val loss rose at 3 consecutive evals (``stop_rule = "rising"``), or
  when ``stop_metric`` has not reached a new best for ``patience`` evals (``stop_rule = "no_best"``).

Settings come from ``trainer.config.dataset_kwargs["eval"]`` (see DEFAULTS).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from flux_action.models.text_encoder import VEC_DIM
from flux_action.processing import packing, video_only

from . import sampling
from .stream import DEFAULTS as STREAM_DEFAULTS
from .stream import EpisodeFetcher, decode_frames, fixed_windows, load_splits, trim, window_frame_indices

DEFAULTS = {
    "loss_windows": 32,  # total, split over the ranks
    "loss_batch": 4,
    "t_grid": 8,  # timesteps per window: the midpoints of 8 equal-mass bins of the training distribution
    "sample_windows": 8,
    "sample_steps": 30,
    "sample_cfg": 4.0,
    "sample_shift": 5.0,
    "patience": 3,
    "stop_rule": "rising",  # rising | no_best
    "stop_metric": "val_loss",  # any key of the result, e.g. val_loss_k9_mean_loops (looped: mean over eval depths)
    "train_ref_windows": 0,  # fixed train windows for the train-reference loss (0 = off)
}


def _rank_world() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _load_windows(picks: list[dict], meta: dict, hw=None) -> list[dict]:
    """Decode the frames of fixed (episode, window) picks at K_MAX context, resized to ``hw`` like training."""
    fetcher, out = EpisodeFetcher(), []
    stride, future, kmax = meta["stride"], meta["future_frames"], max(meta["k_set"])
    for p in picks:
        ep, w = p["episode"], p["window"]
        idx = window_frame_indices(w["anchor"], kmax, stride, future)
        frames = decode_frames(fetcher.fetch(ep), set(idx), hw)
        if all(i in frames for i in idx):
            out.append({"frames": np.stack([frames[i] for i in idx]), "kmax": kmax, "caption": w["caption"],
                        "episode": ep["episode"], "anchor": w["anchor"]})
    return out


class EvalState:
    def __init__(self, trainer):
        cfg = {**STREAM_DEFAULTS, **trainer.config.dataset_kwargs}
        self.opts = {**DEFAULTS, **cfg.get("eval", {})}
        self.meta = load_splits(cfg["splits"])["meta"]
        rank, world = _rank_world()
        split = cfg.get("eval_split", "val")
        n_loss = self.opts["loss_windows"]
        picks = fixed_windows(split, n_loss + self.opts["sample_windows"], seed=0, splits_path=cfg["splits"])
        loss_picks, sample_picks = picks[:n_loss], picks[n_loss:]
        hw = cfg.get("frame_hw")
        self.loss_windows = _load_windows(loss_picks[rank::world], self.meta, hw)
        per_rank = math.ceil(len(sample_picks) / world)
        self.sample_windows = _load_windows(sample_picks[rank * per_rank : (rank + 1) * per_rank], self.meta, hw)
        n_ref = self.opts["train_ref_windows"]
        ref = fixed_windows("train", n_ref, seed=0, splits_path=cfg["splits"]) if n_ref else []
        self.train_windows = _load_windows(ref[rank::world], self.meta, hw)
        self.history: list[float] = []
        self.out_dir = Path(trainer.output_dir) / "eval"
        self.out_dir.mkdir(parents=True, exist_ok=True)


def _t_grid(n: int, mean: float, std: float) -> torch.Tensor:
    """Timesteps at the quantiles (i + 0.5) / n of t = sigmoid(N(mean, std))."""
    q = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    z = torch.distributions.Normal(0.0, 1.0).icdf(q)
    return torch.sigmoid(z * std + mean).float()


@torch.no_grad()
def val_loss(policy, windows: list[dict], k: int, opts: dict, t_values: torch.Tensor,
             num_loops: int | None = None) -> tuple[float, int]:
    """Sum of the velocity MSE over windows x timesteps (fixed noise), and the count. ``num_loops`` overrides
    the loop depth of a looped model (None = as trained)."""
    extra = {"num_loops": num_loops} if num_loops is not None else {}
    total, count = 0.0, 0
    future = policy.config.future_frames
    for s in range(0, len(windows), opts["loss_batch"]):
        chunk = windows[s : s + opts["loss_batch"]]
        frames = torch.stack([trim(w, k, future) for w in chunk])
        cond, fut = sampling.encode_windows(policy, frames, k)
        ctxs = [c for c, _ in policy._contexts([w["caption"] for w in chunk], policy.device)]
        video = video_only.pack(cond, fut, k, policy.config.fps)
        for j, t in enumerate(t_values.tolist()):
            with torch.random.fork_rng(devices=[policy.device]):
                torch.manual_seed(1000 * s + j)
                kw, target = video_only.forward_kwargs(video, torch.full((len(chunk),), t))
            packed, seqlens = packing.pack_windows(kw, ctxs, VEC_DIM)
            pred = policy.dit(**policy._cast_inputs(packed), seqlens=seqlens, **extra)["x_video"]
            total += float(video_only.flow_loss(pred, packing.flatten_windows({"x": target})["x"])["loss"]) * len(chunk)
            count += len(chunk)
    return total, count


def _all_sum(values: list[float], device) -> list[float]:
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t)
    return t.tolist()


def _save_grid(gt: torch.Tensor, gen: torch.Tensor, k: int, path: Path, fps: float) -> None:
    """gt uint8 (B, k + F, 3, H, W); gen float (B, 3, F, H, W) in [-1, 1] -> mp4: rows = windows, cols = (gt | gen)."""
    import imageio.v3 as iio

    gen_u8 = ((gen.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 1, 3, 4).cpu()  # (B, F, 3, H, W)
    rows = []
    for g_t, g_g in zip(gt, gen_u8):
        pred = torch.cat([g_t[:k], g_g[: g_t.shape[0] - k]])  # context + generated future
        rows.append(torch.cat([g_t, pred], dim=-1))  # side by side
    grid = torch.cat(rows, dim=-2).permute(0, 2, 3, 1).numpy()  # (T, H*B, 2W, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, grid, fps=fps, codec="libx264", macro_block_size=1)


_STATE: EvalState | None = None


def eval_hook(trainer) -> dict:
    global _STATE
    if _STATE is None:
        _STATE = EvalState(trainer)
    st, policy, opts = _STATE, trainer.policy, _STATE.opts
    rank, _ = _rank_world()
    cfg = policy.config
    t_values = _t_grid(opts["t_grid"], cfg.video_logit_mean, cfg.video_logit_std)
    train_ks = list(trainer.config.dataset_kwargs.get("k_set") or st.meta["k_set"])
    ks = [0, *st.meta["k_set"]]  # every k the windows support, so the per-k curves stay comparable
    sums = []
    for k in ks:
        sums += list(val_loss(policy, st.loss_windows, k, opts, t_values))
    sums = _all_sum(sums, policy.device)
    per_k = {f"val_loss_k{k}": sums[2 * i] / max(sums[2 * i + 1], 1) for i, k in enumerate(ks)}
    mean = float(np.mean([per_k[f"val_loss_k{k}"] for k in [0, *train_ks]]))  # early-stop metric: trained k only
    loop = getattr(policy.dit, "loop", None)
    if loop is not None:  # looped model: the largest trained k at other inference depths
        k_top = max(train_ks)
        depths = [d for d in opts.get("eval_loops", [1, 2, 4, 8]) if d != loop.num_loops]
        sums_d = []
        for d in depths:
            sums_d += list(val_loss(policy, st.loss_windows, k_top, opts, t_values, num_loops=d))
        sums_d = _all_sum(sums_d, policy.device)
        per_k.update({f"val_loss_k{k_top}_loops{d}": sums_d[2 * i] / max(sums_d[2 * i + 1], 1)
                      for i, d in enumerate(depths)})
        per_k[f"val_loss_k{k_top}_mean_loops"] = float(np.mean(
            [per_k[f"val_loss_k{k_top}"]] + [per_k[f"val_loss_k{k_top}_loops{d}"] for d in depths]))
    if opts["train_ref_windows"]:  # same protocol on fixed train windows: trained k, inference depth as trained
        sums_r = []
        for k in [0, *train_ks]:
            sums_r += list(val_loss(policy, st.train_windows, k, opts, t_values))
        sums_r = _all_sum(sums_r, policy.device)
        ref = {f"train_ref_loss_k{k}": sums_r[2 * i] / max(sums_r[2 * i + 1], 1) for i, k in enumerate([0, *train_ks])}
        ref["train_ref_loss"] = float(np.mean(list(ref.values())))
        ref["gap"] = mean - ref["train_ref_loss"]  # val - train on the same k/depth mix as the val_loss mean
        per_k.update(ref)

    # Samples with the EMA weights (CPU EMA swapped into the shards on every rank).
    ema = next(iter(trainer.emas.values())) if trainer.emas else None
    stash = ema.swap_into(policy) if ema is not None and hasattr(ema, "swap_into") and ema.last_step >= 0 else None
    try:
        k = max(train_ks)
        win = st.sample_windows
        frames = torch.stack([trim(w, k, cfg.future_frames) for w in win])
        cond, _ = sampling.encode_windows(policy, frames, k)
        lat = sampling.generate(policy, cond, k, [w["caption"] for w in win],
                                n_future_latents=1 + (cfg.future_frames - 1) // 4, steps=opts["sample_steps"],
                                cfg_scale=opts["sample_cfg"], shift=opts["sample_shift"], seed=1234 + rank)  # fixed across evals
        video = sampling.decode(policy, lat)
        _save_grid(frames, video, k, st.out_dir / f"step-{trainer.step:06d}_rank{rank}.mp4", cfg.fps)
    finally:
        if stash is not None:
            ema.restore(policy, stash)

    result = {"val_loss": mean, **per_k, "ema_samples": stash is not None}
    st.history.append(result[opts["stop_metric"]])
    p, h = opts["patience"], st.history
    if opts["stop_rule"] == "no_best":  # no new best in the last p evals (history starts at this launch)
        stop = len(h) > p and min(h[-p:]) >= min(h[:-p])
    else:
        stop = len(h) > p and all(h[-i] > h[-i - 1] for i in range(1, p + 1))
    result["stop"] = bool(stop)
    if rank == 0:
        with open(st.out_dir / "val.jsonl", "a") as f:
            f.write(json.dumps({"step": trainer.step, **result}) + "\n")
    return result
