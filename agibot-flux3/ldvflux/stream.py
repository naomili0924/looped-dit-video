"""Streaming AgiBot World Beta windows for flux-action's trainer (``TrainConfig.dataset = "ldvflux.stream:build"``).

Nothing is stored on disk. Each loader worker owns a seeded share of the split's episodes; per episode it reads the
head-camera mp4 out of its (uncompressed) tar with one HTTP Range request (offsets from scripts/build_index.py),
decodes every frame its windows need, and pushes the windows into a shuffle buffer. Batches are cut from the
buffer with ONE context length k per batch (no padding): k cycles over K_SET by batch index, and with probability
``k0_prob`` a batch is text-only (k = 0).

A window yields ``images.head`` uint8 (k + F, 3, H, W) — k context frames then F future frames at 15 fps —
plus ``task`` (the segment caption), ``context_frames`` (k) and ``window_seed``.
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tar_index import RangeReader  # noqa: E402

DEFAULTS = {
    "splits": str(ROOT / "data" / "splits.json"),
    "split": "train",
    "k0_prob": 0.1,
    "shuffle_windows": 96,  # per worker; a 50-frame 480x640 window is ~46 MB of uint8
    "frame_hw": None,  # resize decoded frames (H, W); None keeps the native 480x640
    "k_set": None,  # context lengths to train on; None = splits.json meta (every k the windows were built for)
}


def load_splits(path: str) -> dict:
    return json.loads(Path(path).read_text())


def decode_frames(data: bytes, wanted: set[int], hw: tuple[int, int] | None = None) -> dict[int, np.ndarray]:
    """Decode the 30 fps frames with indices in ``wanted`` -> {index: uint8 (3, H, W)}; stops after the last one."""
    import av

    out, last = {}, max(wanted)
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                img = frame.to_ndarray(format="rgb24")
                if hw is not None and img.shape[:2] != tuple(hw):
                    t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
                    t = torch.nn.functional.interpolate(t, size=tuple(hw), mode="bilinear", antialias=True)
                    img = t[0].round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
                out[i] = np.ascontiguousarray(img.transpose(2, 0, 1))
            if i >= last:
                break
    return out


def window_frame_indices(anchor: int, k: int, stride: int, future: int) -> list[int]:
    """30 fps indices of k context frames (oldest first) then ``future`` frames, all ``stride`` apart."""
    return [anchor - (k - j) * stride for j in range(k)] + [anchor + j * stride for j in range(future)]


class EpisodeFetcher:
    """Head-camera bytes of an episode by Range request; one RangeReader (signed URL + session) per tar."""

    def __init__(self):
        self.readers: dict[str, RangeReader] = {}

    def fetch(self, ep: dict) -> bytes:
        reader = self.readers.get(ep["tar"])
        if reader is None:
            if len(self.readers) > 64:
                self.readers.clear()
            reader = self.readers[ep["tar"]] = RangeReader(ep["tar"])
        return reader.read(ep["offset"], ep["size"])


def episode_windows(ep: dict, meta: dict, fetcher: EpisodeFetcher, hw=None) -> list[dict]:
    """All windows of one episode, decoded: each holds frames for its largest valid k."""
    stride, future = meta["stride"], meta["future_frames"]
    wins = []
    for w in ep["windows"]:
        kmax = max(w["valid_k"])
        wins.append((w, window_frame_indices(w["anchor"], kmax, stride, future)))
    wanted = {i for _, idx in wins for i in idx}
    frames = decode_frames(fetcher.fetch(ep), wanted, hw)
    out = []
    for w, idx in wins:
        if not all(i in frames for i in idx):  # labels past the end of the video
            continue
        out.append({"frames": np.stack([frames[i] for i in idx]), "kmax": max(w["valid_k"]),
                    "valid_k": w["valid_k"], "caption": w["caption"], "episode": ep["episode"],
                    "task": ep["task"], "anchor": w["anchor"]})
    return out


def batch_k(seed: int, epoch: int, batch_index: int, k_set: list[int], k0_prob: float) -> int:
    """Context length of global batch ``batch_index``: cycles over k_set; text-only with probability k0_prob."""
    rng = random.Random(f"k-{seed}-{epoch}-{batch_index}")
    if rng.random() < k0_prob:
        return 0
    return k_set[batch_index % len(k_set)]


def dry_windows(ep: dict) -> list[dict]:
    """``episode_windows`` without fetching or decoding: the same entries with ``frames=None`` (resume skipping)."""
    return [{"frames": None, "kmax": max(w["valid_k"]), "valid_k": w["valid_k"], "caption": w["caption"],
             "episode": ep["episode"], "task": ep["task"], "anchor": w["anchor"]} for w in ep["windows"]]


def trim(win: dict, k: int, future: int) -> torch.Tensor:
    """Frames of a window for context length k: the newest k context frames then the future frames."""
    frames = win["frames"]
    ctx = frames[win["kmax"] - k : win["kmax"]] if k else frames[:0]
    return torch.from_numpy(np.concatenate([ctx, frames[win["kmax"] : win["kmax"] + future]]))


class StreamingWindows(IterableDataset):
    """Infinite-per-epoch stream sized to ``batches_per_rank``; see the module docstring."""

    def __init__(self, cfg: dict, *, seed: int, epoch: int, rank: int, world_size: int, num_workers: int,
                 windows_per_rank: int, skip_batches: int = 0, grad_accumulation: int = 1, **_):
        self.cfg = {**DEFAULTS, **cfg}
        data = load_splits(self.cfg["splits"])
        self.meta = data["meta"]
        self.episodes = data["splits"][self.cfg["split"]]
        self.seed, self.epoch, self.rank, self.world = seed, epoch, rank, world_size
        self.num_workers, self.wpr, self.skip = max(1, num_workers), windows_per_rank, skip_batches
        self.windows_per_rank = windows_per_rank
        n_windows = sum(len(e["windows"]) for e in self.episodes)
        per_worker = n_windows // (world_size * self.num_workers * windows_per_rank)
        # whole optimizer updates per epoch, equal for every worker
        per_worker -= per_worker % max(1, grad_accumulation) if per_worker >= grad_accumulation else 0
        self.batches_per_worker = max(per_worker, grad_accumulation)
        self.batches_per_rank = self.batches_per_worker * self.num_workers

    def _worker(self) -> int:
        info = get_worker_info()
        return info.id if info else 0

    def __iter__(self):
        wid = self._worker()
        gw = self.rank * self.num_workers + wid  # global worker index
        n_gw = self.world * self.num_workers
        rng = random.Random(f"{self.seed}-{self.epoch}-{gw}")
        order = list(range(len(self.episodes)))
        random.Random(f"{self.seed}-{self.epoch}").shuffle(order)
        mine = order[gw::n_gw] or order
        fetcher = EpisodeFetcher()
        buf: list[dict] = []
        cursor = 0
        hw = self.cfg["frame_hw"]
        future, k_set = self.meta["future_frames"], list(self.cfg.get("k_set") or self.meta["k_set"])
        # Resume: the batches already trained are replayed for their RNG draws only. Until the first live batch
        # the buffer holds metadata-only windows (no fetch, no decode); the ones still buffered then get their
        # frames. Windows whose labels run past the video end are not known without decoding, so the window
        # order after a resume can differ slightly from the uninterrupted run.
        dry = self.skip > 0

        def refill(need_k: int) -> None:
            nonlocal cursor
            while len(buf) < self.cfg["shuffle_windows"] or sum(1 for w in buf if need_k <= w["kmax"]) < self.wpr:
                ep = self.episodes[mine[cursor % len(mine)]]
                cursor += 1
                try:
                    buf.extend(dry_windows(ep) if dry else episode_windows(ep, self.meta, fetcher, hw))
                except Exception as e:  # a failed fetch/decode skips the episode, never the run
                    print(f"[stream] skip episode {ep['episode']}: {e!r}", flush=True)
                if cursor > 4 * len(mine) + 64:
                    raise RuntimeError("no decodable windows")

        def materialize() -> None:
            """Fetch the frames of the metadata-only windows left in the buffer when skipping ends."""
            nonlocal buf
            by_ep: dict[int, list[dict]] = {}
            for w in buf:
                if w["frames"] is None:
                    by_ep.setdefault(int(w["episode"]), []).append(w)
            episodes = {int(e["episode"]): e for e in self.episodes}
            for ep_id, ws in by_ep.items():
                try:
                    real = {r["anchor"]: r for r in episode_windows(episodes[ep_id], self.meta, fetcher, hw)}
                except Exception as e:
                    print(f"[stream] skip episode {ep_id}: {e!r}", flush=True)
                    real = {}
                for w in ws:
                    r = real.get(w["anchor"])
                    if r is not None:
                        w.update(r)
            buf = [w for w in buf if w["frames"] is not None]

        # Worker `wid` serves loader batches wid, wid + num_workers, ... (torch DataLoader round-robin).
        for b_local in range(self.batches_per_worker):
            b_rank = b_local * self.num_workers + wid
            skipped = b_rank < self.skip  # already trained (resume): draw its windows, yield nothing
            if dry and not skipped:
                dry = False
                materialize()
            if self.cfg.get("k_schedule"):  # debugging: fixed k per rank-local batch index (a list per rank, or one)
                sched = self.cfg["k_schedule"]
                if isinstance(sched[0], (list, tuple)):
                    sched = sched[self.rank % len(sched)]
                k = sched[b_rank % len(sched)]
            else:
                k = batch_k(self.seed, self.epoch, b_rank * self.world + self.rank, k_set, self.cfg["k0_prob"])
            k_eff = max(k, 1)  # k = 0 trims the context away but any window qualifies
            refill(k_eff)
            for _ in range(self.wpr):
                fits = [i for i, w in enumerate(buf) if w["kmax"] >= k_eff]
                win = buf.pop(rng.choice(fits))
                seed = rng.getrandbits(62)
                if skipped:
                    continue
                yield {"images.head": trim(win, k, future), "task": win["caption"], "context_frames": k,
                       "window_seed": seed, "episode": int(win["episode"]), "anchor": win["anchor"]}
                if len(buf) < self.cfg["shuffle_windows"] // 2:
                    refill(k_eff)


def build(config, **kwargs) -> StreamingWindows:
    """flux-action dataset plug: ``config.dataset_kwargs`` overrides DEFAULTS."""
    return StreamingWindows(dict(getattr(config, "dataset_kwargs", {}) or {}), **kwargs)


def fixed_windows(split: str, n: int, seed: int = 0, splits_path: str = DEFAULTS["splits"]) -> list[dict]:
    """A deterministic list of ``n`` (episode, window) picks of a split for evaluation, spread over episodes."""
    data = load_splits(splits_path)
    eps = data["splits"][split]
    rng = random.Random(f"eval-{split}-{seed}")
    picks = [(e, w) for e in eps for w in e["windows"] if max(w["valid_k"]) == max(data["meta"]["k_set"])]
    rng.shuffle(picks)
    return [{"episode": e, "window": w} for e, w in picks[:n]]

