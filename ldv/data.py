"""WebVid fetching/decoding helpers and the latent-shard training stream.

Shards (written by scripts/prepare_webvid.py) are WebDataset tars whose samples hold
    <key>.latent.pth   Wan2.1 latents [16, T', H/8, W/8] bf16 (normalized)
    <key>.txt          caption
    <key>.json         {"videoid", "fps", "num_frames", ...}
"""

from __future__ import annotations

import glob
import io
import os
import random
import urllib.request
from fractions import Fraction

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .utils import rank, world_size

WEBVID_REPO = "TempoFunk/webvid-10M"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) looped-dit-video/0.1"


# ---------------------------------------------------------------------------
# Download + decode (CPU workers)
# ---------------------------------------------------------------------------


def fetch(url: str, timeout: float = 30.0, max_bytes: int = 64 << 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("video too large")
    return data


def decode_clip(data: bytes, num_frames: int, fps: float, size: int, rng: random.Random) -> np.ndarray:
    """Decode `num_frames` frames sampled at `fps` from a random start, center-crop to a square
    and resize to `size`. Returns uint8 [num_frames, size, size, 3]."""
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate or Fraction(25))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise ValueError("no frames")
    step = max(src_fps / fps, 1.0)
    span = step * (num_frames - 1)
    if span + 1 > len(frames):
        raise ValueError(f"clip too short: {len(frames)} frames at {src_fps:.1f} fps")
    start = rng.uniform(0, len(frames) - 1 - span)
    idx = [int(round(start + i * step)) for i in range(num_frames)]
    clip = np.stack([frames[i] for i in idx])  # [T, H, W, 3]
    t, h, w, _ = clip.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    clip = torch.from_numpy(clip[:, y0 : y0 + s, x0 : x0 + s]).permute(0, 3, 1, 2).float()
    clip = torch.nn.functional.interpolate(clip, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return clip.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def fetch_and_decode(job: tuple) -> tuple | None:
    """Worker entry point: (videoid, url, caption, num_frames, fps, size, seed) -> (videoid, caption, clip) or None."""
    videoid, url, caption, num_frames, fps, size, seed = job
    if torch.get_num_threads() != 1:  # many workers in parallel: one thread each, or they oversubscribe the CPUs
        torch.set_num_threads(1)
    try:
        clip = decode_clip(fetch(url), num_frames, fps, size, random.Random(seed))
        return videoid, caption, clip
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Training stream over latent shards
# ---------------------------------------------------------------------------


def list_shards(spec: str | list[str]) -> list[str]:
    """A directory, a glob, or a list of either; also passes through http(s)/pipe: urls."""
    specs = [spec] if isinstance(spec, str) else list(spec)
    out = []
    for s in specs:
        if s.startswith(("http://", "https://", "pipe:")):
            out.append(s)
        elif os.path.isdir(s):
            out += sorted(glob.glob(os.path.join(s, "*.tar")))
        else:
            out += sorted(glob.glob(s))
    if not out:
        raise FileNotFoundError(f"no shards found for {spec}")
    return out


class LatentShardStream(IterableDataset):
    """Infinite shuffled stream of (latents, caption) from WebDataset shards, split over
    (rank, dataloader worker)."""

    def __init__(self, shards: list[str], shuffle_buffer: int = 2000, seed: int = 0):
        self.shards, self.shuffle_buffer, self.seed = shards, shuffle_buffer, seed

    def _my_shards(self) -> tuple[list[str], int]:
        info = get_worker_info()
        nw, wid = (info.num_workers, info.id) if info else (1, 0)
        index, count = rank() * nw + wid, world_size() * nw
        mine = self.shards[index::count] or [self.shards[index % len(self.shards)]]
        return mine, index

    def _samples(self, shards: list[str], rng: random.Random):
        import webdataset as wds

        def keep(urls):  # shards are already split over (rank, worker) in _my_shards
            return urls

        while True:
            seen = 0
            for url in rng.sample(shards, len(shards)):
                try:
                    # Without the identity splitters WebDataset splits this single url over the
                    # dataloader workers again, and every worker but the first reads nothing.
                    for s in wds.WebDataset(url, shardshuffle=False, empty_check=False,
                                            nodesplitter=keep, workersplitter=keep):
                        lat = torch.load(io.BytesIO(s["latent.pth"]), map_location="cpu", weights_only=True)
                        seen += 1
                        yield {"latents": lat, "caption": s["txt"].decode("utf-8")}
                except Exception as e:  # a truncated shard should not kill training
                    print(f"[data] skipping {url}: {e}", flush=True)
            if not seen:  # never spin silently on shards that yield nothing
                raise RuntimeError(f"no samples could be read from {shards}")

    def __iter__(self):
        shards, index = self._my_shards()
        rng = random.Random(self.seed + index)
        buf: list[dict] = []
        for sample in self._samples(shards, rng):
            if self.shuffle_buffer <= 0:
                yield sample
                continue
            if len(buf) < self.shuffle_buffer:
                buf.append(sample)
                continue
            j = rng.randrange(len(buf))
            yield buf[j]
            buf[j] = sample


def collate(samples: list[dict]) -> dict:
    return {
        "latents": torch.stack([s["latents"] for s in samples]),
        "caption": [s["caption"] for s in samples],
    }


def make_loader(shards: str | list[str], batch_size: int, num_workers: int = 4,
                shuffle_buffer: int = 2000, seed: int = 0) -> DataLoader:
    ds = LatentShardStream(list_shards(shards), shuffle_buffer, seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
                      pin_memory=True, persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None)
