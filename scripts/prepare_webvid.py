"""Download WebVid clips, decode and encode them with the Wan2.1 VAE, and write latent shards.

Raw mp4s are never stored: CPU workers download into memory and decode, the GPU encodes,
and only the latents (~160 KB per 17x256x256 clip) are kept, as WebDataset tars.

    python scripts/prepare_webvid.py --out /dev/shm/ldv/webvid50k --num-clips 50000
    # scale-up: stream shards to the HF Hub and delete them locally
    python scripts/prepare_webvid.py --out /dev/shm/ldv/webvid1m --num-clips 1000000 \
        --start-partition 5 --hub-repo <user>/webvid1m-wan-latents --delete-after-upload

Resumable: finished shards are recorded in <out>/manifest.json and their partitions skipped.
Rows come from the TempoFunk/webvid-10M partition CSVs (~10.7K rows each).
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldv.data import WEBVID_REPO, fetch_and_decode  # noqa: E402
from ldv.encoders import WAN_VAE, WanVAE  # noqa: E402


def partition_rows(index: int) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WEBVID_REPO, f"data/train/partitions/{index:04d}.csv", repo_type="dataset")
    return pd.read_csv(path)


class ShardWriter:
    def __init__(self, path: Path):
        self.path, self.tmp = path, path.with_suffix(".tar.tmp")
        self.tar = tarfile.open(self.tmp, "w")
        self.count = 0

    def add(self, key: str, files: dict[str, bytes]) -> None:
        for ext, data in files.items():
            info = tarfile.TarInfo(f"{key}.{ext}")
            info.size, info.mtime = len(data), int(time.time())
            self.tar.addfile(info, io.BytesIO(data))
        self.count += 1

    def close(self) -> None:
        self.tar.close()
        os.replace(self.tmp, self.path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--num-clips", type=int, default=50_000)
    p.add_argument("--start-partition", type=int, default=0)
    p.add_argument("--frames", type=int, default=17, help="1 + 4k frames for the Wan VAE")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--min-seconds", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--encode-batch", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--vae", default=WAN_VAE)
    p.add_argument("--hub-repo", default="", help="HF dataset repo to upload shards to")
    p.add_argument("--delete-after-upload", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = open(out / ".lock", "w")  # held for the life of the process
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another prepare_webvid.py is already writing {out}")
    for stale in shard_dir.glob("*.tar.tmp"):  # left by an interrupted run
        stale.unlink()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"shards": {}, "clips": 0}
    device = torch.device("cuda")
    vae = WanVAE(device, repo=args.vae)
    api = None
    if args.hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hub_repo, repo_type="dataset", exist_ok=True, private=True)

    pool = mp.get_context("spawn").Pool(args.workers)
    partition = args.start_partition
    t0 = time.time()
    while manifest["clips"] < args.num_clips:
        name = f"webvid-{partition:04d}"
        if name in manifest["shards"] or any(k.startswith(name + "-") for k in manifest["shards"]):
            partition += 1
            continue
        rows = partition_rows(partition)
        dur = pd.to_timedelta(rows["duration"].str.replace("PT", "").str.lower(), errors="coerce").dt.total_seconds()
        rows = rows[dur >= args.min_seconds]
        jobs = [(int(r.videoid), r.contentUrl, str(r.name), args.frames, args.fps, args.size, int(r.videoid))
                for r in rows.itertuples()]
        part_idx, writer, batch, ok, fail = 0, None, [], 0, 0

        def flush(batch):
            nonlocal writer, part_idx
            clips = torch.from_numpy(__import__("numpy").stack([b[2] for b in batch]))  # [B, T, H, W, 3]
            video = clips.permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1.0)
            lat = vae.encode(video).to(torch.bfloat16).cpu()
            for (vid, caption, _), z in zip(batch, lat):
                if writer is None:
                    writer = ShardWriter(shard_dir / f"{name}-{part_idx:02d}.tar")
                buf = io.BytesIO()
                torch.save(z.clone(), buf)
                meta = {"videoid": vid, "fps": args.fps, "num_frames": args.frames, "size": args.size}
                writer.add(str(vid), {"latent.pth": buf.getvalue(), "txt": caption.encode(),
                                      "json": json.dumps(meta).encode()})
                if writer.count >= args.shard_size:
                    finish_shard()

        def finish_shard():
            nonlocal writer, part_idx
            if writer is None or writer.count == 0:
                return
            writer.close()
            key = writer.path.stem
            if api is not None:
                api.upload_file(path_or_fileobj=str(writer.path), path_in_repo=f"shards/{writer.path.name}",
                                repo_id=args.hub_repo, repo_type="dataset")
                if args.delete_after_upload:
                    writer.path.unlink()
            manifest["shards"][key] = writer.count
            manifest["clips"] += writer.count
            manifest_path.write_text(json.dumps(manifest, indent=1))
            writer, part_idx = None, part_idx + 1

        for res in pool.imap_unordered(fetch_and_decode, jobs, chunksize=4):
            if res is None:
                fail += 1
                continue
            ok += 1
            batch.append(res)
            if len(batch) >= args.encode_batch:
                flush(batch)
                batch = []
            if manifest["clips"] + (writer.count if writer else 0) >= args.num_clips:
                break
            if (ok + fail) % 500 == 0:
                done = manifest["clips"] + (writer.count if writer else 0)
                print(f"[{name}] ok {ok} fail {fail} | total {done} clips, "
                      f"{done / (time.time() - t0):.1f} clips/s", flush=True)
        if batch:
            flush(batch)
        finish_shard()
        if not any(k.startswith(name) for k in manifest["shards"]):
            manifest["shards"][name + "-empty"] = 0
            manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[{name}] done: ok {ok} fail {fail}; total {manifest['clips']} clips", flush=True)
        partition += 1
        if manifest["clips"] >= args.num_clips:
            # Remaining jobs of this partition are abandoned; restart the pool to drop them.
            pool.terminate()
            break
    pool.close()


if __name__ == "__main__":
    main()
