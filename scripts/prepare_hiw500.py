"""HIW-500 (BitRobot/HIW-500-LeRobot): humanoid egocentric clips -> FLUX 3 video-VAE latent shards.

Unitree G1 head camera (stereo 1280x480, AV1, 30 fps); we take the left eye, resize to 384x512 (4:3)
and cut clips of 17 frames at stride 4 (7.5 fps, ~2.1 s). Captions come from the episode's timestamped
sub-task list (`language_persistent`): the clip's sub-task plus the episode task, phrased with one of
several templates.

Diversity rules (see README): clips never overlap; every sub-task segment long enough for a clip gets at
least one; clips are spread evenly through each segment; at most `--max-per-episode` clips per episode.

    python scripts/prepare_hiw500.py plan   --out /dev/shm/ldv/hiw500 --target 200000
    python scripts/prepare_hiw500.py encode --out /dev/shm/ldv/hiw500 --workers 48

`plan` downloads the metadata and the per-frame language columns (~11 GB, discarded after reading) and
writes plan.json. `encode` streams the head-camera video files (~800 GB, never stored), decodes the
planned clips on CPU workers, encodes them with the FLUX VAE on the GPU and writes shards. Resumable.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import random
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = "BitRobot/HIW-500-LeRobot"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
FPS = 30
STRIDE = 4  # 30 fps -> 7.5 fps
FRAMES = 17
SPAN = (FRAMES - 1) * STRIDE + 1  # source frames covered by one clip
OUT_HW = (384, 512)
TEMPLATES = [
    "Egocentric view from a humanoid robot: {sub}.",
    "First-person video from a humanoid robot's head camera while it is working to {task}: {sub}.",
    "A humanoid robot {sub_verb}, seen from its own head camera.",
    "Humanoid robot egocentric footage, task: {task}. Current step: {sub}.",
    "{sub_cap}. Head-camera view of a humanoid robot in a home.",
    "POV of a humanoid robot as it {sub_verb}.",
]


def fetch(url: str, retries: int = 5) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "looped-dit-video/0.1"}), timeout=120) as r:
                return r.read()
        except Exception as e:  # transient Hub errors
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def caption(sub: str, task: str, rng: random.Random) -> str:
    sub, task = sub.strip().rstrip("."), task.strip().rstrip(".")
    verb = sub if sub.split()[0].endswith("s") else sub.replace(sub.split()[0], sub.split()[0] + "s", 1)
    return rng.choice(TEMPLATES).format(sub=sub, task=task, sub_cap=sub[:1].upper() + sub[1:], sub_verb=verb)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def read_language(path: str) -> dict[int, list[tuple[float, str]]]:
    """episode_index -> [(start_seconds, text)] from a data parquet's first row per episode."""
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["episode_index", "frame_index", "language_persistent"]).to_pandas()
    t = t[t.frame_index == 0]
    out = {}
    for ep, lp in zip(t.episode_index, t.language_persistent):
        entries = [(float(e["timestamp"]), str(e["content"])) for e in (lp if lp is not None else [])]
        out[int(ep)] = sorted(entries)
    return out


def plan(args) -> None:
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "meta"
    meta.mkdir(exist_ok=True)
    eps = []
    for i in range(2):
        p = meta / f"episodes-{i}.parquet"
        if not p.exists():
            p.write_bytes(fetch(BASE + f"meta/episodes/chunk-000/file-{i:03d}.parquet"))
        eps.append(pq.read_table(p, columns=["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
                                             "videos/observation.images.head/chunk_index", "videos/observation.images.head/file_index",
                                             "videos/observation.images.head/from_timestamp"]).to_pandas())
    import pandas as pd

    eps = pd.concat(eps).sort_values("episode_index").reset_index(drop=True)
    print(f"{len(eps)} episodes, {eps.length.sum() / FPS / 3600:.0f} hours", flush=True)

    # per-frame language columns: one data parquet (~68 MB) per ~114 episodes, read once and dropped
    lang_path = out / "language.json"
    if lang_path.exists():
        lang = {int(k): v for k, v in json.loads(lang_path.read_text()).items()}
    else:
        lang = {}
        files = sorted(set(zip(eps["data/chunk_index"], eps["data/file_index"])))
        from concurrent.futures import ThreadPoolExecutor

        def one(cf):
            c, f = cf
            tmp = out / f"tmp-{c}-{f}.parquet"
            tmp.write_bytes(fetch(BASE + f"data/chunk-{c:03d}/file-{f:03d}.parquet"))
            try:
                return read_language(str(tmp))
            finally:
                tmp.unlink()

        with ThreadPoolExecutor(8) as ex:
            for i, r in enumerate(ex.map(one, files)):
                lang.update(r)
                if i % 20 == 0:
                    print(f"language: {i + 1}/{len(files)} files", flush=True)
        lang_path.write_text(json.dumps(lang))

    # clip plan
    rng = random.Random(args.seed)
    per_ep_target = args.target / len(eps)
    clips, n_segments, by_task = [], 0, defaultdict(int)
    for row in eps.to_dict("records"):
        ep, length = int(row["episode_index"]), int(row["length"])
        task = str(list(row["tasks"])[0]) if len(row["tasks"]) else "household task"
        entries = lang.get(ep) or [(0.0, task)]
        bounds = [min(length, int(round(ts * FPS))) for ts, _ in entries] + [length]
        segs = [(bounds[i], bounds[i + 1], entries[i][1]) for i in range(len(entries)) if bounds[i + 1] - bounds[i] >= SPAN]
        n_segments += len(segs)
        # clips per segment: proportional to length, at least one; then trim the episode to the cap
        want = []
        for a, b, text in segs:
            slots = (b - a) // SPAN
            k = max(1, min(slots, int(round(slots * per_ep_target / max(length // SPAN, 1)))))
            starts = [a + int((b - a - SPAN) * (j + rng.random()) / k) for j in range(k)] if k > 1 else [a + rng.randrange(b - a - SPAN + 1)]
            starts = sorted(set(starts))
            # enforce no overlap after jitter
            kept = []
            for s in starts:
                if not kept or s - kept[-1] >= SPAN:
                    kept.append(s)
            want += [(s, text) for s in kept]
        if len(want) > args.max_per_episode:
            idx = np.linspace(0, len(want) - 1, args.max_per_episode).round().astype(int)
            want = [want[i] for i in sorted(set(idx))]
        for s, text in want:
            clips.append({"id": f"hiw-{ep:06d}-{s:06d}", "ep": ep, "start": s, "sub": text, "task": task,
                          "chunk": int(row["videos/observation.images.head/chunk_index"]),
                          "file": int(row["videos/observation.images.head/file_index"]),
                          "from_ts": float(row["videos/observation.images.head/from_timestamp"]),
                          "caption": caption(text, task, rng)})
            by_task[task] += 1
    random.Random(args.seed).shuffle(clips)
    (out / "plan.json").write_text(json.dumps({"clips": clips, "hw": OUT_HW, "frames": FRAMES, "stride": STRIDE}))
    print(f"planned {len(clips)} clips from {n_segments} sub-task segments; per task: {dict(by_task)}", flush=True)
    subs = defaultdict(int)
    for c in clips:
        subs[c["sub"]] += 1
    print(f"{len(subs)} distinct sub-task texts; most common: {sorted(subs.items(), key=lambda x: -x[1])[:8]}", flush=True)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def decode_file(job: tuple) -> list[tuple[str, np.ndarray]] | str:
    """(url, [(clip_id, first_frame_in_file)]) -> [(clip_id, uint8 [FRAMES, H, W, 3])] or an error string."""
    import av

    torch.set_num_threads(1)
    url, items = job
    try:
        data = fetch(url)
        needed = {}
        for cid, f0 in items:
            for j in range(FRAMES):
                needed.setdefault(f0 + j * STRIDE, []).append((cid, j))
        last = max(needed)
        got: dict[str, list] = {cid: [None] * FRAMES for cid, _ in items}
        with av.open(io.BytesIO(data)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for i, frame in enumerate(c.decode(s)):
                if i in needed:
                    arr = frame.to_ndarray(format="rgb24")[:, : frame.width // 2]  # left eye
                    x = torch.from_numpy(arr).permute(2, 0, 1).float()[None]
                    x = torch.nn.functional.interpolate(x, size=OUT_HW, mode="bilinear", antialias=True, align_corners=False)
                    arr = x[0].round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    for cid, j in needed[i]:
                        got[cid][j] = arr
                if i >= last:
                    break
        return [(cid, np.stack(fr)) for cid, fr in got.items() if all(f is not None for f in fr)]
    except Exception as e:
        return f"{url}: {type(e).__name__}: {e}"


class ShardWriter:
    def __init__(self, path: Path):
        self.path, self.tmp = path, path.with_suffix(".tar.tmp")
        self.tar, self.count = tarfile.open(self.tmp, "w"), 0

    def add(self, key: str, files: dict[str, bytes]) -> None:
        for ext, data in files.items():
            info = tarfile.TarInfo(f"{key}.{ext}")
            info.size, info.mtime = len(data), int(time.time())
            self.tar.addfile(info, io.BytesIO(data))
        self.count += 1

    def close(self) -> None:
        self.tar.close()
        os.replace(self.tmp, self.path)


def encode(args) -> None:
    import fcntl

    from flux_action.models.video_vae import load_video_vae

    out = Path(args.out)
    lock = open(out / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another encode is already writing {out}")
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    for stale in shard_dir.glob("*.tmp"):
        stale.unlink()
    plan_ = json.loads((out / "plan.json").read_text())
    clips = {c["id"]: c for c in plan_["clips"]}
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"shards": {}, "clips": 0}
    # Progress is what the finished shards contain (a crash loses the open shard; its clips are redone).
    done = set()
    for key in manifest["shards"]:
        with tarfile.open(shard_dir / f"{key}.tar") as tar:
            done.update(n.rsplit(".", 2)[0] for n in tar.getnames() if n.endswith(".latent.pth"))
    done_path = out / "done.txt"
    done_path.write_text("\n".join(sorted(done)) + ("\n" if done else ""))
    todo = defaultdict(list)
    for c in plan_["clips"]:
        if c["id"] not in done:
            key = (c["chunk"], c["file"])
            todo[key].append((c["id"], int(round(c["from_ts"] * FPS)) + c["start"]))
    jobs = [(BASE + f"videos/observation.images.head/chunk-{c:03d}/file-{f:03d}.mp4", items) for (c, f), items in sorted(todo.items())]
    print(f"{len(clips)} planned, {len(done)} done, {sum(len(i) for _, i in jobs)} to encode from {len(jobs)} video files", flush=True)

    dev = torch.device("cuda")
    vae = load_video_vae(args.vae, dev, compile_model=True)
    api = None
    if args.hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hub_repo, repo_type="dataset", exist_ok=True, private=True)
    shard_idx = len(manifest["shards"])
    writer, batch, t0, n_new, n_fail = None, [], time.time(), 0, 0

    def flush():
        nonlocal writer, shard_idx, n_new
        if not batch:
            return
        x = torch.from_numpy(np.stack([b[1] for b in batch])).permute(0, 4, 1, 2, 3).to(dev, torch.bfloat16).div_(127.5).sub_(1.0)
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            z = vae.encode(x).clone().cpu()
        for (cid, _), lat in zip(batch, z):
            if writer is None:
                writer = ShardWriter(shard_dir / f"hiw500-{shard_idx:05d}.tar")
            c = clips[cid]
            buf = io.BytesIO()
            torch.save(lat.clone(), buf)  # clone: a view would serialize the whole batch's storage
            meta = {"id": cid, "source": "hiw500", "episode": c["ep"], "start_frame": c["start"], "subtask": c["sub"],
                    "task": c["task"], "fps": FPS / STRIDE, "num_frames": FRAMES, "hw": OUT_HW}
            writer.add(cid, {"latent.pth": buf.getvalue(), "txt": c["caption"].encode(), "json": json.dumps(meta).encode()})
            with open(done_path, "a") as f:
                f.write(cid + "\n")
            n_new += 1
            if writer.count >= args.shard_size:
                close_shard()
        batch.clear()

    def close_shard():
        nonlocal writer, shard_idx
        if writer is None or writer.count == 0:
            return
        writer.close()
        if api is not None:
            api.upload_file(path_or_fileobj=str(writer.path), path_in_repo=f"shards/{writer.path.name}", repo_id=args.hub_repo, repo_type="dataset")
        manifest["shards"][writer.path.stem] = writer.count
        manifest["clips"] += writer.count
        manifest_path.write_text(json.dumps(manifest, indent=1))
        writer, shard_idx = None, shard_idx + 1

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for res in pool.imap_unordered(decode_file, jobs):
            if isinstance(res, str):
                n_fail += 1
                print("[fail]", res[:200], flush=True)
                continue
            for item in res:
                batch.append(item)
                if len(batch) >= args.encode_batch:
                    flush()
            if n_new and n_new % 500 < args.encode_batch:
                print(f"{n_new} clips encoded, {n_new / (time.time() - t0):.1f} clips/s, {n_fail} failed files, total {manifest['clips'] + (writer.count if writer else 0)}", flush=True)
    flush()
    close_shard()
    print(f"done: {n_new} new clips, {n_fail} failed files, {manifest['clips']} clips in {len(manifest['shards'])} shards", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["plan", "encode"])
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=200_000)
    p.add_argument("--max-per-episode", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--encode-batch", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--vae", default="/dev/shm/ldv/flux3/base/video_vae.safetensors")
    p.add_argument("--hub-repo", default="")
    args = p.parse_args()
    plan(args) if args.cmd == "plan" else encode(args)


if __name__ == "__main__":
    main()
