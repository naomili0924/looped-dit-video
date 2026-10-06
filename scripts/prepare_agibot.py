"""AgiBot World 2026 (agibot-world/AgiBotWorld2026): humanoid egocentric clips -> FLUX 3 VAE latent shards.

The release is 334 tar.gz bundles (~30 GB each, 10.45 TB) of LeRobot v2.1 datasets; each bundle holds
~130 episodes of the AgiBot G2 robot with 7 cameras. Only the head camera (`observation.images.top_head`,
640x400, AV1, 30 fps) is used, but a bundle is a gzip stream, so every bundle has to be streamed through
in full. Each bundle's `meta/info.json` carries step-level `instruction_segments` ("The left arm picks up
the red-capped drink from the shopping cart.") and sub-task "Task Frame" spans; these become captions.

Clips: 17 frames at stride 4 (7.5 fps), the 640x400 frame cropped to 4:3 and resized to 384x512.

Diversity: bundles are visited round-robin over the 37 task folders; each task gets a clip budget
proportional to sqrt(its size) so no task dominates; per episode at most `--max-per-episode` clips, one
per step segment, spread over the episode; no two clips overlap.

    python scripts/prepare_agibot.py --out /dev/shm/ldv/agibot --target 300000 --streams 4

Resumable at bundle granularity (done_bundles.txt). The streams are the bottleneck: at ~30-100 MB/s
per stream the full release takes one to several days.
"""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import tarfile
import time
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_hiw500 import FRAMES, SPAN, STRIDE, OUT_HW, ShardWriter, fetch  # noqa: E402

REPO = "agibot-world/AgiBotWorld2026"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
FPS = 30
HEAD_KEY = "observation.images.top_head"
EPISODES_PER_BUNDLE = 133  # typical; used only to size per-task budgets before streaming
TEMPLATES = [
    "Egocentric view from a humanoid robot: {instr}",
    "First-person video from a humanoid robot's head camera in a {scene}: {instr}",
    "{instr} Seen from the humanoid robot's own head camera.",
    "Humanoid robot egocentric footage, {scene}. Task: {sub} Step: {instr}",
    "POV of a humanoid robot working in a {scene}. {instr}",
    "A humanoid robot in a {scene}, head-camera view: {instr}",
]


def list_bundles() -> list[dict]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true&expand=false") as r:
        tree = json.load(r)
    out = []
    for x in tree:
        if x["type"] == "file" and x["path"].endswith(".tar.gz") and not x["path"].startswith("simulation"):
            parts = x["path"].split("/")
            out.append({"path": x["path"], "task": "/".join(parts[:3]), "size": x.get("size") or 0})
    return out


def scene_name(task_label: str) -> str:
    s = task_label.split(" - ")[0].strip().rstrip(".").lower()
    return s or "workspace"


def make_caption(instr: str, sub: str, scene: str, rng: random.Random) -> str:
    instr = instr.strip()
    if instr and instr[-1] not in ".!?":
        instr += "."
    sub = sub.strip() or instr
    if sub and sub[-1] not in ".!?":
        sub += "."
    return rng.choice(TEMPLATES).format(instr=instr, sub=sub, scene=scene)


def plan_episode(ep: int, length: int, info: dict, task_label: str, per_episode: float, rng: random.Random) -> list[dict]:
    """Choose clip starts for one episode from its step segments (fallback: sub-task spans, then the whole episode)."""
    segs = []
    for s in (info.get("instruction_segments") or {}).get(str(ep), []):
        a, b = int(s.get("start_frame_index", 0)), int(s.get("end_frame_index", 0))
        if b - a >= SPAN and s.get("instruction"):
            segs.append((a, min(b, length), s["instruction"]))
    task_frames = [(int(x["start"]), int(x["end"]), x["frame_detail"].get("comment", ""))
                   for x in (info.get("key_frame") or {}).get(str(ep), {}).get("dual", [])
                   if x.get("frame_type_name") == "Task Frame"]
    if not segs:
        segs = [(a, min(b, length), c) for a, b, c in task_frames if b - a >= SPAN and c] or [(0, length, task_label)]
    k = int(per_episode) + (1 if rng.random() < per_episode - int(per_episode) else 0)
    k = max(1, k)
    # one clip per segment, segments chosen evenly over the episode; extra clips go to the longest segments
    order = sorted(range(len(segs)), key=lambda i: segs[i][0])
    if k <= len(order):
        chosen = [order[int(i)] for i in np.linspace(0, len(order) - 1, k).round()]
    else:
        chosen = order + sorted(range(len(segs)), key=lambda i: -(segs[i][1] - segs[i][0]))[: k - len(order)]
    clips, used = [], []
    for i in sorted(set(chosen)):
        a, b, instr = segs[i]
        for _ in range(8):  # find a start that overlaps nothing already chosen
            s = a + rng.randrange(b - a - SPAN + 1)
            if all(abs(s - u) >= SPAN for u in used):
                used.append(s)
                sub = next((c for ta, tb, c in task_frames if ta <= s < tb), "")
                clips.append({"start": s, "instr": instr, "sub": sub})
                break
    return clips


def stream_bundle(job: tuple) -> None:
    """Worker: stream one bundle, decode its planned head-camera clips, push them to the queue."""
    bundle, per_episode, seed, queue = job
    torch.set_num_threads(1)
    import av

    url = BASE + bundle["path"]
    rng = random.Random(seed)
    info, lengths, tasks, planned, n_clips = None, {}, {}, {}, 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "looped-dit-video/0.1"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for m in tar:
                    name = m.name
                    if name.endswith("meta/info.json"):
                        info = json.load(tar.extractfile(m))
                    elif name.endswith("meta/episodes.jsonl"):
                        for line in tar.extractfile(m).read().decode().splitlines():
                            if line.strip():
                                e = json.loads(line)
                                lengths[int(e["episode_index"])] = int(e["length"])
                                tasks[int(e["episode_index"])] = (e.get("tasks") or ["workspace"])[0]
                    elif f"/{HEAD_KEY}/" in name and name.endswith(".mp4"):
                        ep = int(re.search(r"episode_(\d+)\.mp4$", name).group(1))
                        if info is None or ep not in lengths:
                            continue
                        if ep not in planned:
                            planned[ep] = plan_episode(ep, lengths[ep], info, tasks[ep], per_episode, rng)
                        if not planned[ep]:
                            continue
                        data = tar.extractfile(m).read()
                        needed = {}
                        for ci, c in enumerate(planned[ep]):
                            for j in range(FRAMES):
                                needed.setdefault(c["start"] + j * STRIDE, []).append((ci, j))
                        frames = {ci: [None] * FRAMES for ci in range(len(planned[ep]))}
                        last = max(needed)
                        with av.open(io.BytesIO(data)) as cont:
                            st = cont.streams.video[0]
                            for i, fr in enumerate(cont.decode(st)):
                                if i in needed:
                                    arr = fr.to_ndarray(format="rgb24")
                                    h, w = arr.shape[:2]
                                    cw = min(w, int(round(h * 4 / 3)))
                                    x0 = (w - cw) // 2
                                    x = torch.from_numpy(np.ascontiguousarray(arr[:, x0 : x0 + cw])).permute(2, 0, 1).float()[None]
                                    x = torch.nn.functional.interpolate(x, size=OUT_HW, mode="bilinear", antialias=True, align_corners=False)
                                    arr = x[0].round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
                                    for ci, j in needed[i]:
                                        frames[ci][j] = arr
                                if i >= last:
                                    break
                        scene = scene_name(tasks[ep])
                        for ci, c in enumerate(planned[ep]):
                            if all(f is not None for f in frames[ci]):
                                cid = f"agi-{bundle['task'].split('/')[-1]}-{Path(bundle['path']).stem.split('.')[0]}-{ep:06d}-{c['start']:06d}"
                                meta = {"id": cid, "source": "agibot2026", "bundle": bundle["path"], "task_folder": bundle["task"],
                                        "episode": ep, "start_frame": c["start"], "instruction": c["instr"], "subtask": c["sub"],
                                        "scene": tasks[ep], "fps": FPS / STRIDE, "num_frames": FRAMES, "hw": OUT_HW}
                                queue.put(("clip", cid, make_caption(c["instr"], c["sub"], scene, rng), meta, np.stack(frames[ci])))
                                n_clips += 1
        queue.put(("done", bundle["path"], n_clips, ""))
    except Exception as e:
        queue.put(("done", bundle["path"], n_clips, f"{type(e).__name__}: {str(e)[:200]}"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=300_000)
    p.add_argument("--max-per-episode", type=float, default=12)
    p.add_argument("--streams", type=int, default=4, help="bundles streamed in parallel")
    p.add_argument("--max-bundles", type=int, default=0, help="stop after this many bundles (0 = all)")
    p.add_argument("--bundle-filter", default="", help="only bundles whose path contains this (testing)")
    p.add_argument("--encode-batch", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae", default="/dev/shm/ldv/flux3/base/video_vae.safetensors")
    p.add_argument("--hub-repo", default="")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another prepare_agibot.py is already writing {out}")
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    for stale in shard_dir.glob("*.tmp"):
        stale.unlink()

    bundles_path = out / "bundles.json"
    bundles = json.loads(bundles_path.read_text()) if bundles_path.exists() else list_bundles()
    bundles_path.write_text(json.dumps(bundles))
    by_task = defaultdict(list)
    for b in bundles:
        by_task[b["task"]].append(b)
    # per-task clip budget ~ sqrt(bundles), scaled to the target; per-episode rate from the bundle count
    weights = {t: math.sqrt(len(bs)) for t, bs in by_task.items()}
    scale = args.target / sum(weights.values())
    per_episode = {t: min(args.max_per_episode, max(0.5, weights[t] * scale / (len(bs) * EPISODES_PER_BUNDLE)))
                   for t, bs in by_task.items()}
    # round-robin over tasks so a partial run is still diverse
    order, queues = [], {t: sorted(bs, key=lambda b: b["path"]) for t, bs in by_task.items()}
    while any(queues.values()):
        for t in sorted(queues):
            if queues[t]:
                order.append(queues[t].pop(0))
    done_path = out / "done_bundles.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [b for b in order if b["path"] not in done and args.bundle_filter in b["path"]]
    if args.max_bundles:
        todo = todo[: args.max_bundles]
    print(f"{len(bundles)} bundles in {len(by_task)} task folders, {sum(b['size'] for b in bundles) / 2**40:.2f} TB; "
          f"{len(done)} done, {len(todo)} to stream; per-episode clip rates: "
          f"{ {t.split('/')[-1]: round(r, 2) for t, r in sorted(per_episode.items())} }", flush=True)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"shards": {}, "clips": 0}
    from flux_action.models.video_vae import load_video_vae

    dev = torch.device("cuda")
    vae = load_video_vae(args.vae, dev, compile_model=True)
    api = None
    if args.hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hub_repo, repo_type="dataset", exist_ok=True, private=True)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=256)
    shard_idx = len(manifest["shards"])
    writer, batch, t0, n_new = None, [], time.time(), 0

    def flush():
        nonlocal writer, shard_idx, n_new
        if not batch:
            return
        x = torch.from_numpy(np.stack([b[3] for b in batch])).permute(0, 4, 1, 2, 3).to(dev, torch.bfloat16).div_(127.5).sub_(1.0)
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            z = vae.encode(x).clone().cpu()
        for (cid, cap, meta, _), lat in zip(batch, z):
            if writer is None:
                writer = ShardWriter(shard_dir / f"agibot-{shard_idx:05d}.tar")
            buf = io.BytesIO()
            torch.save(lat.clone(), buf)
            writer.add(cid, {"latent.pth": buf.getvalue(), "txt": cap.encode(), "json": json.dumps(meta).encode()})
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

    running: dict[str, mp.Process] = {}
    pending = list(todo)
    finished = 0
    while pending or running:
        while pending and len(running) < args.streams:
            b = pending.pop(0)
            pr = ctx.Process(target=stream_bundle, args=((b, per_episode[b["task"]], args.seed + zlib.crc32(b["path"].encode()), queue),), daemon=True)
            pr.start()
            running[b["path"]] = pr
        msg = queue.get()
        if msg[0] == "clip":
            batch.append(msg[1:])
            if len(batch) >= args.encode_batch:
                flush()
        else:
            _, path, n, err = msg
            running.pop(path).join()
            finished += 1
            if err:
                print(f"[fail] {path}: {err}", flush=True)
            else:
                with open(done_path, "a") as f:
                    f.write(path + "\n")
            el = time.time() - t0
            print(f"bundle {finished}/{len(todo)} {path.split('/')[-2]}/{path.split('/')[-1]}: {n} clips | total new {n_new}, "
                  f"{n_new / el:.1f} clips/s, {el / 3600:.1f} h elapsed", flush=True)
            if manifest["clips"] + n_new >= args.target:
                pending.clear()
    flush()
    close_shard()
    print(f"done: {n_new} new clips, {manifest['clips']} total in {len(manifest['shards'])} shards", flush=True)


if __name__ == "__main__":
    main()
