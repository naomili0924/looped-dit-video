"""Export the evaluation splits (val / test / test_unseen) as a self-contained dataset: the original head-camera
mp4 of every episode (the exact bytes our loader decodes, so another machine reproduces the frames bit for bit),
the window definitions, the fixed evaluation picks, a standalone loader and the protocol. Optionally uploads it
to a private Hugging Face dataset repo.

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/export_eval_dataset.py \
        --out /dev/shm/export/agibot-flux3-eval --workers 8 --upload --repo Jinyan0924/agibot-flux3-eval-splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tar_index (RangeReader)

from ldvflux.stream import EpisodeFetcher, fixed_windows, load_splits, window_frame_indices

FRAME_HW = (384, 512)
FIXED = {"val": [(0, 40)], "test": [(0, 1000)], "test_unseen": [(0, 1000)]}  # (seed, n) lists to record per split


def fetch_episode(ep: dict, dst: Path) -> dict:
    fetcher = EpisodeFetcher()
    if dst.exists() and dst.stat().st_size == ep["size"]:
        data = dst.read_bytes()
    else:
        data = fetcher.fetch(ep)
        if len(data) != ep["size"]:
            raise RuntimeError(f"episode {ep['episode']}: got {len(data)} bytes, expected {ep['size']}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    import av, io  # decode check: the container opens and reports a video stream

    with av.open(io.BytesIO(data)) as c:
        s = c.streams.video[0]
        info = {"frames": s.frames, "fps": float(s.average_rate) if s.average_rate else None,
                "width": s.codec_context.width, "height": s.codec_context.height}
    return {"episode": ep["episode"], "task": ep["task"], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "source_tar": ep["tar"], "offset": ep["offset"], **info}


def window_rows(ep: dict, meta: dict) -> list[dict]:
    stride, future = meta["stride"], meta["future_frames"]
    rows = []
    for w in ep["windows"]:
        kmax = max(w["valid_k"])
        idx = window_frame_indices(w["anchor"], kmax, stride, future)
        rows.append({"episode": ep["episode"], "task": ep["task"], "task_name": ep.get("task_name"), "seg": w["seg"],
                     "anchor": w["anchor"], "valid_k": w["valid_k"], "caption": w["caption"],
                     "context_frames_30fps_k%d" % kmax: idx[:kmax], "future_frames_30fps": idx[kmax:]})
    return rows


LOADER = '''"""Standalone loader for the AgiBot Flux-3 evaluation splits (needs: av, numpy, torch).

    from load_windows import load_split, load_window, fixed_picks
    rows = load_split("windows/val.jsonl")                       # window definitions
    picks = fixed_picks("fixed/val_seed0_n40.json")              # the fixed evaluation windows, in order
    frames = load_window("videos/val/685032.mp4", anchor=100, k=9)  # uint8 (9 + 33, 3, 384, 512), 15 fps

A window = k context frames (oldest first) then 33 future frames, all 2 source frames (30 fps) apart, i.e. 15 fps.
Frame i of the window is source frame  anchor - (k - i) * 2  for i < k  and  anchor + (i - k) * 2  for i >= k.
Frames are resized from the source resolution to 384 x 512 with torch bilinear interpolation (antialias=True),
rounded and clipped to uint8 -- the same code the training and evaluation used.
"""
import io
import json

import numpy as np
import torch

STRIDE, FUTURE, HW = 2, 33, (384, 512)


def window_frame_indices(anchor, k, stride=STRIDE, future=FUTURE):
    return [anchor - (k - j) * stride for j in range(k)] + [anchor + j * stride for j in range(future)]


def decode_frames(data: bytes, wanted, hw=HW):
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


def load_window(mp4_path, anchor, k, future=FUTURE, hw=HW):
    idx = window_frame_indices(anchor, k, STRIDE, future)
    frames = decode_frames(open(mp4_path, "rb").read(), set(idx), hw)
    missing = [i for i in idx if i not in frames]
    if missing:
        raise ValueError(f"{mp4_path}: frames {missing[:3]} past the end of the video")
    return np.stack([frames[i] for i in idx])


def load_split(path):
    return [json.loads(l) for l in open(path)]


def fixed_picks(path):
    return json.load(open(path))["picks"]
'''


def write_readme(out: Path, meta: dict, counts: dict) -> None:
    rows = "\n".join(f"| {s} | {c['episodes']} | {c['windows']} | {c['GB']:.1f} |" for s, c in counts.items())
    out.joinpath("README.md").write_text(f"""---
license: cc-by-nc-sa-4.0
task_categories: [video-generation]
pretty_name: AgiBot Flux-3 evaluation splits
---
# AgiBot World Beta — evaluation splits for the Flux-3 video fine-tune

Private mirror of the **evaluation** episodes used in the Flux-3-action video fine-tune on AgiBot World Beta,
so that the same validation / test windows can be reproduced on another machine without re-indexing the
original tars. Source: `agibot-world/AgiBotWorld-Beta` (head camera, `observations/<task>/*.tar`), licence
CC BY-NC-SA 4.0 — this mirror inherits that licence and is non-commercial research use only.

| split | episodes | windows | head video GB |
|---|---:|---:|---:|
{rows}

Split rule (seed {meta['seed']}): {meta['split_rule']}. Held-out tasks (test_unseen): {", ".join(meta['test_unseen_tasks'])}.
The training split (7,960 episodes, 188 GB) is **not** included; `windows/train.jsonl` lists its windows.

## Layout

- `videos/<split>/<episode>.mp4` — the episode's head-camera video, **byte-identical** to the AgiBot release
  (sha256 in `episodes/<split>.json`). 30 fps, source resolution.
- `windows/<split>.jsonl` — one line per window: episode, task, caption, `anchor` (30 fps index of the first
  future frame), `valid_k` (which context lengths fit inside the video), the 30 fps frame indices of the
  context (at k = 17) and of the 33 future frames.
- `fixed/<split>_seed<S>_n<N>.json` — the fixed evaluation picks (episode, anchor, caption) in order:
  `val_seed0_n40.json` = the checkpoint-selection set (windows 0–31: validation loss; 32–39: sample grids);
  `test_seed0_n1000.json`, `test_unseen_seed0_n1000.json` = the Phase-4 sets (first 200: PSNR/SSIM/LPIPS,
  first 1000: FVD, first 100: prompt-adherence judge).
- `load_windows.py` — standalone loader (av, numpy, torch) with the exact decode + resize code.
- `meta.json` — protocol constants.

## Protocol

- Task: k past frames + caption → 33 future frames, 15 fps (source 30 fps, stride 2), frames resized to
  384 × 512 (bilinear, antialias). k ∈ {{0, 1, 5, 9, 17}} at evaluation (0 = text only); 1 + 4n context frames
  because of the 4× causal temporal VAE.
- Window frames: context frame j (oldest first) is source frame `anchor - (k - j) * 2`; future frame j is
  `anchor + j * 2`.
- Validation loss: velocity (flow-matching) MSE over 32 fixed windows × 8 timesteps at the midpoints of 8
  equal-mass bins of the training timestep distribution (logit-normal; see `meta.json`), fixed noise seeds
  `1000 * window_batch_index + timestep_index`.
- Sampling for metrics: Euler, 30 steps, CFG 4, shift 5, seed per window.
- Captions are AgiBot's own per-window annotations.
""")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-json", default="/workspace/loop-dit/data/splits.json")
    p.add_argument("--splits", nargs="+", default=["val", "test", "test_unseen"])
    p.add_argument("--out", default="/dev/shm/export/agibot-flux3-eval")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--repo", default="Jinyan0924/agibot-flux3-eval-splits")
    p.add_argument("--limit", type=int, default=None, help="smoke test: episodes per split")
    args = p.parse_args()
    data = load_splits(args.splits_json)
    meta = data["meta"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in args.splits:
        eps = data["splits"][split][: args.limit]
        (out / "windows").mkdir(exist_ok=True)
        with (out / "windows" / f"{split}.jsonl").open("w") as f:
            n_win = 0
            for ep in eps:
                for r in window_rows(ep, meta):
                    f.write(json.dumps(r) + "\n")
                    n_win += 1
        (out / "fixed").mkdir(exist_ok=True)
        for seed, n in FIXED.get(split, []):
            picks = fixed_windows(split, n, seed=seed, splits_path=args.splits_json)
            (out / "fixed" / f"{split}_seed{seed}_n{n}.json").write_text(json.dumps({
                "split": split, "seed": seed, "n": n,
                "rule": "windows valid at k = 17, shuffled with random.Random(f'eval-{split}-{seed}'), first n",
                "picks": [{"i": i, "episode": pk["episode"]["episode"], "task": pk["episode"]["task"],
                           "anchor": pk["window"]["anchor"], "caption": pk["window"]["caption"]}
                          for i, pk in enumerate(picks)]}, indent=1))
        t0 = time.time()
        results, failed = [], []
        with ProcessPoolExecutor(args.workers) as pool:
            futs = {pool.submit(fetch_episode, ep, out / "videos" / split / f"{ep['episode']}.mp4"): ep for ep in eps}
            for i, fut in enumerate(as_completed(futs), 1):
                ep = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    failed.append({"episode": ep["episode"], "error": repr(e)})
                    print(f"[{split}] episode {ep['episode']} FAILED: {e!r}", flush=True)
                if i % 25 == 0 or i == len(eps):
                    gb = sum(r["bytes"] for r in results) / 2**30
                    print(f"[{split}] {i}/{len(eps)} episodes, {gb:.1f} GB, {time.time() - t0:.0f}s", flush=True)
        results.sort(key=lambda r: r["episode"])
        (out / "episodes").mkdir(exist_ok=True)
        (out / "episodes" / f"{split}.json").write_text(json.dumps({"episodes": results, "failed": failed}, indent=1))
        counts[split] = {"episodes": len(results), "windows": n_win, "GB": sum(r["bytes"] for r in results) / 2**30,
                         "failed": len(failed)}
        print(f"[{split}] done: {counts[split]}", flush=True)
    # the training windows (definitions only) and the protocol
    with (out / "windows" / "train.jsonl").open("w") as f:
        for ep in data["splits"]["train"]:
            for r in window_rows(ep, meta):
                f.write(json.dumps(r) + "\n")
    cfg = json.load(open("/workspace/loop-dit/configs/video_ft_384_r2.json"))
    from flux_action.config import PolicyConfig

    pc = PolicyConfig(**{**cfg["policy"], "torch_dtype": "bfloat16"})
    (out / "meta.json").write_text(json.dumps({
        **meta, "frame_hw": list(FRAME_HW), "eval_k": [0, 1, 5, 9, 17], "train_k_set": cfg["dataset_kwargs"]["k_set"],
        "video_logit_mean": pc.video_logit_mean, "video_logit_std": pc.video_logit_std,
        "eval": cfg["dataset_kwargs"]["eval"], "counts": counts}, indent=1))
    (out / "load_windows.py").write_text(LOADER)
    shutil.copy(args.splits_json, out / "splits.json")
    write_readme(out, meta, counts)
    print(json.dumps(counts, indent=1), flush=True)
    if args.upload:
        from huggingface_hub import HfApi

        from hub_backup import token

        api = HfApi(token=token())
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        t0 = time.time()
        api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(out), print_report=False)
        print(f"uploaded to {args.repo} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
