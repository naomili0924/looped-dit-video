"""Sample the episode subset, split it by episode / task, and enumerate the training windows.

Window = K_MAX past frames + F future frames at 15 fps (every 2nd frame of the 30 fps head camera), inside one
`action_config` segment, captioned with that segment's `action_text`. The context length k is chosen at
training time from the window's `valid_k` (the oldest past frames are trimmed); a segment too short for K_MAX
keeps the k values that fit. Writes data/splits.json (episode lists + windows + params) and prints statistics.

    python scripts/plan_splits.py --index /dev/shm/agibot/index --task-info /dev/shm/agibot/task_info
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SRC_FPS = 30
STRIDE = 2  # 30 fps -> 15 fps (the DROID recipe rate)
F = 33  # future frames: 1 + 4 * 8 -> 9 latents when encoded alone (causal 4x VAE)
K_SET = (1, 5, 9, 17)  # context frames: 1 + 4n -> 1, 2, 3, 5 latents
K_MAX = max(K_SET)


def windows_for_segment(start: int, end: int) -> list[dict]:
    """Anchors (first future frame, 30 fps index) inside [start, end). Futures do not overlap; the first anchor
    leaves room for K_MAX context frames when the segment is long enough, else for as much context as fits."""
    fut_span = (F - 1) * STRIDE  # last future frame = anchor + fut_span
    out = []
    a = min(start + K_MAX * STRIDE, end - 1 - fut_span)
    while a + fut_span <= end - 1:
        valid = [k for k in K_SET if a - k * STRIDE >= start]
        if not valid:
            break
        out.append({"anchor": a, "valid_k": valid})
        a += F * STRIDE
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="/dev/shm/agibot/index")
    p.add_argument("--task-info", default="/dev/shm/agibot/task_info")
    p.add_argument("--per-task", type=int, default=40)
    p.add_argument("--unseen-tasks", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.025)
    p.add_argument("--test-frac", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(DATA / "splits.json"))
    args = p.parse_args()
    index_dir, ti_dir = Path(args.index), Path(args.task_info)

    episodes: dict[str, list[dict]] = {}  # task -> sampled episode records
    for task_dir in sorted(index_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        info = {str(e["episode_id"]): e for e in json.loads((ti_dir / f"task_{task}.json").read_text())}
        avail = {}
        for f in sorted(task_dir.glob("*.json")):
            rec = json.loads(f.read_text())
            for ep, e in rec["episodes"].items():
                if ep in info and "head_color" in e["videos"]:
                    off, size = e["videos"]["head_color"]
                    avail[ep] = {"tar": rec["path"], "offset": off, "size": size}
        ids = sorted(avail)
        random.Random(f"{args.seed}-{task}").shuffle(ids)
        chosen = []
        for ep in ids[: args.per_task]:
            segs = info[ep]["label_info"]["action_config"]
            wins = []
            for si, s in enumerate(segs):
                for w in windows_for_segment(int(s["start_frame"]), int(s["end_frame"])):
                    wins.append({"seg": si, **w, "caption": s["action_text"].strip()})
            if wins:
                chosen.append({"task": task, "episode": ep, "task_name": info[ep]["task_name"], **avail[ep],
                               "windows": wins})
        if chosen:
            episodes[task] = chosen

    rng = random.Random(args.seed)
    eligible = sorted(t for t, v in episodes.items() if len(v) >= args.per_task)
    unseen = sorted(rng.sample(eligible, args.unseen_tasks))
    rest = [e for t, v in sorted(episodes.items()) if t not in unseen for e in v]
    rng.shuffle(rest)
    n_val, n_test = round(len(rest) * args.val_frac), round(len(rest) * args.test_frac)
    splits = {"val": rest[:n_val], "test": rest[n_val : n_val + n_test], "train": rest[n_val + n_test :],
              "test_unseen": [e for t in unseen for e in episodes[t]]}
    for v in splits.values():
        v.sort(key=lambda e: (int(e["task"]), int(e["episode"])))

    stats = {}
    for name, eps in splits.items():
        caps = [w["caption"] for e in eps for w in e["windows"]]
        words = sorted(len(c.split()) for c in caps)
        k_counts = collections.Counter(k for e in eps for w in e["windows"] for k in w["valid_k"])
        stats[name] = {"tasks": len({e["task"] for e in eps}), "episodes": len(eps), "windows": len(caps),
                       "unique_captions": len(set(caps)),
                       "caption_words_min_median_max": [words[0], statistics.median(words), words[-1]] if words else None,
                       "windows_valid_for_k": dict(sorted(k_counts.items())),
                       "head_video_GB": round(sum(e["size"] for e in eps) / 1e9, 1)}
    meta = {"seed": args.seed, "per_task": args.per_task, "src_fps": SRC_FPS, "stride": STRIDE, "fps": SRC_FPS / STRIDE,
            "future_frames": F, "k_set": list(K_SET), "test_unseen_tasks": unseen,
            "split_rule": f"{args.unseen_tasks} whole tasks held out; remaining episodes {1 - args.val_frac - args.test_frac:.3f}"
                          f"/{args.val_frac}/{args.test_frac} train/val/test by episode",
            "stats": stats}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "splits": splits}))
    print(json.dumps(meta, indent=1))
    # Leakage check: no episode in two splits, no unseen task elsewhere.
    seen: dict[str, str] = {}
    for name, eps in splits.items():
        for e in eps:
            assert e["episode"] not in seen, f"episode {e['episode']} in {seen[e['episode']]} and {name}"
            seen[e["episode"]] = name
    assert not ({e["task"] for n in ("train", "val", "test") for e in splits[n]} & set(unseen))


if __name__ == "__main__":
    main()
