"""Index enough AgiBot World Beta tars to sample `--per-task` labeled episodes from every task.

Tar names give overlapping episode-id ranges, so an episode's tar is only known after indexing. Per task the
tars are visited in a seeded random order and indexed (scripts/tar_index.py: headers only, ~2-3% of the
tar transferred) until the indexed episodes that have task_info labels reach --per-task. Resumable: each tar's
index is written to <out>/<task>/<tar>.json once complete.

    python scripts/build_index.py --out /dev/shm/agibot/index --workers 32
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tar_index import RangeReader, index_episodes  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def labeled_ids(task_info_dir: Path, task: str) -> set[str]:
    return {str(e["episode_id"]) for e in json.loads((task_info_dir / f"task_{task}.json").read_text())}


def index_one(task: str, tar: str, out: Path) -> dict:
    dst = out / task / tar.replace(".tar", ".json")
    if dst.exists():
        return json.loads(dst.read_text())
    t0 = time.time()
    reader = RangeReader(f"observations/{task}/{tar}")
    episodes = index_episodes(reader)
    rec = {"path": f"observations/{task}/{tar}", "size": reader.size, "episodes": episodes, "complete": True,
           "requests": reader.requests, "transferred": reader.bytes, "seconds": round(time.time() - t0, 1)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(dst)
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/dev/shm/agibot/index")
    p.add_argument("--task-info", default="/dev/shm/agibot/task_info")
    p.add_argument("--tars", default=str(DATA / "beta_obs_tars.json"), help="{task: [[tar, bytes], ...]}")
    p.add_argument("--per-task", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    out, ti = Path(args.out), Path(args.task_info)
    tars = json.loads(Path(args.tars).read_text())
    order = {t: [n for n, _ in sorted(v)] for t, v in tars.items()}
    for t in order:  # seeded per task, independent of the other tasks
        random.Random(f"{args.seed}-{t}").shuffle(order[t])
    labels = {t: labeled_ids(ti, t) for t in tars}
    found: dict[str, set[str]] = {t: set() for t in tars}
    cursor = {t: 0 for t in tars}
    lock = threading.Lock()
    totals = {"tars": 0, "bytes": 0, "tar_bytes": 0}
    t_start = time.time()

    def wanted() -> list[tuple[str, str]]:
        jobs = []
        for t in tars:
            if len(found[t] & labels[t]) < min(args.per_task, len(labels[t])) and cursor[t] < len(order[t]):
                jobs.append((t, order[t][cursor[t]]))
                cursor[t] += 1
        return jobs

    wave = 0
    with ThreadPoolExecutor(args.workers) as ex:
        while jobs := wanted():
            wave += 1
            print(f"[wave {wave}] indexing {len(jobs)} tars", flush=True)
            futs = {ex.submit(index_one, t, n, out): (t, n) for t, n in jobs}
            for f in as_completed(futs):
                t, n = futs[f]
                try:
                    rec = f.result()
                except Exception as e:  # a failed tar is retried in the next run; others continue
                    print(f"[fail] {t}/{n}: {e!r}", flush=True)
                    continue
                with lock:
                    found[t] |= set(rec["episodes"])
                    totals["tars"] += 1
                    totals["bytes"] += rec.get("transferred", 0)
                    totals["tar_bytes"] += rec["size"]
                    print(f"[done] {t}/{n}: {len(rec['episodes'])} episodes, "
                          f"{len(found[t] & labels[t])} labeled for task | {totals['tars']} tars, "
                          f"{totals['bytes'] / 1e9:.1f} GB of {totals['tar_bytes'] / 1e12:.2f} TB read, "
                          f"{(time.time() - t_start) / 60:.0f} min", flush=True)
    short = {t: len(found[t] & labels[t]) for t in tars if len(found[t] & labels[t]) < args.per_task}
    print(json.dumps({"tasks": len(tars), "tars_indexed": totals["tars"], "transferred_GB": round(totals["bytes"] / 1e9, 1),
                      "tasks_below_per_task": short}), flush=True)


if __name__ == "__main__":
    main()
