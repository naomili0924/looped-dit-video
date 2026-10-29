"""Head-camera mp4 of exactly the episodes in a capability cases file (rebuilds the /dev/shm/capeval mirror on a new box).

fetch_eps.py drew its episodes from the first --per-tar episodes of each tar of a task, so every 'unsampled' case
episode sits within that head of one of its task's tars: walk the heads (fetch_eps.head_episodes) until the episode
turns up, then Range-read its head_color mp4. test_unseen episodes come from the eval-splits mirror.

    HF_TOKEN=... python -I scripts/capeval/fetch_cases.py configs/eval_cases_capability.json /dev/shm/capeval/mirror
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fetch_eps import API, H, MIRROR, head_episodes  # noqa: E402
from tar_index import RangeReader  # noqa: E402


def fetch(case: dict, out: Path, per_tar: int) -> str:
    dst = out / "videos" / case["split"] / f"{case['episode']}.mp4"
    if dst.exists():
        return f"{case['id']}: present"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if case["split"] == "test_unseen":
        r = requests.get(MIRROR + f"test_unseen/{case['episode']}.mp4", headers=H, timeout=300)
        r.raise_for_status()
        data = r.content
    else:
        data = None
        for x in requests.get(API + case["task"], headers=H, timeout=60).json():
            r = RangeReader(x["path"])
            eps = head_episodes(r, per_tar)
            if case["episode"] in eps:
                data = r.read(*eps[case["episode"]]["videos"]["head_color"])
                break
        if data is None:
            return f"{case['id']}: episode {case['episode']} NOT FOUND in the first {per_tar} episodes of task {case['task']}'s tars"
    dst.with_suffix(".part").write_bytes(data)
    dst.with_suffix(".part").replace(dst)
    return f"{case['id']}: {len(data) / 1e6:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("--per-tar", type=int, default=8)
    a = ap.parse_args()
    cases = [c for c in json.load(open(a.cases))["cases"] if c["source"] == "real"]
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(lambda c: fetch(c, Path(a.out), a.per_tar), cases):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
