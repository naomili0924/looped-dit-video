"""Initial windows for the action eval: never-trained AgiBot World Beta episodes and the objects in their scene.

Episodes: the val, test and test_unseen splits of Jinyan0924/agibot-flux3-eval-splits (never trained; test_unseen are
the 5 held-out tasks). They are mirrored there as plain head-camera mp4s, so no tar walking is needed. One window per
episode, at the start of its first labelled segment of at least --min-frames source frames (the robot's start pose:
nothing grasped yet, so many actions are possible). anchor = segment start + 2, so the k = 1 context frame is the
segment's first frame (as in the capability eval). Scene objects = every object the parser finds in any caption of the
episode (the episode's own captions name what is on the table), with the span used most and the lexicon's material.
At most --per-task episodes per task, picked with a fixed seed.

    python -I scripts/actioneval/pool.py /dev/shm/actioneval/splits /dev/shm/actioneval/beta/task_info results/actioneval/pool.json
    HF_TOKEN=... python -I scripts/actioneval/pool.py ... --download /workspace/loop_dit_eval/data/actioneval/mirror
"""
import argparse
import collections
import json
import os
import random
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capeval"))
from parse import parse  # noqa: E402

MIRROR = "https://huggingface.co/datasets/Jinyan0924/agibot-flux3-eval-splits/resolve/main/videos/"
ROLES = ("patient", "tool", "target", "source", "contact")


def scene_objects(captions: list[str]) -> list[dict]:
    seen = {}
    for c in captions:
        for cl in parse(c)["clauses"]:
            for role in ROLES:
                o = cl.get(role)
                if not o or o["cat"] == "person":
                    continue
                r = seen.setdefault(o["obj"], {"obj": o["obj"], "cat": o["cat"], "mat": o["mat"], "flags": o["flags"],
                                               "spans": collections.Counter(), "roles": collections.Counter()})
                r["spans"][o["span"]] += 1
                r["roles"][role] += 1
    out = []
    for r in seen.values():
        span = r["spans"].most_common(1)[0][0]
        out.append({"obj": r["obj"], "span": span, "cat": r["cat"], "mat": r["mat"], "flags": r["flags"],
                    "mentions": sum(r["spans"].values()), "roles": dict(r["roles"])})
    return sorted(out, key=lambda o: -o["mentions"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits_dir")
    ap.add_argument("task_info")
    ap.add_argument("out")
    ap.add_argument("--splits", nargs="+", default=["test", "test_unseen", "val"])
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--min-frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download", default=None, help="mirror dir: fetch <split>/<episode>.mp4 of the pool episodes")
    a = ap.parse_args()
    sd, ti = Path(a.splits_dir), Path(a.task_info)
    eps = []
    for s in a.splits:
        for e in json.loads((sd / "episodes" / f"{s}.json").read_text())["episodes"]:
            eps.append({**e, "split": s})
    labels = {}
    for t in {e["task"] for e in eps}:
        for e in json.loads((ti / f"task_{t}.json").read_text()):
            labels[str(e["episode_id"])] = e
    rng = random.Random(a.seed)
    rng.shuffle(eps)
    per_task, windows = collections.Counter(), []
    for e in eps:
        lab = labels.get(e["episode"])
        segs = ((lab or {}).get("label_info") or {}).get("action_config") or []
        if not segs or per_task[e["task"]] >= a.per_task:
            continue
        si = next((i for i, s in enumerate(segs) if int(s["end_frame"]) - int(s["start_frame"]) >= a.min_frames), None)
        if si is None:
            continue
        s = segs[si]
        st = int(s["start_frame"])
        if st < 2 or st + 2 >= e["frames"]:
            continue
        per_task[e["task"]] += 1
        windows.append({
            "id": f"w{e['episode']}", "source": "real", "split": e["split"], "task": e["task"], "task_name": lab["task_name"],
            "episode": e["episode"], "seg": si, "anchor": st + 2, "segment_frames": int(s["end_frame"]) - st,
            "episode_frames": e["frames"], "caption": s["action_text"].strip(), "skill": s.get("skill"),
            "init_scene": lab.get("init_scene_text"), "episode_captions": [x["action_text"].strip() for x in segs],
            "objects": scene_objects([x["action_text"].strip() for x in segs]),
        })
    windows.sort(key=lambda w: (w["split"], w["task"], w["episode"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"windows": windows}, indent=1, ensure_ascii=False))
    print(f"{len(windows)} windows from {len(per_task)} tasks:", dict(collections.Counter(w["split"] for w in windows)))
    print("objects per window: median", sorted(len(w["objects"]) for w in windows)[len(windows) // 2])
    if a.download:
        import requests

        h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

        def get(w):
            dst = Path(a.download) / "videos" / w["split"] / f"{w['episode']}.mp4"
            if dst.exists():
                return 0
            dst.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(5):  # the box's DNS / the Hub drop requests now and then
                try:
                    r = requests.get(MIRROR + f"{w['split']}/{w['episode']}.mp4", headers=h, timeout=600)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(10 * (attempt + 1))
            dst.with_suffix(".part").write_bytes(r.content)
            dst.with_suffix(".part").replace(dst)
            return len(r.content)

        with ThreadPoolExecutor(16) as ex:
            n = sum(ex.map(get, windows))
        print(f"downloaded {n / 1e9:.1f} GB to {a.download}")


if __name__ == "__main__":
    main()
