"""Head-camera mp4 of never-trained AgiBot World Beta episodes, by HTTP Range requests (no full tar download).

Episodes inside a tar are not in id order, so a given episode cannot be bisected to. Instead, per task, the first
--per-tar episodes of each tar are indexed with scripts/tar_index.py's episode walk (videos + bisection over the
depth PNGs), and the slot's candidates are drawn from the indexed episodes that are in no split (never trained,
never fetched for training) and have a segment matching the slot. test_unseen episodes come from the eval-splits
mirror. Output: <out>/videos/<split>/<episode>.mp4 (the layout ldvflux.rollout reads) and <out>/picked.json.

    HF_TOKEN=... python -I fetch_eps.py pool.json /dev/shm/capeval/mirror --per-slot 3
"""
import argparse, json, os, random, sys, tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/tar_index.py
from tar_index import BLOCK, RangeReader, header_at_or_after, walk_from  # noqa: E402

API = "https://huggingface.co/api/datasets/agibot-world/AgiBotWorld-Beta/tree/main/observations/"
MIRROR = "https://huggingface.co/datasets/Jinyan0924/agibot-flux3-eval-splits/resolve/main/videos/"
H = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}


def head_episodes(reader, n_max: int) -> dict:
    """tar_index.index_episodes, stopped after n_max episodes: {ep: {"videos": {camera: [offset, size]}}}."""
    episodes, pos = {}, 0
    while pos < reader.size and len(episodes) < n_max:
        _, ep_start = walk_from(reader, pos, lambda n: bool(n))
        if ep_start >= reader.size:
            break
        ep = tarfile.TarInfo.frombuf(reader.read(ep_start, BLOCK), tarfile.ENCODING, "surrogateescape").name.split("/")[0]
        depth = f"{ep}/depth/"
        members, depth_start = walk_from(reader, ep_start, lambda n: n.startswith(depth) and n.endswith(".png")
                                         or not n.startswith(f"{ep}/") and n != ep)
        videos = {Path(m["name"]).stem: [m["offset"], m["size"]] for m in members if m["name"].endswith(".mp4")}

        def in_depth(p):
            h = header_at_or_after(reader, p)
            return h is not None and h[1].startswith(depth)

        end = depth_start
        if depth_start < reader.size and in_depth(depth_start):
            lo, step = depth_start, 64 << 20
            hi = min(lo + step, reader.size)
            while hi < reader.size and in_depth(hi):
                lo, step = hi, step * 2
                hi = min(lo + step, reader.size)
            while hi - lo > 256 * 1024:
                mid = (lo + hi) // 2 // BLOCK * BLOCK
                lo, hi = (mid, hi) if in_depth(mid) else (lo, mid)
            tail, end = walk_from(reader, header_at_or_after(reader, lo)[0], lambda n: not n.startswith(f"{ep}/"))
            videos.update({Path(m["name"]).stem: [m["offset"], m["size"]] for m in tail if m["name"].endswith(".mp4")})
        episodes[ep] = {"videos": videos}
        if end <= pos:
            break
        pos = end
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", help="{slot: [segment rows]} from select.py")
    ap.add_argument("out")
    ap.add_argument("--per-slot", type=int, default=3)
    ap.add_argument("--per-tar", type=int, default=8)
    a = ap.parse_args()
    pool, out = json.load(open(a.pool)), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx_p = out / "tar_heads.json"
    heads = json.loads(idx_p.read_text()) if idx_p.exists() else {}  # tar path -> {"episodes": {ep: videos}}
    tasks = sorted({c["task"] for cs in pool.values() for c in cs if c["split"] == "unsampled"})

    def index_task(task):
        got = {}
        for x in requests.get(API + task, headers=H, timeout=60).json():
            if x["path"] not in heads:
                r = RangeReader(x["path"])
                got[x["path"]] = {"episodes": head_episodes(r, a.per_tar), "requests": r.requests}
        return got

    with ThreadPoolExecutor(16) as ex:
        for task, got in zip(tasks, ex.map(index_task, tasks)):
            heads.update(got)
            idx_p.write_text(json.dumps(heads))
            print(f"task {task}: {sum(len(v['episodes']) for v in got.values())} episodes indexed", flush=True)
    where = {ep: (tar, v["videos"]) for tar, h in heads.items() for ep, v in h["episodes"].items()}
    picked = {}
    rng = random.Random(42)
    for slot, cs in pool.items():
        ok = [c for c in cs if c["split"] == "test_unseen" or (c["episode"] in where and "head_color" in where[c["episode"]][1])]
        rng.shuffle(ok)
        chosen, seen_e, per_task = [], set(), {}
        for c in ok:
            if c["episode"] in seen_e or per_task.get(c["task"], 0) >= 2:
                continue
            seen_e.add(c["episode"])
            per_task[c["task"]] = per_task.get(c["task"], 0) + 1
            chosen.append(c)
            if len(chosen) == a.per_slot:
                break
        picked[slot] = chosen
        for c in chosen:
            dst = out / "videos" / c["split"] / f"{c['episode']}.mp4"
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if c["split"] == "test_unseen":
                r = requests.get(MIRROR + f"test_unseen/{c['episode']}.mp4", headers=H, timeout=300)
                r.raise_for_status()
                data = r.content
            else:
                tar, vids = where[c["episode"]]
                data = RangeReader(tar).read(*vids["head_color"])
            dst.with_suffix(".part").write_bytes(data)
            dst.with_suffix(".part").replace(dst)
        print(f"[{slot}] {len(ok)} reachable, picked {[c['episode'] for c in chosen]}", flush=True)
    (out / "picked.json").write_text(json.dumps(picked, indent=1))


if __name__ == "__main__":
    main()
