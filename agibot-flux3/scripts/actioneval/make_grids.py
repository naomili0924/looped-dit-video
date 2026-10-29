"""Web videos of the action eval page: per window and depth one grid, 6 rows (p0 = own caption, p1-p5 = probes) x 3
columns (seeds 0, 1, 2), each tile 224x168; plus the real footage at 224x168 and the start frame as a poster. One grid
per window keeps the page under the artifact file-count limit and plays every seed of every prompt in sync.

    python3 scripts/actioneval/make_grids.py configs/eval_cases_action.json results/actioneval/rlong9500 <web dir> [--depths 1]
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

W, H = 224, 168
ENC = ["-c:v", "libx264", "-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]
SEEDS = (0, 1, 2)


def ff(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("run")
    ap.add_argument("web")
    ap.add_argument("--depths", type=int, nargs="+", default=[1])
    a = ap.parse_args()
    run, web = Path(a.run), Path(a.web)
    cases = json.loads(Path(a.cases).read_text())["cases"]
    jobs = []
    for c in cases:
        cid, pks = c["id"], list(c["prompts"])
        for d in a.depths:
            dst = web / "grid" / f"{cid}_N{d}.mp4"
            clips = [run / "clips" / cid / f"{pk}_N{d}_s{s}.mp4" for pk in pks for s in SEEDS]
            if dst.exists() or not all(p.exists() for p in clips):
                continue
            ins = [x for p in clips for x in ("-i", str(p))]
            scale = "".join(f"[{i}]scale={W}:{H}[v{i}];" for i in range(len(clips)))
            layout = "|".join(f"{(i % 3) * W}_{(i // 3) * H}" for i in range(len(clips)))
            flt = scale + "".join(f"[v{i}]" for i in range(len(clips))) + f"xstack=inputs={len(clips)}:layout={layout}"
            jobs.append(ins + ["-filter_complex", flt, *ENC, str(dst)])
        g = run / "gt" / f"{cid}.mp4"
        if g.exists():
            for sub, args in (("gt", ["-vf", f"scale={W}:{H}", *ENC]), ("poster", ["-frames:v", "1", "-vf", f"scale={W * 2}:{H * 2}", "-q:v", "4"])):
                dst = web / sub / (f"{cid}.mp4" if sub == "gt" else f"{cid}.jpg")
                if not dst.exists():
                    jobs.append(["-i", str(g), *args, str(dst)])
    for sub in ("grid", "gt", "poster"):
        (web / sub).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(ff, jobs))
    size = sum(p.stat().st_size for p in web.rglob("*") if p.is_file())
    print(f"{len(jobs)} files written; {web} holds {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
