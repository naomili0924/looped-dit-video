"""Web assets of the results page: per case × prompt × depth one strip with seeds 0, 1, 2 side by side (320x240 each),
the ground truth at 320x240, and the context frame as poster. One strip instead of three clips keeps the page under the
artifact file limit and the seeds in sync.

    python3 make_strips.py <run dir> <web dir>
"""
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENC = ["-c:v", "libx264", "-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]


def ff(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def main():
    run, web = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = []
    for cdir in sorted((run / "clips").iterdir()):
        groups = {}
        for p in cdir.glob("*.mp4"):
            pk, d, s = p.stem.split("_")
            groups.setdefault((pk, d), {})[s] = p
        for (pk, d), seeds in groups.items():
            dst = web / "strips" / cdir.name / f"{pk}_{d}.mp4"
            if dst.exists() or set(seeds) != {"s0", "s1", "s2"}:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            ins = [a for s in ("s0", "s1", "s2") for a in ("-i", str(seeds[s]))]
            flt = "[0]scale=320:240[a];[1]scale=320:240[b];[2]scale=320:240[c];[a][b][c]hstack=3"
            jobs.append(ins + ["-filter_complex", flt, *ENC, str(dst)])
    for g in sorted((run / "gt").glob("*.mp4")):
        (web / "gt").mkdir(parents=True, exist_ok=True)
        (web / "poster").mkdir(parents=True, exist_ok=True)
        if not (web / "gt" / g.name).exists():
            jobs.append(["-i", str(g), "-vf", "scale=320:240", *ENC, str(web / "gt" / g.name)])
        if not (web / "poster" / f"{g.stem}.jpg").exists():
            jobs.append(["-i", str(g), "-frames:v", "1", "-vf", "scale=320:240", "-q:v", "4", str(web / "poster" / f"{g.stem}.jpg")])
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(ff, jobs))
    print(len(jobs), "files written to", web)


if __name__ == "__main__":
    main()
