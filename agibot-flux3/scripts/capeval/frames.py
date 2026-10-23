"""Contact sheet per candidate: frames at 0, 25, 50, 75, 100 % of the segment (30 fps source, 320x240 each).
usage: python -I frames.py candidates.json <mirror> <out_dir> [per_slot]"""
import json, subprocess, sys
from pathlib import Path
cands, mirror, out = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3])
per = int(sys.argv[4]) if len(sys.argv) > 4 else 3
out.mkdir(parents=True, exist_ok=True)
for slot, cs in cands.items():
    for c in cs[:per]:
        v = mirror / "videos" / c["split"] / f"{c['episode']}.mp4"
        dst = out / f"{slot.replace('/', '__')}__{c['episode']}.jpg"
        if not v.exists() or dst.exists():
            continue
        idx = [c["start"] + round(f * (c["len"] - 1)) for f in (0, .25, .5, .75, 1)]
        sel = "+".join(f"eq(n\\,{i})" for i in idx)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v), "-vf", f"select='{sel}',scale=320:240,tile=5x1",
                        "-frames:v", "1", "-vsync", "0", str(dst)], check=True)
