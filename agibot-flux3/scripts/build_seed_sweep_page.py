"""Rebuild the "Random-Depth Seed Sweep" artifact page from the seed-sweep results.

The page is results/seedsweep/page/template.html (the published page) with its `const DATA = {...};` line replaced.
Checkpoints without local clips keep the status / metrics / grid list already in the template (the step-3000 sweep and the
reference columns were made on a box that is gone; their videos stay on the artifact). A checkpoint with local clips
(results/seedsweep/rlong<step>/clips) gets its status, times and metrics (summary.json of ldvflux.seed_metrics) from them.

    python scripts/build_seed_sweep_page.py --steps 3000 4000 5000 6000 7000 8000 9000 9500 --running 9500
Writes results/seedsweep/page/index.html and prints the grid videos changed since the last build (page/published.json)
as an Artifact `files` map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SW = ROOT / "results" / "seedsweep"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--running", type=int, default=None)
    p.add_argument("--tag", default="rlong")
    a = p.parse_args()
    tpl = (SW / "page" / "template.html").read_text()
    m = re.search(r"^const DATA = (\{.*\});$", tpl, re.M)
    data = json.loads(m.group(1))
    cases = json.loads((ROOT / "configs" / "eval_cases_gallery8.json").read_text())["cases"]
    n_prompts = {c["id"]: 1 + len(c.get("alt_prompts", [])) for c in cases}
    seeds, depths = data["seeds"], data["depths"]
    data["steps"], data["running"] = a.steps, a.running
    grids_dir = SW / "grids"
    files = {}
    for step in a.steps:
        tag = f"{a.tag}{step}"
        clips = SW / tag / "clips"
        if not clips.exists():
            data["status"].setdefault(str(step), {"seeds_done": [], "any": False, "times": {}, "metrics": None})
            continue
        done = [s for s in seeds if all((clips / c / f"p{pi}_N{d}_s{s}.mp4").exists() for c in n_prompts for pi in range(n_prompts[c]) for d in depths)]
        times = json.loads((SW / tag / "times.json").read_text()) if (SW / tag / "times.json").exists() else {}
        mean_t = {}
        for d in depths:
            t = [v for k, v in times.items() if f"_N{d}_s" in k]
            if t:
                mean_t[str(d)] = round(sum(t) / len(t), 1)
        summ = SW / tag / "summary.json"
        metrics = json.loads(summ.read_text()) if summ.exists() and len(done) == len(seeds) else None
        any_grid = False
        for c in n_prompts:
            for pi in range(n_prompts[c]):
                key = f"{tag}_{c}_p{pi}"
                g = grids_dir / f"{key}.mp4"
                if g.exists():
                    any_grid = True
                    have = [s for s in seeds if all((clips / c / f"p{pi}_N{d}_s{s}.mp4").exists() for d in depths)]
                    data["grids"][key] = {"seeds": have}
                    files[f"videos/{key}.mp4"] = str(g)
        data["status"][str(step)] = {"seeds_done": done, "any": any_grid, "times": mean_t, "metrics": metrics}
    html = tpl.replace(m.group(0), "const DATA = " + json.dumps(data) + ";")
    # the published file carries the viewer's skeleton: keep only the page content
    html = re.sub(r"^<!doctype html>.*?<body>\n", "", html, flags=re.S | re.I)
    html = re.sub(r"\s*</body></html>\s*$", "\n", html)
    (SW / "page" / "index.html").write_text(html)
    pub_p = SW / "page" / "published.json"
    pub = json.loads(pub_p.read_text()) if pub_p.exists() else {}
    digest = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in files.items()}
    changed = {k: files[k] for k in files if pub.get(k) != digest[k]}
    (SW / "page" / "pending.json").write_text(json.dumps({"files": changed, "digest": {k: digest[k] for k in changed}}, indent=1))
    print(json.dumps(changed))


if __name__ == "__main__":
    main()
