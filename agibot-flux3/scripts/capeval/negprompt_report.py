"""Negative-prompt test report (run_negprompt.sh): judge scores by arm × prompt type / exposure level / split, the real-action
fallback (each clip judged against the case's real caption), mean motion, and per-prompt paired gains over cfg4.

    python3 -I scripts/capeval/negprompt_report.py results/capeval/negprompt cfg4 neg4 neg7 cfg4neg3 cfg7
Writes <root>/report.json.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

root, arms = Path(sys.argv[1]), sys.argv[2:]
base = Path(__file__).resolve().parents[2]
ladder = {(r["case"], r["prompt"]): r["level"] for r in json.load(open(base / "results/capeval/prompt_ladder_v2.json"))}
cases = {c["id"]: c for c in json.load(open(base / "configs/eval_cases_capability_v2.json"))["cases"]}


def rows(arm):
    j, m = json.load(open(root / arm / "judge.json")), json.load(open(root / arm / "metrics.json"))
    for k, v in j.items():
        if k.startswith(("cross/", "gt/")):
            continue
        cid, rest = k.split("/")
        pk = rest.split("_N")[0]
        c, mo = j.get("cross/" + k), m.get(k, {}).get("motion")
        yield dict(arm=arm, case=cid, prompt=pk, level=ladder.get((cid, pk), "-"), split=cases[cid]["split"], score=v["score"],
                   attempted=v.get("attempted") == "yes", plausible=v.get("plausible") == "yes",
                   fallback=c["score"] if c else None, motion=float(np.mean(mo)) if mo else None)


allr = [r for a in arms for r in rows(a)]
report = {}
for name, key in [("arm", lambda r: r["arm"]), ("prompt", lambda r: f"{r['prompt']}|{r['arm']}"),
                  ("level", lambda r: f"{r['level']}|{r['arm']}"), ("split", lambda r: f"{r['split']}|{r['arm']}")]:
    g = collections.defaultdict(list)
    for r in allr:
        g[key(r)].append(r)
    report[name] = {}
    print()
    for k in sorted(g):
        rs = g[k]
        f = {x: float(np.mean([r[x] for r in rs if r[x] is not None])) if any(r[x] is not None for r in rs) else None
             for x in ("score", "attempted", "plausible", "fallback", "motion")}
        report[name][k] = {"n": len(rs), **f}
        print(f"{k:28s} n={len(rs):3d} " + " ".join(f"{x}={'-' if v is None else f'{v:.2f}'}" for x, v in f.items()))
g = collections.defaultdict(list)
for r in allr:
    g[(r["case"], r["prompt"], r["arm"])].append(r["score"])
keys = sorted({(r["case"], r["prompt"]) for r in allr if r["prompt"] != "real"})
report["paired_vs_cfg4"] = {}
print()
for a in arms:
    if a == "cfg4":
        continue
    d = np.array([np.mean(g[k + (a,)]) - np.mean(g[k + ("cfg4",)]) for k in keys if g[k + (a,)]])
    ci = np.percentile([np.random.default_rng(i).choice(d, len(d)).mean() for i in range(4000)], [2.5, 97.5])
    report["paired_vs_cfg4"][a] = {"mean": float(d.mean()), "ci95": ci.tolist(), "better": int((d > 0).sum()),
                                   "worse": int((d < 0).sum()), "n": len(d)}
    print(f"{a} vs cfg4: {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  better {(d > 0).sum()} worse {(d < 0).sum()} of {len(d)}")
(root / "report.json").write_text(json.dumps(report, indent=1))
