"""Action eval report from the judge's answers only (ldvflux.act_judge): per action of the index, per group, per
training-exposure level, per family and per loop depth N, with the seed spread.

score = completed yes 1 / partial 0.5 / no 0, averaged over a probe's seeds, then over probes (each probe counts once,
whatever its seed count). success = share of clips judged "yes". 95 % intervals: bootstrap over probes.
A probe whose real footage the judge already passes ("gt_satisfies": the robot's own action does what the probe asks,
or the judge is lenient there) cannot tell following the prompt from doing the scene's usual action; it is kept in the
per-action table, marked, and left out of the "clean" group numbers.

    python -I scripts/actioneval/report.py configs/eval_cases_action.json results/actioneval/rlong9500
Writes <run>/summary.json and prints the tables (markdown).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}
GROUPS = ["control", "seen_new_object", "seen_dir_new_obj", "unseen_action"]
GROUP_LABEL = {"control": "control (own caption)", "seen_new_object": "seen action, new object",
               "seen_dir_new_obj": "seen action + direction, new object", "unseen_action": "unseen action (marked)"}


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [None, None]
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main():
    cases_p, run = Path(sys.argv[1]), Path(sys.argv[2])
    cases = {c["id"]: c for c in json.loads(cases_p.read_text())["cases"]}
    ans = {}
    for p in sorted(run.glob("judge_*.json")):
        ans |= json.loads(p.read_text())
    # probe -> depth -> [clip answers]
    clips = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, a in ans.items():
        if key.startswith("gt/"):
            continue
        cid, name = key.split("/")
        pk, d, s = name.split("_")
        clips[(cid, pk)][d].append({"seed": int(s[1:]), "score": SCORE.get(a["completed"], 0.0), "yes": a["completed"] == "yes",
                                    "attempted": a["attempted"] == "yes", "plausible": a["plausible"] == "yes"})
    gt = {tuple(k.split("/")[1:]): a["completed"] for k, a in ans.items() if k.startswith("gt/")}
    depths = sorted({d for v in clips.values() for d in v}, key=lambda d: int(d[1:]))
    probes = []
    for (cid, pk), by_d in clips.items():
        meta = cases[cid]["probes"][pk]
        g = gt.get((cid, pk))
        row = {"case": cid, "p": pk, "task": cases[cid]["task"], "split": cases[cid]["split"], "group": meta["group"],
               "entry": meta["entry"], "verb": meta["verb"], "family": meta["family"], "level": meta["level"],
               "object": meta.get("object"), "prompt": meta["prompt"], "gt": g,
               "gt_satisfies": meta["group"] != "control" and g == "yes", "by_depth": {}}
        for d, cl in by_d.items():
            sc = [c["score"] for c in cl]
            row["by_depth"][d] = {"n_seeds": len(cl), "score": float(np.mean(sc)), "success": float(np.mean([c["yes"] for c in cl])),
                                  "attempted": float(np.mean([c["attempted"] for c in cl])),
                                  "plausible": float(np.mean([c["plausible"] for c in cl])),
                                  "seeds_agree": len({c["yes"] for c in cl}) == 1, "any_yes": any(c["yes"] for c in cl)}
        probes.append(row)

    def agg(rows, d):
        r = [x["by_depth"][d] for x in rows if d in x["by_depth"]]
        if not r:
            return None
        sc = [x["score"] for x in r]
        return {"probes": len(r), "clips": sum(x["n_seeds"] for x in r), "score": round(float(np.mean(sc)), 3), "ci": boot(sc),
                "success": round(float(np.mean([x["success"] for x in r])), 3),
                "attempted": round(float(np.mean([x["attempted"] for x in r])), 3),
                "plausible": round(float(np.mean([x["plausible"] for x in r])), 3),
                "seeds_agree": round(float(np.mean([x["seeds_agree"] for x in r])), 3),
                "any_seed_yes": round(float(np.mean([x["any_yes"] for x in r])), 3)}

    def table(key, rows, clean=False):
        out = {}
        for k in sorted({key(x) for x in rows}, key=str):
            sel = [x for x in rows if key(x) == k and not (clean and x["gt_satisfies"])]
            out[str(k)] = {d: agg(sel, d) for d in depths}
        return out

    windows = {}
    for d in depths:
        per_w = collections.defaultdict(list)
        for x in probes:
            if d in x["by_depth"]:
                per_w[x["case"]].append(x["by_depth"][d]["success"] >= 0.5)
        windows[d] = {"windows": len(per_w), "mean_prompts_done_of_6": round(float(np.mean([sum(v) for v in per_w.values()])), 2),
                      "hist": dict(collections.Counter(sum(v) for v in per_w.values()))}
    calib = {g: {"gt_yes_rate": round(float(np.mean([x["gt"] == "yes" for x in probes if x["group"] == g and x["gt"]])), 3),
                 "n": sum(1 for x in probes if x["group"] == g and x["gt"])} for g in GROUPS if any(x["group"] == g and x["gt"] for x in probes)}
    summary = {"depths": depths, "n_probes": len(probes), "judge_calibration": calib,
               "group": table(lambda x: x["group"], probes), "group_clean": table(lambda x: x["group"], probes, clean=True),
               "level": table(lambda x: x["level"], [x for x in probes if x["group"] != "control"], clean=True),
               "family": table(lambda x: f'{x["group"]}|{x["family"]}', [x for x in probes if x["group"] != "control"]),
               "action": table(lambda x: f'{x["group"]}|{x["entry"]}', [x for x in probes if x["group"] != "control"]),
               "split": table(lambda x: f'{x["group"]}|{x["split"]}', probes, clean=True),
               "windows": windows, "probes": probes}
    (run / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    hd = " | ".join(f"N = {d[1:]}" for d in depths)
    print(f"Judge on the real footage: own caption yes {calib.get('control', {}).get('gt_yes_rate')}; "
          + ", ".join(f"{GROUP_LABEL[g]} yes {v['gt_yes_rate']}" for g, v in calib.items() if g != "control"))
    print(f"\n| group | probes | {hd} |\n|---|---:|" + "---:|" * len(depths))
    for g in GROUPS:
        r = summary["group_clean"].get(g)
        if r:
            n = next(v["probes"] for v in r.values() if v)
            print(f"| {GROUP_LABEL[g]} | {n} | " + " | ".join(
                f"{v['score']:.2f} [{v['ci'][0]}, {v['ci'][1]}]" if v else "-" for v in r.values()) + " |")
    print(f"\n| level | probes | {hd} |\n|---|---:|" + "---:|" * len(depths))
    for k, r in summary["level"].items():
        n = next((v["probes"] for v in r.values() if v), 0)
        print(f"| {k} | {n} | " + " | ".join(f"{v['score']:.2f}" if v else "-" for v in r.values()) + " |")
    print(f"\n| action | group | probes | {hd} | seeds agree |\n|---|---|---:|" + "---:|" * (len(depths) + 1))
    for k, r in summary["action"].items():
        g, e = k.split("|", 1)
        v0 = r[depths[0]]
        if v0:
            flag = " *" if any(x["gt_satisfies"] for x in probes if x["entry"] == e and x["group"] == g) else ""
            print(f"| {e}{flag} | {g} | {v0['probes']} | " + " | ".join(f"{v['score']:.2f}" if v else "-" for v in r.values())
                  + f" | {v0['seeds_agree']:.2f} |")
    print("\n* = at least one of its probes is passed by the real footage too (see gt_satisfies)")
    for d, w in windows.items():
        print(f"{d}: windows {w['windows']}, prompts of 6 done (majority of seeds) {w['mean_prompts_done_of_6']}, hist {w['hist']}")


if __name__ == "__main__":
    main()
