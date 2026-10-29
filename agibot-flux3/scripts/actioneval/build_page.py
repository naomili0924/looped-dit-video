"""Results page of the action eval: cases + judge answers + report summary -> <web>/index.html (data inlined into
actpage_template.html). Videos are make_grids.py's copies in <web>/{grid,gt,poster}.

    python3 scripts/actioneval/build_page.py configs/eval_cases_action.json results/actioneval/rlong9500 <web dir>
"""
import json
import sys
from pathlib import Path

GROUP_LABEL = {"control": "Own caption", "seen_new_object": "Seen action, new object",
               "seen_dir_new_obj": "Seen action + direction, new object", "unseen_action": "Unseen action"}


def main():
    cases_p, run, web = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    root = Path(__file__).resolve().parents[2]
    cases = json.loads(cases_p.read_text())["cases"]
    summ = json.loads((run / "summary.json").read_text())
    idx = json.loads((root / "results/actioneval/action_index.json").read_text())
    ans = {}
    for p in sorted(run.glob("judge_*.json")):
        ans |= json.loads(p.read_text())
    times = {}
    for p in run.glob("times_*.json"):
        times |= json.loads(p.read_text())
    depths = summ["depths"]
    pick = lambda a: a and {"c": a.get("completed"), "a": a.get("attempted"), "p": a.get("plausible"), "r": a.get("reason")}
    windows = []
    for c in cases:
        cid = c["id"]
        rows = []
        for pk, pr in c["probes"].items():
            rows.append({"p": pk, "prompt": pr["prompt"], "group": pr["group"], "entry": pr["entry"], "family": pr["family"],
                         "level": pr["level"], "object": pr.get("span"), "gt": pick(ans.get(f"gt/{cid}/{pk}")),
                         "clips": {d: [pick(ans.get(f"{cid}/{pk}_{d}_s{s}")) for s in (0, 1, 2)] for d in depths}})
        windows.append({"id": cid, "task": c["task_name"], "split": c["split"], "episode": c["episode"],
                        "scene": [o["span"] for o in c["scene"]], "rows": rows,
                        "grids": {d: f"grid/{cid}_{d}.mp4" for d in depths if (web / "grid" / f"{cid}_{d}.mp4").exists()},
                        "gtVideo": f"gt/{cid}.mp4" if (web / "gt" / f"{cid}.mp4").exists() else None,
                        "poster": f"poster/{cid}.jpg"})
    actions = []
    for k, r in summ["action"].items():
        g, e = k.split("|", 1)
        verb = e.split(" | ")[0]
        s = idx["seen"].get(verb)
        actions.append({"group": g, "entry": e, "family": next((x["family"] for x in summ["probes"] if x["entry"] == e), None),
                        "trainWindows": (s["dirs"].get(e.split(" | ")[1], 0) if " | " in e else s["train_windows"]) if s else 0,
                        "flag": any(x["gt_satisfies"] for x in summ["probes"] if x["entry"] == e and x["group"] == g),
                        "byDepth": r})
    sec = {d: round(sum(v for k, v in times.items() if f"_{d}_" in k) / max(1, sum(1 for k in times if f"_{d}_" in k)), 1)
           for d in depths}
    data = {"depths": depths, "groupLabel": GROUP_LABEL, "group": summ["group"], "groupClean": summ["group_clean"],
            "level": summ["level"], "split": summ["split"], "calib": summ["judge_calibration"], "windowsDone": summ["windows"],
            "actions": actions, "windows": windows, "secPerClip": sec,
            "counts": {"windows": len(cases), "probes": sum(len(c["probes"]) - 1 for c in cases), "tasks": len({c["task"] for c in cases}),
                       "actions": len(actions), "seen": sum(1 for v in idx["seen"].values() if v["testable"]),
                       "unseen": sum(1 for v in idx["unseen"].values() if v["testable"]), "dirEntries": len(idx["dir_entries"])}}
    tpl = (Path(__file__).parent / "actpage_template.html").read_text()
    findings = run / "findings.html"  # hand-written reading of the numbers, if there is one
    tpl = tpl.replace("<!--__FINDINGS__-->", findings.read_text() if findings.exists() else "")
    (web / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False)))
    print("wrote", web / "index.html", len(windows), "windows,", len(actions), "actions")


if __name__ == "__main__":
    main()
