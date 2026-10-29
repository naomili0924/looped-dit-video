"""Balanced selection: every action of the index gets the same number of probes, and every chosen window gets 6
prompts (its own caption as the control + 5 probes). Writes the cases file ldvflux.cap_eval rolls out.

Only VLM-feasible candidates are used (feasible_*.json). Actions are served scarcest first (fewest feasible windows),
each needing --per-action probes in windows of different tasks where possible; a probe goes into an already opened
window when one fits (so few windows are opened), else into a new one. Within a window the verbs are distinct and no
object is used more than twice, and a task opens at most --max-per-task windows. Opened windows short of 5 probes are topped up from the least-covered actions.

    python -I scripts/actioneval/select.py results/actioneval configs/eval_cases_action.json --per-action 3
"""
import argparse
import collections
import json
import random
from pathlib import Path

SLOTS = 5  # probes per window; + the control = 6 prompts


def ok_feasible(answers: dict, c: dict) -> bool:
    """The VLM's yes; turning the base names no object, so it is possible from any frame (the VLM tends to say no)."""
    return answers.get(c["prompt"]) == "yes" or c["verb"] == "turn base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--per-action", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--max-per-task", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.root)
    windows = {w["id"]: w for w in json.loads((root / "candidates.json").read_text())["windows"]}
    feas = {}
    for p in sorted(root.glob("feasible_*.json")):
        feas |= json.loads(p.read_text())
    rng = random.Random(a.seed)
    by_entry = collections.defaultdict(list)  # entry -> [(window id, candidate)]
    for wid, w in windows.items():
        if wid not in feas:
            continue
        for c in w["candidates"]:
            if ok_feasible(feas[wid], c):
                by_entry[c["entry"]].append((wid, c))
    all_entries = sorted({c["entry"] for w in windows.values() for c in w["candidates"]})
    chosen = collections.defaultdict(list)  # window -> [candidate]
    entry_tasks = collections.defaultdict(set)
    count = collections.Counter()

    def fits(wid, c):
        ws = chosen[wid]
        return (len(ws) < SLOTS and all(x["verb"] != c["verb"] for x in ws)
                and sum(x["object"] == c["object"] for x in ws if c["object"]) < 2)

    def opened():
        return sum(1 for v in chosen.values() if v)

    def task_full(wid):  # a new window only while its task has fewer than --max-per-task opened ones (scene variety)
        t = windows[wid]["task"]
        return not chosen[wid] and sum(1 for w, v in chosen.items() if v and windows[w]["task"] == t) >= a.max_per_task

    for e in sorted(by_entry, key=lambda e: (len({w for w, _ in by_entry[e]}), e)):
        opts = by_entry[e][:]
        rng.shuffle(opts)
        while count[e] < a.per_action:
            ok = [(w, c) for w, c in opts if fits(w, c) and not any(x is c for x in chosen[w]) and not task_full(w)]
            if a.max_windows and opened() >= a.max_windows:
                ok = [(w, c) for w, c in ok if chosen[w]]
            if not ok:
                break
            # new task for this action first, then an already opened window, then the fullest one
            w, c = min(ok, key=lambda x: (windows[x[0]]["task"] in entry_tasks[e], not chosen[x[0]], -len(chosen[x[0]])))
            chosen[w].append(c)
            entry_tasks[e].add(windows[w]["task"])
            count[e] += 1
    for wid in [w for w, v in chosen.items() if v]:  # top up opened windows to 5 probes
        while len(chosen[wid]) < SLOTS:
            opts = [c for c in windows[wid]["candidates"] if ok_feasible(feas[wid], c) and fits(wid, c)
                    and not any(x is c for x in chosen[wid])]
            if not opts:
                break
            c = min(opts, key=lambda c: (count[c["entry"]], rng.random()))
            chosen[wid].append(c)
            count[c["entry"]] += 1

    cases = []
    for wid, cs in sorted(chosen.items()):
        if not cs:
            continue
        w = windows[wid]
        probes = [w["control"]] + cs
        cases.append({k: w[k] for k in ("id", "source", "split", "task", "task_name", "episode", "seg", "anchor",
                                         "segment_frames", "caption", "skill", "init_scene")}
                     | {"family": w["control"]["family"], "prompts": {f"p{i}": c["prompt"] for i, c in enumerate(probes)},
                        "probes": {f"p{i}": c for i, c in enumerate(probes)}, "scene": w["scene"]})
    short = {e: count[e] for e in all_entries if count[e] < a.per_action}
    spec = {"note": "Action eval cases (scripts/actioneval). Each case is an initial window of a never-trained AgiBot World "
                    "Beta episode (val / test / test_unseen of Jinyan0924/agibot-flux3-eval-splits) with 6 prompts: p0 = the "
                    "episode's own caption (control, the real footage shows it), p1-p5 = probes from the action index, each "
                    "labelled in probes.<p> with its group, action entry, object and training exposure level.",
            "per_action": a.per_action, "counts": dict(sorted(count.items())), "short": short, "cases": cases}
    Path(a.out).write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    g = collections.Counter(c["group"] for cs in chosen.values() for c in cs)
    print(f"{len(cases)} windows, {sum(len(v) for v in chosen.values())} probes {dict(g)}; actions {len(all_entries)}, "
          f"at quota {sum(count[e] >= a.per_action for e in all_entries)}, short {len(short)}")
    for e, n in sorted(short.items(), key=lambda x: x[1]):
        print(f"  short: {e:28s} {n}/{a.per_action}  (feasible windows {len({w for w, _ in by_entry.get(e, [])})})")


if __name__ == "__main__":
    main()
