"""Mine never-trained segments per action family from the full AgiBot World Beta task_info.
usage: python -I mine.py <task_info_dir> <data_splits_dir> <out.json>"""
import collections, json, os, sys
from pathlib import Path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]  # parse.py / lexicon.py are the caption atlas parser
from parse import main_clause, parse  # noqa
from full_beta_util import n_windows  # noqa
ti, ds, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
FAMS = ["pour", "rotate", "tool_use", "press", "locomotion", "transport", "gesture", "cut_shape", "mix", "tie"]
split_of = {}
for s in ("train", "val", "test", "test_unseen"):
    for l in open(ds / "windows" / f"{s}.jsonl"):
        split_of[json.loads(l)["episode"]] = s
train_caps = collections.Counter(json.loads(l)["caption"] for l in open(ds / "windows/train.jsonl"))
unseen_tasks = {json.loads(l)["task"] for l in open(ds / "windows/test_unseen.jsonl")}
P = {}
rows = []
for f in sorted(ti.glob("task_*.json")):
    tid = f.stem.split("_")[1]
    for e in json.loads(f.read_text()):
        segs = (e.get("label_info") or {}).get("action_config") or []
        ep = str(e["episode_id"])
        for si, s in enumerate(segs):
            cap = s["action_text"].strip()
            if cap not in P:
                m = main_clause(parse(cap))
                o = m and (m["patient"] or m["tool"])
                P[cap] = (m and m["family"], m and m["verb"], o and o["obj"], o and o["mat"])
            fam, verb, obj, mat = P[cap]
            st, en = int(s["start_frame"]), int(s["end_frame"])
            rows.append(dict(task=tid, name=e["task_name"], episode=ep, seg=si, start=st, end=en, len=en - st,
                             caption=cap, skill=s.get("skill"), family=fam, verb=verb, obj=obj, mat=mat,
                             split=split_of.get(ep, "unsampled"), held_out_task=tid in unseen_tasks,
                             cap_in_train=train_caps.get(cap, 0), n_win=n_windows(st, en)))
json.dump(rows, open(out, "w"))
C = collections.Counter
print("rows", len(rows))
for fam in FAMS:
    r = [x for x in rows if x["family"] == fam]
    print(f"\n== {fam}: segs {len(r)}  by split {dict(C(x['split'] for x in r))}")
    un = [x for x in r if x["split"] == "unsampled" and not x["held_out_task"]]
    tasks = C((x["task"], x["name"]) for x in un)
    print("  unsampled tasks:", len(tasks), tasks.most_common(8))
    print("  verbs:", C(x["verb"] for x in r).most_common(12))
    print("  novel-caption unsampled segs:", sum(1 for x in un if x["cap_in_train"] == 0))
