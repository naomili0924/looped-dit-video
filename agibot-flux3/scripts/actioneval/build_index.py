"""Action index for the action eval: every action of AgiBot World Beta, with how much of it training saw.

Actions are the Caption Atlas's verb primitives (scripts/capeval/parse.py + lexicon.py, the same parser), counted on
the training captions (windows/train.jsonl, weighted by windows) and on the full release (task_info, segments):
  verb                  training windows, training tasks, release segments
  verb x object         canonical objects the verb was trained on (an unseen pair = a new object for a seen action)
  verb x material       material classes the verb was trained on
  verb x direction      direction words trained with the verb (push aside, turn base leftward, ...)
  verb x obj x dir      for the "seen action + direction on a new object" probes
plus the exact training captions and every raw training word (the unseen-action check is on raw words, so that
"waving" or "torn" in a caption counts as seen even where the lexicon has no entry).

    python -I scripts/actioneval/build_index.py /dev/shm/actioneval/splits/windows /dev/shm/actioneval/beta/task_info \
        results/actioneval/action_index.json
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capeval"))
from catalog import DIR_ENTRIES, SKIP_VERBS, UNSEEN  # noqa: E402
from parse import parse  # noqa: E402


def obj_of(cl):
    return cl["patient"] or cl["tool"]


def main():
    wdir, ti_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    caps, cap_task = collections.Counter(), collections.defaultdict(set)
    for line in open(wdir / "train.jsonl"):
        r = json.loads(line)
        caps[r["caption"]] += 1
        cap_task[r["caption"]].add(r["task"])
    words = collections.Counter()
    verb, fam, tasks = collections.Counter(), {}, collections.defaultdict(set)
    v_obj, v_mat, v_dir, v_obj_dir = (collections.defaultdict(collections.Counter) for _ in range(4))
    obj_mat, example = {}, {}
    for c, n in caps.items():
        for w in re.findall(r"[a-z]+", c.lower()):
            words[w] += n
        for cl in parse(c)["clauses"]:
            if not cl.get("verb"):
                continue
            v, o = cl["verb"], obj_of(cl)
            verb[v] += n
            fam[v] = cl["family"]
            tasks[v] |= cap_task[c]
            example.setdefault(v, c)
            if o:
                v_obj[v][o["obj"]] += n
                v_mat[v][o["mat"]] += n
                obj_mat[o["obj"]] = o["mat"]
            for d in cl["dirs"]:
                v_dir[v][d] += n
                if o:
                    v_obj_dir[v][f"{o['obj']}|{d}"] += n

    # the full release: segments per verb (any clause), so the index covers every action AgiBot World Beta has
    rel = collections.Counter()
    parsed, n_seg = {}, 0
    for f in sorted(ti_dir.glob("task_*.json")):
        for e in json.loads(f.read_text()):
            for s in (e.get("label_info") or {}).get("action_config") or []:
                n_seg += 1
                cap = s["action_text"].strip()
                if cap not in parsed:
                    parsed[cap] = {cl["verb"] for cl in parse(cap)["clauses"] if cl.get("verb")}
                for v in parsed[cap]:
                    rel[v] += 1
                    fam.setdefault(v, None)

    seen = {}
    for v in sorted(set(verb) | set(rel), key=lambda v: -rel[v]):
        seen[v] = {
            "family": fam.get(v), "train_windows": verb[v], "train_tasks": len(tasks[v]), "release_segments": rel[v],
            "testable": v not in SKIP_VERBS and verb[v] > 0,
            "objects": dict(v_obj[v].most_common()), "materials": dict(v_mat[v].most_common()),
            "dirs": dict(v_dir[v].most_common()), "obj_dirs": dict(v_obj_dir[v].most_common()),
            "example": example.get(v),
        }
    unseen = {}
    for name, u in UNSEEN.items():
        hits = {w: words[w] for w in u["forms"] if words[w]}
        unseen[name] = {"family": u["family"], "train_word_hits": hits, "testable": not hits}
    dir_entries = [{"verb": v, "dir": d, "train_windows": seen.get(v, {}).get("dirs", {}).get(d, 0)} for v, d in DIR_ENTRIES]
    index = {
        "note": "Action index of AgiBot World Beta (Caption Atlas parser). seen: verb primitives with training counts; "
                "unseen: verbs no training caption contains in any form (raw-word check); dir_entries: verb x direction "
                "pairs trained (the seen action + direction probes).",
        "train": {"windows": sum(caps.values()), "captions": len(caps)},
        "release": {"captions": len(parsed), "segments": n_seg},
        "seen": seen, "unseen": unseen, "dir_entries": dir_entries,
        "obj_mat": obj_mat, "captions": dict(caps), "words": dict(words),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, ensure_ascii=False))
    ts = [v for v, r in seen.items() if r["testable"]]
    print(f"seen verbs {len(seen)} (testable {len(ts)}), unseen {sum(u['testable'] for u in unseen.values())}/{len(unseen)}, "
          f"verb x dir entries {sum(e['train_windows'] > 0 for e in dir_entries)}/{len(dir_entries)}")
    for name, u in unseen.items():
        if not u["testable"]:
            print("  unseen verb has training hits, dropped:", name, u["train_word_hits"])
    for e in dir_entries:
        if not e["train_windows"]:
            print("  verb x dir entry not in training:", e)


if __name__ == "__main__":
    main()
