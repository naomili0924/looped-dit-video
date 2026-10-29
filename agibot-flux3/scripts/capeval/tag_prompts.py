"""Training-exposure ladder for capability-eval prompts: how close did training come to each prompt's action?

Each prompt's main clause (parse.py) is matched against the parsed training captions, counted in training windows:
  L1 caption   the exact caption occurs in training (recall, not transfer)
  L2 verb+obj  the same verb primitive on the same canonical object occurs (easy transfer)
  L3 verb      the verb occurs, only on other objects (fair transfer test: a seen action on a new object)
  L4 unseen    the verb never occurs (the model is asked to imagine an action: report separately)
For directional verbs the verb × direction count is reported too ("turn base" + leftward), and "in task" says whether the
verb occurs in the case's own training task (same scene, a trained alternative action).

    python -I scripts/capeval/tag_prompts.py <train.jsonl> configs/eval_cases_capability.json results/capeval/prompt_ladder.json
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse import main_clause, parse  # noqa: E402

LEVELS = ("L1 caption", "L2 verb+obj", "L3 verb", "L4 unseen")
# Parser misses, checked by hand against the raw training captions (case, prompt) -> (level, evidence)
OVERRIDES = {
    ("move_guard", "alt2"): ("L2 verb+obj", "'hand guard' is the object, not the robot's hand: 'Move the held hand guard above "
                             "the vegetable cutter' (46 w, own task); only the destination is new"),
    ("base_turn", "alt2"): ("L2 verb+obj", "turning left is in the own task: 'Rotate towards left and move the curtain' (49 w)"),
    ("tie_tieback", "alt2"): ("L2 verb+obj", "'Cooperatively untie the curtain ties with both hands' (72 w, own task)"),
    ("clap", "action"): ("L3 verb", "the lexicon has no 'waving': 'Lower the left arm while waving the right arm' (121 w, "
                         "task Wave goodbye); waving at a TV is new"),
    ("clap", "alt2"): ("L3 verb", "the lexicon has no 'waving': 'Lower the left arm while waving the right arm' (121 w, "
                       "task Wave goodbye); waving at a TV is new"),
}


def obj_name(cl):
    o = cl["patient"] or cl["tool"]
    return o["obj"] if o else None


def main():
    train_p, cases_p, out_p = sys.argv[1:4]
    caps = collections.Counter()
    cap_task = collections.defaultdict(set)
    for line in open(train_p):
        r = json.loads(line)
        caps[r["caption"]] += 1
        cap_task[r["caption"]].add(r["task"])
    verb, verb_obj, verb_dir = collections.Counter(), collections.Counter(), collections.Counter()
    verb_tasks, verb_objs, verb_ex = collections.defaultdict(set), collections.defaultdict(collections.Counter), {}
    for c, n in caps.items():
        for cl in parse(c)["clauses"]:
            if not cl.get("verb"):
                continue
            v, o = cl["verb"], obj_name(cl)
            verb[v] += n
            verb_tasks[v] |= cap_task[c]
            verb_objs[v][o] += n
            verb_obj[(v, o)] += n
            for d in cl["dirs"]:
                verb_dir[(v, d)] += n
            verb_ex.setdefault(v, c)
    cases = json.load(open(cases_p))["cases"]
    rows = []
    for case in cases:
        for pk, text in case["prompts"].items():
            cl = main_clause(parse(text))
            v, o = (cl["verb"], obj_name(cl)) if cl else (None, None)
            dirs = cl["dirs"] if cl else []
            level = (LEVELS[0] if caps[text] else LEVELS[1] if verb_obj[(v, o)] and o else
                     LEVELS[2] if verb[v] else LEVELS[3])
            rows.append({
                "case": case["id"], "prompt": pk, "text": text, "task": case["task"], "verb": v, "object": o, "dirs": dirs,
                "level": level, "caption_windows": caps[text], "verb_windows": verb[v], "verb_obj_windows": verb_obj[(v, o)],
                "verb_dir_windows": {d: verb_dir[(v, d)] for d in dirs}, "verb_tasks": len(verb_tasks[v]),
                "verb_in_own_task": case["task"] in verb_tasks[v],
                "verb_top_objects": [[k, n] for k, n in verb_objs[v].most_common(4)], "verb_example": verb_ex.get(v),
            })
            if (case["id"], pk) in OVERRIDES:
                rows[-1]["parser_level"] = level
                rows[-1]["level"], rows[-1]["override"] = OVERRIDES[(case["id"], pk)]
    Path(out_p).write_text(json.dumps(rows, indent=1))
    for r in rows:
        if r["prompt"] in ("real", "alt1"):
            continue
        dd = " ".join(f"{d}:{n}" for d, n in r["verb_dir_windows"].items())
        print(f"{r['case']:20s} {r['level']:12s} verb={r['verb']!s:14s} obj={r['object']!s:24s} verb {r['verb_windows']:6d}w "
              f"/{r['verb_tasks']:3d}t  verb+obj {r['verb_obj_windows']:5d}w  own-task {'Y' if r['verb_in_own_task'] else '-'}  {dd}")
        print(f"{'':20s} \"{r['text']}\"  | e.g. \"{r['verb_example']}\"  | objs {r['verb_top_objects'][:3]}")


if __name__ == "__main__":
    main()
