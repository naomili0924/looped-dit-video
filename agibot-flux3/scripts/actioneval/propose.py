"""Candidate probes per initial window: every action of the index on every scene object it can act on, labelled with
how far training was from it. The VLM then checks feasibility (ldvflux.act_vlm --mode feasible) and select.py picks.

Scene objects = the episode's caption objects the VLM saw in the frame + the other objects it listed
(ldvflux.act_vlm --mode scene), mapped to the lexicon's canonical objects where the parser knows the noun.

Groups (the question each probe answers):
  seen_new_object   a trained verb on an object it was never trained on (verb x object = 0 training windows)
  seen_dir_new_obj  a trained verb x direction pair on an object it never had with that direction
  unseen_action     a verb no training caption contains in any form (reported separately: not expected to work)
  control           the episode's own caption (AgiBot's annotation; the real footage shows it)
Levels (Caption Atlas ladder): L0 exact caption, L1 seen verb + object (+ direction), L3 new direction for this verb +
object, L4 new verb-object pair (verb seen on this material), L5 verb never on this material, L6 verb never in training.

    python -I scripts/actioneval/propose.py results/actioneval   (reads action_index.json, pool.json, scene_*.json)
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capeval"))
from catalog import BASE_DIR_TEXT, DIR_ENTRIES, DIR_PHRASE, DIR_TEMPLATES, SEEN, TARGET_CATS, UNSEEN  # noqa: E402
from parse import main_clause, parse  # noqa: E402

MAT_CLASSES = {"rigid", "articulated", "cloth", "thin_flexible", "elastic", "plastic", "linear", "liquid", "granular",
               "divisible", "fixture"}
PER_ENTRY = 2  # candidate objects per window and action (the VLM keeps the feasible ones; select.py picks one)
# VLM-listed things that are not objects to act on: the robot itself, markings, background, vague names
NOT_OBJECT = re.compile(r"\b(robot|arm|arms|gripper|hand|hands|finger|camera|label|logo|sticker|text|shadow|reflection|"
                        r"floor|wall|ceiling|window|light|object|item|thing|stuff|background)\b")


def obj_of(cl):
    return cl and (cl["patient"] or cl["tool"])


def canonical(name: str):
    """A free-text object name -> the lexicon's object (via the parser on 'pick up the <name>') or None."""
    o = obj_of(main_clause(parse(f"pick up the {name}")))
    return o if o and o["span"] in name.lower() else None


def scene(w: dict, vlm: dict | None) -> list[dict]:
    vis = (vlm or {}).get("visible", {})
    objs, have = [], set()
    for o in w["objects"]:  # caption objects, unless the VLM says they are not in view
        if vis.get(o["span"]) is False:
            continue
        if o["obj"] == "item (generic)":
            continue
        objs.append({**o, "from": "caption"})
        have.add(o["obj"])
    for x in (vlm or {}).get("objects", []):
        name, mat = str(x.get("name", "")).strip().lower(), str(x.get("material", "")).strip().lower()
        if not name or len(name) > 40 or NOT_OBJECT.search(name):
            continue
        c = canonical(name)
        if c:
            if c["obj"] in have or c["cat"] == "person" or c["obj"] == "item (generic)":
                continue
            objs.append({"obj": c["obj"], "span": name, "cat": c["cat"], "mat": c["mat"], "flags": c["flags"], "from": "vlm"})
            have.add(c["obj"])
        elif mat in MAT_CLASSES and f"new:{name}" not in have:  # a noun training captions never use: a new object for every verb
            objs.append({"obj": f"new:{name}", "span": name, "cat": "surface" if mat == "fixture" else None, "mat": mat,
                         "flags": [], "from": "vlm"})
            have.add(f"new:{name}")
    return objs


def targets(objs, o, kind):
    """Scene objects that can be the {t} of a template: a container for into/onto, else any surface or furniture."""
    if kind == "into":  # a box, basket, bowl, cup, pot, bin: not a packaged good that merely has contents
        pool = [t for t in objs if t is not o and t["cat"] in ("container", "tableware")]
    else:  # a surface first (table, counter, shelf), then furniture / a container
        pool = sorted((t for t in objs if t is not o and (t["cat"] in TARGET_CATS or t["mat"] == "fixture")),
                      key=lambda t: (t["mat"] != "fixture" and t["cat"] not in ("surface", "fixture")))
    return pool[0] if pool else None


def allowed(o, mats, fixture):
    if o["mat"] == "fixture" or o["cat"] in ("surface", "fixture"):
        return fixture
    return o["mat"] in mats


def label(idx, text, verb, obj, mat, dirs):
    """Training exposure of a prompt whose main clause is verb (+ dirs) on obj (canonical, 'new:...' or None)."""
    s = idx["seen"].get(verb)
    caption_w = idx["captions"].get(text, 0)
    if not s or not s["train_windows"]:
        return {"level": "L6 verb never in training", "caption_windows": 0, "verb_windows": 0}
    pair = s["objects"].get(obj, 0) if obj else None
    vmat = s["materials"].get(mat, 0) if mat else None
    vdir = {d: s["dirs"].get(d, 0) for d in dirs}
    odir = {d: s["obj_dirs"].get(f"{obj}|{d}", 0) for d in dirs} if obj else {}
    if caption_w:
        lv = "L0 exact caption"
    elif obj is None:
        lv = "L1 seen verb + direction, new scene" if dirs else "L1 seen verb, no object"
    elif pair and (not dirs or all(odir.values())):
        lv = "L1 seen verb + object"
    elif pair:
        lv = "L3 new direction for this verb + object"
    elif vmat:
        lv = "L4 new object, verb seen on this material"
    else:
        lv = "L5 verb never on this material"
    return {"level": lv, "caption_windows": caption_w, "verb_windows": s["train_windows"], "verb_tasks": s["train_tasks"],
            "verb_obj_windows": pair, "verb_mat_windows": vmat, "verb_dir_windows": vdir, "verb_obj_dir_windows": odir}


def fill(tmpl, o, objs, d=None):
    t = None
    if "{t}" in tmpl:
        t = targets(objs, o, "into" if any(w in tmpl for w in (" into ", " over ")) else "on")
        if t is None:
            return None, None
    return tmpl.format(o=o["span"] if o else "", t=t["span"] if t else "", d=DIR_PHRASE.get(d, "")), t


def main():
    root = Path(sys.argv[1])
    idx = json.loads((root / "action_index.json").read_text())
    windows = json.loads((root / "pool.json").read_text())["windows"]
    vlm = {}
    for p in sorted(root.glob("scene_*.json")):
        vlm |= json.loads(p.read_text())
    rng = random.Random(0)
    out, n_cand = [], 0
    for w in windows:
        if w["id"] not in vlm:
            continue
        objs = scene(w, vlm[w["id"]])
        cands = []

        def add(group, entry, verb, family, text, o, t, dirs, lab):
            cands.append({"group": group, "entry": entry, "verb": verb, "family": family, "prompt": text, "dirs": dirs,
                          "object": o and o["obj"], "span": o and o["span"], "material": o and o["mat"],
                          "object_from": o and o["from"], "target": t and t["span"], **lab})

        for verb, tm in SEEN.items():  # a trained verb on a new object
            mats = tm["mats"] | {m for m in idx["seen"][verb]["materials"] if m not in ("fixture", "person")}
            got = []
            for o in rng.sample(objs, len(objs)):
                if not allowed(o, mats, tm["fixture"]):
                    continue
                text, t = fill(tm["t"], o, objs)
                if not text:
                    continue
                cl = main_clause(parse(text))
                po = obj_of(cl)
                if not cl or cl["verb"] != verb:
                    continue
                obj = o["obj"] if o["obj"].startswith("new:") else po and po["obj"]
                if obj != o["obj"]:
                    continue  # the parser read another object as the patient: the label would be wrong
                lab = label(idx, text, verb, obj, o["mat"], [])
                if lab["level"].startswith(("L4", "L5")):
                    got.append((text, o, t, lab))
            for text, o, t, lab in got[:PER_ENTRY]:
                add("seen_new_object", verb, verb, idx["seen"][verb]["family"], text, o, t, [], lab)
        for verb, d in DIR_ENTRIES:  # a trained verb x direction on an object it never had with that direction
            tm = DIR_TEMPLATES[verb]
            tmpl = BASE_DIR_TEXT.get((verb, d), tm["t"])
            got = []
            for o in ([None] if "{o}" not in tmpl else rng.sample(objs, len(objs))):
                if o is not None and not allowed(o, tm["mats"], tm["fixture"]):
                    continue
                text, t = fill(tmpl, o, objs, d)
                if not text:
                    continue
                cl = main_clause(parse(text))
                if not cl or cl["verb"] != verb or d not in cl["dirs"]:
                    continue
                po = obj_of(cl)
                obj = (o["obj"] if o and o["obj"].startswith("new:") else po and po["obj"]) if verb not in ("move base", "turn base") else None
                lab = label(idx, text, verb, obj, o and o["mat"], [d])
                if lab["level"].startswith(("L1 seen verb + direction", "L3", "L4", "L5")):
                    got.append((text, o, t, lab))
            for text, o, t, lab in got[:PER_ENTRY]:
                add("seen_dir_new_obj", f"{verb} | {d}", verb, idx["seen"][verb]["family"], text, o, t, [d], lab)
        for verb, u in UNSEEN.items():  # verbs no training caption contains
            if not idx["unseen"][verb]["testable"]:
                continue
            got = [o for o in rng.sample(objs, len(objs)) if allowed(o, u["mats"], False)][:PER_ENTRY]
            for o in got:
                text, t = fill(u["t"], o, objs)
                add("unseen_action", verb, verb, u["family"], text, o, t, [],
                    {"level": "L6 verb never in training", "caption_windows": 0, "verb_windows": 0})
        cl = main_clause(parse(w["caption"]))  # control: the episode's own caption
        po = obj_of(cl)
        ctrl = {"group": "control", "entry": "control", "verb": cl and cl["verb"], "family": cl and cl["family"],
                "prompt": w["caption"], "dirs": cl["dirs"] if cl else [], "object": po and po["obj"], "span": po and po["span"],
                "material": po and po["mat"], **label(idx, w["caption"], cl and cl["verb"], po and po["obj"], po and po["mat"],
                                                      cl["dirs"] if cl else [])}
        out.append({**w, "scene": objs, "control": ctrl, "candidates": cands})
        n_cand += len(cands)
    (root / "candidates.json").write_text(json.dumps({"windows": out}, indent=1, ensure_ascii=False))
    entries = {c["entry"] for w in out for c in w["candidates"]}
    print(f"{len(out)} windows, {n_cand} candidates ({n_cand / max(1, len(out)):.0f} per window), {len(entries)} actions with a candidate")


if __name__ == "__main__":
    main()
