"""Parse AgiBot captions into (verb primitive, family, directions, arm, patient / tool / target / source objects).

usage: python -I parse.py <windows_dir> <out_dir>
"""
import collections
import json
import re
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from lexicon import OBJECTS, VERBS  # noqa: E402

VERB_TOK = {}
for phrase, prim, fam in VERBS:
    VERB_TOK.setdefault(phrase.split()[0], []).append((phrase.split(), prim, fam))
for k in VERB_TOK:
    VERB_TOK[k].sort(key=lambda v: -len(v[0]))

OBJ_TOK = {}
for phrase, canon, cat, mat, flags in OBJECTS:
    toks = phrase.split()
    OBJ_TOK.setdefault(toks[0], []).append((toks, canon, cat, mat, flags))
for k in OBJ_TOK:
    OBJ_TOK[k].sort(key=lambda v: -len(v[0]))

EFFECTOR = {"arm", "arms", "hand", "hands"}
ROLE_PREP = {
    "into": "target", "onto": "target", "on": "target", "in": "target", "inside": "target", "under": "target",
    "at": "target", "to": "target", "towards": "target", "toward": "target", "over": "target", "above": "target",
    "beside": "target", "near": "target", "next": "target", "behind": "target", "below": "target", "beneath": "target",
    "from": "source", "off": "source", "out": "source",
    "with": "tool", "using": "tool",
    "containing": "contents", "filled": "contents", "contained": "contents",
}
DIRECTIONS = [
    ("left→right", r"from (the )?left to (the )?right"),
    ("right→left", r"from (the )?right to (the )?left"),
    ("back-and-forth", r"back and forth|twice|three times|\btimes\b"),
    ("leftward", r"\b(to|towards?|in the direction of) (the )?left\b(?! (arm|hand))|\bleftward"),
    ("rightward", r"\b(to|towards?|in the direction of) (the )?right\b(?! (arm|hand))|\brightward"),
    ("up", r"\b(up|upward|upwards)\b"),
    ("down", r"\b(down|downward|downwards)\b"),
    ("upright", r"\bupright\b"),
    ("forward", r"\bforward\b"),
    ("backward", r"\bbackwards?\b"),
    ("back (return)", r"\bback\b(?! and forth)"),
    ("in", r"\b(in|inward|inwards)\b(?= (with|using)\b|$)|\binwards?\b"),
    ("out", r"\bout\b(?! of)|\boutward"),
    ("aside / away", r"\baside\b|\baway\b"),
    ("closer", r"\bcloser\b"),
    ("over", r"\bover\b"),
    ("to the edge", r"\bto the (table )?edge\b|\bto the end\b"),
    ("to the center", r"\bto the (center|middle)\b"),
    ("horizontally", r"\bhorizontally\b"),
    ("open (result)", r"\b(open|opened)\b(?! (the|a)\b)"),
    ("closed (result)", r"\bclosed\b"),
]
DIRECTIONS = [(n, re.compile(p)) for n, p in DIRECTIONS]


def normalize(c):
    c = c.lower().strip().rstrip(".").replace(",", " ,")
    c = re.sub(r"\bpicked-up\b|\bpicked up\b", "picked", c)
    c = c.replace("pulled-out", "pulled").replace("t-shirts", "t-shirt").replace("short sleeve", "short-sleeve")
    c = re.sub(r"\bpant legs?\b", "pant-leg", c)
    c = re.sub(r"[^a-z0-9'/\- ,]", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    # subject-first forms: "left arm picks up X" -> "pick up X with left arm"
    c = re.sub(r"^(left|right) arm (picks|takes|holds|moves) (up )?(.*)$", r"\2 \3\4 with \1 arm", c)
    c = re.sub(r"^picks ", "pick ", c).replace("takes ", "take ")
    # "lift the left arm to pick up X" / "raise the right arm to ..." -> "pick up X"
    c = re.sub(r"^(lift|raise|lower) the (left|right) arm to ", "", c)
    # "lift the left arm with X" -> "lift X with left arm"
    c = re.sub(r"^(lift|grasp|grip|swing) the (left|right) arm (with|holding) (.*)$", r"\1 \4 with \2 arm", c)
    c = re.sub(r"^(cooperatively|coordinate arms to|use both (hands|arms) to) ", "", c)
    c = re.sub(r"\bthe light\b(?= (with|and|,)|$)", "the light-switch", c)
    c = re.sub(r"^the robot ", "", c).replace("holds ", "hold ").replace("moves ", "move ")
    c = re.sub(r"^wait for the person to turn , then ", "", c)
    return c


def match_verb(toks, i):
    for vt, prim, fam in VERB_TOK.get(toks[i], []):
        if toks[i:i + len(vt)] == vt:
            return len(vt), prim, fam
    return None


def hit_role(role):
    return role


def find_objects(toks):
    out, i, role = [], 0, "patient"
    while i < len(toks):
        t = toks[i]
        if t in ROLE_PREP and not (t in ("filled", "contained") and (i + 1 >= len(toks) or toks[i + 1] != "with")):
            role = ROLE_PREP[t]
            if t == "with" and i + 1 < len(toks) and toks[i + 1] in ("both", "left", "right", "the", "your"):
                nxt = toks[i + 1:i + 4]
                if any(x in EFFECTOR for x in nxt) and not any(x in OBJ_TOK and x not in ("left", "right") for x in nxt[1:2]):
                    role = "effector"
            if t == "held" or (t == "by" and role == "tool"):
                role = "effector"
        if t in ("held", "holding") and i + 1 < len(toks) and toks[i + 1] in ("by", "in"):
            role = "effector"
        hit = None
        for ot, canon, cat, mat, flags in OBJ_TOK.get(t, []):
            n = len(ot)
            if toks[i:i + n] == ot or (toks[i:i + n - 1] == ot[:-1] and i + n - 1 < len(toks)
                                        and toks[i + n - 1] in (ot[-1] + "s", ot[-1] + "es")):
                hit = (n, canon, cat, mat, flags)
                break
        if hit and out and out[-1].get("_end") == i and out[-1]["role"] == hit_role(role) and hit[2] != "person":
            out.pop()  # adjacent nouns: "fabric storage box", "paper cup" -> keep the head (last) noun
        if hit:
            n, canon, cat, mat, flags = hit
            if cat == "person" and canon == "body part" and not (i > 0 and toks[i - 1] == "person's"):
                i += n  # the acting limb ("with the right arm", "held in the left hand"), not an object
                continue
            if role != "effector":
                out.append({"role": role, "obj": canon, "cat": cat, "mat": mat, "flags": list(flags),
                            "span": " ".join(toks[i:i + n]), "_end": i + n})
            i += n
            continue
        i += 1
    return out


def arm_of(c):
    l = bool(re.search(r"\bleft (arm|hand)", c))
    r = bool(re.search(r"\bright (arm|hand)", c))
    both = bool(re.search(r"\bboth (arms|hands)|cooperativ|coordinat|\barms\b(?! of)", c))
    if both:
        return "both"
    if l and r:
        return "left+right (different roles)"
    if l:
        return "left"
    if r:
        return "right"
    return "unspecified"


SPLIT = re.compile(r" (?:, )?(?:and then|and|then|, then|while|to|until|after) (?=\S)")


def parse(caption):
    c = normalize(caption)
    arm = arm_of(c)
    # split into clauses at a conjunction / purpose "to" followed by a known verb
    pieces, last = [], 0
    for m in SPLIT.finditer(c):
        toks = c[m.end():].split()
        if toks and match_verb(toks, 0) and not c[m.end():].startswith(("the ", "a ")):
            pieces.append(c[last:m.start()])
            last = m.end()
    pieces.append(c[last:])
    clauses, prev_patient, prev_use_tool = [], None, None
    for p in pieces:
        toks = [t for t in p.split() if t != ","]
        if not toks:
            continue
        # "use the held X (in the right arm) to VERB ..." handled by the split; here X is the tool of the next clause
        mv = match_verb(toks, 0)
        if not mv:
            clauses.append({"verb": None, "family": "unparsed", "text": p})
            continue
        n, prim, fam = mv
        rest_toks = toks[n:]
        rest = " ".join(rest_toks)
        # locomotion: turning / moving the base
        if prim in ("turn", "rotate", "move", "carry") and re.search(r"\bchassis\b", rest) or \
                (prim in ("turn", "move", "rotate") and (not rest or re.match(
                    r"(towards?|toward|forward|backwards?|back|slightly|a distance|a bit|away|closer|to |left|right|"
                    r"until|around|\d)", rest))):
            prim, fam = ("turn base" if prim in ("turn", "rotate") else "move base"), "locomotion"
        if re.match(r"(the |your |the other )?(left |right )?(hand|arm)s?\b", rest) and fam not in ("tool_use",) \
                and not re.match(r"(the )?(left |right )?arm (with|holding)", rest):
            prim, fam = "move arm (no object)", "gesture"  # "pull the hand back", "place the right arm back"
        if prim == "turn" and re.search(r"^(on|off)\b|\b(on|off)$|\b(on|off) (with|using)\b", rest) and fam == "rotate":
            prim = "turn on/off"
        if prim == "press" and re.search(r"\bwith the (held )?steam iron\b|\biron\b", rest):
            prim, fam = "iron", "tool_use"
        dirs = [name for name, rx in DIRECTIONS if rx.search(rest)]
        if prim in ("pull open", "push open", "twist open"):
            dirs.append("open (result)")
        if prim in ("pull out", "take out"):
            dirs.append("out")
        if prim == "pull down":
            dirs.append("down")
        if prim == "lift":
            dirs.append("up")
        if prim == "lower":
            dirs.append("down")
        objs = find_objects(rest_toks)
        if prim in ("use", "aim") and objs:
            objs[0]["role"] = "tool"
        patient = next((o for o in objs if o["role"] == "patient"), None)
        if patient is None and re.match(r"(it|them|they)\b", rest) and prev_patient:
            patient = dict(prev_patient, role="patient", span="it")
            objs.insert(0, patient)
        tool = next((o for o in objs if o["role"] == "tool"), None)
        if fam == "clean" and not tool and clauses and clauses[-1].get("patient") and \
                clauses[-1]["family"] == "grasp" and patient and patient["mat"] == "fixture":
            tool = dict(clauses[-1]["patient"], role="tool")  # "grasp the cloth and wipe the stain"
        if prev_use_tool and not tool:
            tool = dict(prev_use_tool, role="tool")
        prev_use_tool = tool if prim == "use" else None
        if patient is None and tool is None and fam not in ("locomotion", "gesture", "wait") and prim != "move arm (no object)" and clauses \
                and clauses[-1].get("verb"):
            patient, tool = clauses[-1]["patient"], clauses[-1]["tool"]  # "... to scan", "... to swipe": implicit object
        target = next((o for o in objs if o["role"] == "target"), None)
        source = next((o for o in objs if o["role"] == "source"), None)
        # a liquid / powder that is only carried around is really its container ("place the held salad dressing")
        if patient and patient["mat"] in ("liquid", "granular") and fam not in ("pour", "mix", "clean", "cut_shape"):
            patient = dict(patient, obj=patient["obj"] + " (in container)", mat="rigid", substance=patient["mat"],
                           flags=sorted(set(patient["flags"]) | {"container", "contents"}))
        # what is in the gripper / under the hand
        contact = tool or patient
        if patient and patient["mat"] in ("liquid", "granular") and not tool:
            contact = source if source and source["mat"] not in ("fixture",) else None
        if patient:
            prev_patient = patient
        clauses.append({"verb": prim, "family": fam, "dirs": sorted(set(dirs)), "patient": patient, "tool": tool,
                        "target": target, "source": source, "contact": contact, "text": p})
    return {"caption": caption, "norm": c, "arm": arm, "clauses": clauses}


def main_clause(parsed):
    cl = [x for x in parsed["clauses"] if x.get("verb")]
    if not cl:
        return None
    goal = [x for x in cl if x["family"] not in ("grasp", "wait", "other", "locomotion")]
    return (goal or cl)[-1]


if __name__ == "__main__":
    wdir, out = sys.argv[1], sys.argv[2]
    res = {}
    for split in ("train", "val", "test", "test_unseen"):
        cnt = collections.Counter()
        tasks = collections.defaultdict(collections.Counter)
        eps = collections.defaultdict(set)
        for line in open(f"{wdir}/{split}.jsonl"):
            r = json.loads(line)
            cnt[r["caption"]] += 1
            tasks[r["caption"]][r["task_name"]] += 1
            eps[r["caption"]].add(r["episode"])
        res[split] = [dict(parse(c), windows=n, episodes=len(eps[c]), tasks=dict(tasks[c])) for c, n in cnt.most_common()]
    json.dump(res, open(f"{out}/parsed.json", "w"))
    tr = res["train"]
    tot = sum(x["windows"] for x in tr)
    unp = [x for x in tr if not main_clause(x)]
    nopat = [x for x in tr if main_clause(x) and not (main_clause(x)["patient"] or main_clause(x)["tool"])]
    print("train windows", tot, "| no verb:", sum(x["windows"] for x in unp), len(unp),
          "| verb but no object:", sum(x["windows"] for x in nopat), len(nopat))
    print("--- no verb"); [print(x["windows"], x["caption"]) for x in unp[:40]]
    print("--- no object"); [print(x["windows"], x["caption"], "=>", main_clause(x)["verb"]) for x in nopat[:80]]
