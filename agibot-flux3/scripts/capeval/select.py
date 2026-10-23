"""Candidate eval segments per action slot: never-trained (episode in no split), 120-270 source frames (4-9 s at
30 fps; one rollout covers 266), one candidate per episode and task, seeded. usage: python -I select.py segs.json out.json"""
import collections, json, random, re, sys
R = json.load(open(sys.argv[1]))
SLOTS = {  # family: [(slot, regex on caption)]
    "pour": [("pour", r"^pour "), ("scoop", r"^(scoop|squeeze|sprinkle) ")],
    "rotate": [("twist", r"^(twist|unscrew|tighten)"), ("turn", r"^turn (on|off) ")],
    "tool_use": [("scan", r"^(aim the barcode|scan the (item|barcode|goods|code))"), ("iron_stamp", r"^(stamp|iron |press the .* with the steam iron)")],
    "press": [("press_button", r"^press .*(button|switch|key)"), ("press_other", r"^press (?!.*(button|switch|key|iron))")],
    "locomotion": [("turn_base", r"^turn (towards|to|around|left|right)"), ("move_base", r"^(move|advance|step)")],
    "transport": [("carry", r"^carry "), ("move_obj", r"^move the (?!chassis|base)")],
    "gesture": [("clap", r"clap"), ("wave", r"waving|wave goodbye")],
    "cut_shape": [("cut", r"^cut "), ("peel_knead_roll", r"^(peel|knead|roll the dough)")],
    "mix": [("stir", r"^stir "), ("whisk_shake", r"^(whisk|whip|shake) ")],
    "tie": [("tie", r"^tie "), ("untie", r"untie")],
}
MAXLEN = {"wave": 300}  # waving segments are 8-12 s; the rollout covers 8.9 s
rng = random.Random(42)
out = {}
for fam, slots in SLOTS.items():
    for slot, rx in slots:
        pool = [x for x in R if x["split"] == "unsampled" and not x["held_out_task"] and 120 <= x["len"] <= MAXLEN.get(slot, 270)
                and re.search(rx, x["caption"].lower())]
        rng.shuffle(pool)
        seen_t, seen_e, cands = collections.Counter(), set(), []
        for x in pool:
            if x["episode"] in seen_e:
                continue
            seen_e.add(x["episode"]); seen_t[x["task"]] += 1; cands.append(x)
        out[f"{fam}/{slot}"] = cands
        print(f"{fam}/{slot}: pool {len(pool)}")
        for c in cands[:6]:
            print(f"   {c['task']} {c['episode']} s{c['seg']} {c['len']:3d}f train_cap={c['cap_in_train']:4d} | {c['caption'][:90]}")
# held-out tasks: never-trained (test_unseen episodes are never trained; unsampled ones were never even fetched)
for t in sorted({x["task"] for x in R if x["held_out_task"]}):
    pool = [x for x in R if x["task"] == t and x["split"] == "test_unseen" and 120 <= x["len"] <= 270]
    rng.shuffle(pool)
    out[f"heldout/{t}"] = pool
    print(f"heldout/{t} {pool[0]['name'] if pool else ''}: pool {len(pool)}")
    for c in pool[:6]:
        print(f"   {c['episode']} s{c['seg']} {c['len']:3d}f train_cap={c['cap_in_train']:4d} | {c['caption'][:90]}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
