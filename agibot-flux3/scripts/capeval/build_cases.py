"""Write configs/eval_cases_capability.json: the chosen never-trained segments + real/alt1/alt2 prompts, with a leak check.
usage: python -I build_cases.py segs.json <data_splits_dir> <out.json>"""
import json, sys
from pathlib import Path
# (id, family, episode, seg, alt1, alt2, alt2_family). real = AgiBot's caption of the segment, verbatim.
# alt1: same action, plain wording, no arm named. alt2: a different action on the same scene (no ground truth).
CASES = [
 ("pour_dressing", "pour", "728347", None, "Pour the salad dressing over the slice of meat.", "Squeeze the salad dressing bottle over the slice of meat.", "pour"),
 ("scoop_tapioca", "pour", "791259", None, "Use the spoon to scoop up some black tapioca pearls from the container.", "Pour the tapioca pearls from the spoon back into the container.", "pour"),
 ("twist_cap", "rotate", "899308", None, "Screw the cap tightly onto the water bottle.", "Unscrew the cap and take it off the water bottle.", "rotate"),
 ("turn_faucet", "rotate", "681627", None, "Rotate the faucet handle to shut off the water.", "Turn on the faucet.", "rotate"),
 ("scan_juice", "tool_use", "706508", None, "Scan the barcode on the orange juice bottle with the scanner.", "Scan the yellow bottle on the counter with the barcode scanner.", "tool_use"),
 ("stamp_form", "tool_use", "900248", None, "Press the stamp down onto the paper to stamp it.", "Stamp the paper twice, in two different places.", "tool_use"),
 ("press_oven", "press", "677826", None, "Push the start button on the oven.", "Press the oven start button twice.", "press"),
 ("press_ricecooker", "press", "810754", None, "Press down on the lid of the rice cooker to close it.", "Press the button on the front of the rice cooker.", "press"),
 ("base_forward", "locomotion", "886770", None, "Drive forward toward the person sitting at the table.", "Move backward away from the table.", "locomotion"),
 ("base_turn", "locomotion", "864794", None, "Rotate the robot's body to face the curtains.", "Turn to the left, away from the curtains.", "locomotion"),
 ("carry_bag", "transport", "876183", None, "Carry the bag over to the trash bin.", "Carry the bag toward the wooden table.", "transport"),
 ("move_guard", "transport", "840253", None, "Bring the hand guard over the vegetable slicer.", "Move the hand guard to the side of the counter, away from the slicer.", "transport"),
 ("clap", "gesture", "870019", None, "Bring both hands together to clap.", "Wave one hand at the TV.", "gesture"),
 ("wave", "gesture", "885609", None, "Wave goodbye to the person.", "Clap hands.", "gesture"),
 ("cut_dough", "cut_shape", "901777", None, "Slice the dough into small pieces with the knife.", "Cut the dough into two halves.", "cut_shape"),
 ("knead_dough", "cut_shape", "911223", None, "Squeeze and press the dough ball with both hands.", "Roll the dough into a long rope.", "cut_shape"),
 ("stir_coffee", "mix", "722051", None, "Stir the drink in the cup with the spoon.", "Shake the cup gently to mix the coffee.", "mix"),
 ("shake_decanter", "mix", "901598", None, "Swirl the wine around in the decanter.", "Pour the wine from the decanter into the glass.", "pour"),
 ("untie_curtain", "tie", "858022", None, "Undo the tie holding the curtain together.", "Tie the curtain tie tighter.", "tie"),
 ("tie_tieback", "tie", "886835", 0, "Wrap the tieback around the curtain and fasten the velcro.", "Untie the curtain tieback and let the curtain hang.", "tie"),
 ("heldout_toast", "heldout", "688266", None, "Pick up a slice of bread from the basket.", "Put a slice of bread into the toaster.", "place"),
 ("heldout_supermarket", "heldout", "655512", None, "Take a small milk bottle out of the fridge shelf.", "Put the milk bottle into the shopping cart.", "place"),
 ("heldout_water", "heldout", "716307", None, "Take one water bottle out of the carton.", "Place a water bottle upright on the table next to the carton.", "place"),
 ("heldout_curtains", "heldout", "861620", None, "Pull the curtain down to the windowsill.", "Pull the curtain back up to open it.", "push_pull"),
 ("heldout_menu", "heldout", "889403", None, "Bring the menu over to the guest at the table.", "Hand the menu to the guest.", "handover"),
]
segs, ds, out = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3])
picked = json.load(open("/dev/shm/capeval/mirror/picked.json")) | json.load(open("/dev/shm/capeval/mirror/picked_tie.json"))
by_ep = {c["episode"]: c for cs in picked.values() for c in cs}
trained = {json.loads(l)["episode"] for s in ("train", "val", "test") for l in open(ds / "windows" / f"{s}.jsonl")}
train_caps = {json.loads(l)["caption"] for l in open(ds / "windows/train.jsonl")}
cases = []
for cid, fam, ep, _, a1, a2, a2f in CASES:
    c = by_ep[ep]
    assert ep not in trained, f"{cid}: episode {ep} is in train/val/test"
    assert c["split"] in ("unsampled", "test_unseen"), c["split"]
    cases.append({
        "id": cid, "family": fam, "source": "real", "split": c["split"], "task": c["task"], "task_name": c["name"],
        "episode": ep, "seg": c["seg"], "anchor": c["start"] + 2, "segment_frames": c["len"],
        "caption": c["caption"], "prompts": {"real": c["caption"], "alt1": a1, "alt2": a2}, "alt2_family": a2f,
        "provenance": {"episode_in_any_split": False, "held_out_task": c["held_out_task"],
                       "real_caption_in_training": c["caption"] in train_caps, "alt1_in_training": a1 in train_caps,
                       "alt2_in_training": a2 in train_caps, "skill": c["skill"]},
    })
spec = {"note": "Capability eval cases. Every case is a REAL AgiBot World Beta episode that training never saw: either in "
        "no split of Jinyan0924/agibot-flux3-eval-splits (never sampled, never fetched; split 'unsampled') or a test_unseen "
        "episode of one of the 5 held-out tasks. anchor = segment start + 2, so the k = 1 context frame is the segment's first "
        "frame. real = AgiBot's annotation (verbatim, may name an arm); alt1 = same action reworded, no arm; alt2 = a different "
        "action in the same scene (no ground truth). Prompts written by Claude (LLM) from the segment's frames. A case with "
        "\"source\": \"generated\" would carry an edited context_image instead (none so far).",
        "families": {"pour": "Pour / scoop / squeeze", "rotate": "Turn / twist / rotate", "tool_use": "Tool use (scan, stamp, iron, …)",
                     "press": "Press / click", "locomotion": "Base: turn / move / face", "transport": "Carry / move",
                     "gesture": "Gesture (wave, clap)", "cut_shape": "Cut / peel / knead / roll", "mix": "Stir / whisk / shake",
                     "tie": "Tie / untie", "heldout": "Held-out task (test_unseen)"},
        "cases": cases}
out.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
print(len(cases), "cases ->", out)
for c in cases:
    print(f"{c['id']:20s} {c['split']:11s} {c['segment_frames']:3d}f cap_in_train={c['provenance']['real_caption_in_training']!s:5s} | {c['caption'][:70]}")
