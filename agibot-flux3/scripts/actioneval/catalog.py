"""Action catalog of the action eval: one prompt template per action, which materials it can act on, and the unseen
actions (verbs no training caption contains).

Template fields: {o} the acted-on object (its span in the episode's own captions), {t} a second scene object used as
the target / place (a container, surface or piece of furniture), {d} a direction phrase (DIR_PHRASE). Prompts name no
arm. Materials are the lexicon's classes (scripts/capeval/lexicon.py); an action is proposed on an object whose
material is in the template's set or in the materials training used the verb on (so "squeeze" goes from the trained
ketchup bottle to other elastic, thin or soft objects, and a rigid bottle is the "verb never on this material" level).
"""

SOLID = "rigid elastic thin_flexible cloth divisible plastic linear"
SOFT = "elastic thin_flexible cloth plastic divisible"
FLOWS = "liquid granular"
TARGET_CATS = {"container", "tableware", "surface", "furniture", "appliance", "fixture"}


def T(text, mats="", target=False, fixture=False):
    return {"t": text, "mats": set(mats.split()), "target": target, "fixture": fixture}


# Verb primitives of the Caption Atlas that are not a self-contained instruction from a neutral start frame
# (no object, a generic verb, or the end of another action); they stay in the index but get no probe.
SKIP_VERBS = {"coordinate", "use", "move arm (no object)", "release", "adjust", "tidy", "align", "leave", "receive",
              "move base", "turn base", "step"}  # the base verbs are probed with directions (DIR_ENTRIES)

# seen actions (training windows > 0): "seen action on a new object" probes
SEEN = {
    # grasp
    "pick up": T("Pick up the {o}.", SOLID), "grasp": T("Grasp the {o}.", SOLID), "grab": T("Grab the {o}.", SOLID),
    "take": T("Take the {o}.", SOLID), "retrieve": T("Retrieve the {o}.", SOLID), "grip": T("Grip the {o}.", SOLID + " articulated"),
    "hold": T("Hold the {o}.", SOLID), "clamp": T("Clamp the {o} between both hands.", SOLID),
    "remove": T("Remove the {o} from the {t}.", SOLID, target=True),
    # lift / lower
    "lift": T("Lift the {o}.", SOLID), "lower": T("Lower the {o}.", SOLID),
    # place / insert / hang / throw
    "place": T("Place the {o} on the {t}.", SOLID, target=True), "hang": T("Hang the {o} on the {t}.", SOLID, target=True),
    "insert": T("Insert the {o} into the {t}.", SOLID, target=True), "throw": T("Throw the {o} into the {t}.", SOLID, target=True),
    "dump": T("Dump the {o} into the {t}.", SOLID + " " + FLOWS, target=True), "stack": T("Stack the {o} on the {t}.", "rigid", target=True),
    "pack": T("Pack the {o} into the {t}.", SOLID, target=True), "unplug": T("Unplug the {o}.", "rigid"),
    # transport / handover
    "carry": T("Carry the {o} to the {t}.", SOLID, target=True), "move": T("Move the {o} to the {t}.", SOLID, target=True),
    "bring": T("Bring the {o} to the {t}.", SOLID, target=True), "pass": T("Pass the {o} from one hand to the other.", SOLID),
    "hand over": T("Hand over the {o}.", SOLID),
    # push / pull / slide
    "push": T("Push the {o}.", SOLID + " articulated"), "pull": T("Pull the {o}.", SOLID + " articulated"),
    "slide": T("Slide the {o} across the table.", SOLID), "pull out": T("Pull out the {o} from the {t}.", SOLID, target=True),
    "pull down": T("Pull down the {o}.", SOLID + " articulated"), "draw": T("Draw the {o}.", "cloth thin_flexible"),
    "swipe": T("Swipe the {o}.", "thin_flexible rigid"),
    # open / close / cover
    "open": T("Open the {o}.", "articulated rigid thin_flexible"), "close": T("Close the {o}.", "articulated rigid thin_flexible"),
    "cover": T("Cover the {o}.", "articulated rigid"), "seal": T("Seal the {o}.", "rigid thin_flexible"),
    "pull open": T("Pull open the {o}.", "articulated rigid"), "push open": T("Push open the {o}.", "articulated rigid"),
    "unlock": T("Unlock the {o}.", "articulated"),
    # turn / twist
    "turn on": T("Turn on the {o}.", "articulated rigid"), "turn off": T("Turn off the {o}.", "articulated rigid"),
    "turn": T("Turn the {o}.", "articulated rigid"), "twist off": T("Twist off the {o}.", "rigid articulated"),
    "twist open": T("Twist open the {o}.", "rigid articulated"), "tighten": T("Tighten the {o}.", "rigid articulated"),
    "unscrew": T("Unscrew the {o}.", "rigid articulated"), "twist": T("Twist the {o}.", "rigid articulated elastic"),
    "flip": T("Flip the {o} over.", SOLID),
    # press
    "press": T("Press the {o}.", "articulated rigid elastic plastic"),
    # pour / scoop / squeeze
    "pour": T("Pour the {o} into the {t}.", FLOWS, target=True), "scoop": T("Scoop up some {o}.", FLOWS + " divisible"),
    "sprinkle": T("Sprinkle the {o} over the {t}.", "granular", target=True), "empty": T("Empty the {o} into the {t}.", FLOWS + " rigid", target=True),
    "dip": T("Dip the {o} into the {t}.", SOLID, target=True), "squeeze": T("Squeeze the {o}.", SOFT),
    "apply": T("Apply the {o} onto the {t}.", FLOWS + " elastic", target=True), "scrape": T("Scrape the {o}.", FLOWS + " divisible plastic"),
    # stir / whisk / shake
    "stir": T("Stir the {o}.", FLOWS), "whisk": T("Whisk the {o}.", FLOWS), "whip": T("Whip the {o}.", FLOWS),
    "shake": T("Shake the {o}.", "rigid elastic thin_flexible"),
    # wipe / scrub / wash / sweep
    "wipe": T("Wipe the {o}.", "rigid articulated", fixture=True), "scrub": T("Scrub the {o}.", "rigid", fixture=True),
    "clean": T("Clean the {o}.", "rigid articulated", fixture=True), "wash": T("Wash the {o}.", "rigid divisible"),
    "rinse": T("Rinse the {o}.", "rigid divisible"), "brush": T("Brush the {o}.", "rigid cloth", fixture=True),
    "sweep": T("Sweep the {o}.", "granular", fixture=True), "vacuum": T("Vacuum the {o}.", "cloth", fixture=True),
    # fold / smooth / spread
    "fold": T("Fold the {o}.", "cloth thin_flexible"), "smooth": T("Smooth out the {o}.", "cloth thin_flexible"),
    "flatten": T("Flatten the {o}.", "cloth thin_flexible plastic"), "spread": T("Spread the {o} flat.", "cloth thin_flexible"),
    "stretch": T("Stretch the {o}.", "cloth elastic linear plastic"),
    # tie
    "tie": T("Tie the {o}.", "linear cloth thin_flexible"), "untie": T("Untie the {o}.", "linear cloth thin_flexible"),
    # cut / peel / knead / roll
    "cut": T("Cut the {o}.", "divisible plastic thin_flexible"), "peel": T("Peel the {o}.", "divisible"),
    "knead": T("Knead the {o}.", "plastic elastic divisible"), "roll": T("Roll the {o} on the table.", "plastic elastic cloth"),
    "shape": T("Shape the {o} into a ball.", "plastic elastic"),
    # tool use
    "scan": T("Scan the {o} with the barcode scanner.", "rigid thin_flexible elastic divisible"),
    "aim": T("Aim the {o} at the {t}.", "rigid", target=True), "stamp": T("Stamp the {o}.", "thin_flexible"),
    "iron": T("Iron the {o}.", "cloth"), "strike": T("Strike the {o}.", "rigid"),
    # gesture
    "swing": T("Swing the {o}.", "rigid elastic cloth"),
    # base, toward an object
    "approach": T("Approach the {o}.", "rigid articulated", fixture=True), "face": T("Face the {o}.", "rigid articulated cloth", fixture=True),
}

# direction words the parser recognises (scripts/capeval/parse.py DIRECTIONS) -> the phrase used in prompts
DIR_PHRASE = {"leftward": "to the left", "rightward": "to the right", "up": "upward", "down": "down", "forward": "forward",
              "backward": "backward", "aside / away": "aside", "closer": "closer", "to the edge": "to the edge of the table",
              "to the center": "to the center", "left→right": "from left to right", "right→left": "from right to left"}

# "seen action + direction on a new object": (verb, direction) pairs that occur in training, with the template.
# lift / lower / pull down are left out: their direction is the verb itself, so they are already the plain probes.
DIR_TEMPLATES = {
    "push": T("Push the {o} {d}.", SOLID), "pull": T("Pull the {o} {d}.", SOLID),
    "slide": T("Slide the {o} {d}.", SOLID), "move": T("Move the {o} {d}.", SOLID),
    "carry": T("Carry the {o} {d}.", SOLID), "fold": T("Fold the {o} {d}.", "cloth thin_flexible"),
    "roll": T("Roll the {o} {d}.", "plastic elastic cloth"), "sweep": T("Sweep the {o} {d}.", "granular", fixture=True),
    "stretch": T("Stretch the {o} {d}.", "cloth elastic linear"), "bring": T("Bring the {o} {d}.", SOLID),
    # base motion: the scene object is a landmark, not the thing moved
    "move base": T("Move {d} toward the {o}.", "rigid articulated cloth", fixture=True),
    "turn base": T("Turn the chassis {d}.", ""),
}
DIR_ENTRIES = [
    ("push", "aside / away"), ("push", "to the edge"), ("pull", "rightward"), ("pull", "to the edge"),
    ("slide", "up"), ("slide", "down"), ("slide", "to the edge"), ("move", "forward"), ("move", "backward"),
    ("move", "aside / away"), ("carry", "forward"), ("carry", "backward"), ("carry", "aside / away"), ("carry", "closer"),
    ("bring", "closer"), ("fold", "down"), ("fold", "up"), ("fold", "leftward"), ("fold", "to the center"),
    ("fold", "right→left"), ("roll", "left→right"), ("roll", "right→left"), ("sweep", "left→right"),
    ("stretch", "leftward"), ("move base", "forward"), ("move base", "backward"), ("move base", "closer"),
    ("turn base", "leftward"), ("turn base", "rightward"),
]
# "Move backward toward X" is not a sentence: base prompts per direction
BASE_DIR_TEXT = {("move base", "forward"): "Move forward toward the {o}.",
                 ("move base", "backward"): "Move backward away from the {o}.",
                 ("move base", "closer"): "Move closer to the {o}."}

# unseen actions: no inflection of the verb occurs in any training caption (checked by build_index.py on raw words)
UNSEEN = {
    "tilt": {"t": "Tilt the {o} to one side.", "mats": "rigid elastic", "forms": ["tilt", "tilts", "tilted", "tilting"]},
    "tap": {"t": "Tap the {o} lightly.", "mats": SOLID + " articulated", "forms": ["tap", "taps", "tapped", "tapping"]},
    "poke": {"t": "Poke the {o}.", "mats": SOFT, "forms": ["poke", "pokes", "poked", "poking"]},
    "toss": {"t": "Toss the {o} onto the table.", "mats": SOLID, "forms": ["toss", "tosses", "tossed", "tossing"]},
    "drop": {"t": "Drop the {o} onto the table.", "mats": SOLID, "forms": ["drop", "drops", "dropped", "dropping"]},
    "crumple": {"t": "Crumple the {o}.", "mats": "thin_flexible cloth", "forms": ["crumple", "crumples", "crumpled", "crumpling"]},
    "squash": {"t": "Squash the {o}.", "mats": SOFT, "forms": ["squash", "squashes", "squashed", "squashing"]},
    "spin": {"t": "Spin the {o} on the table.", "mats": "rigid elastic", "forms": ["spin", "spins", "spun", "spinning"]},
    "knock over": {"t": "Knock over the {o}.", "mats": "rigid elastic", "forms": ["knock", "knocks", "knocked", "knocking"]},
    "rub": {"t": "Rub the {o}.", "mats": SOLID + " articulated", "forms": ["rub", "rubs", "rubbed", "rubbing"]},
    "pat": {"t": "Pat the {o}.", "mats": SOFT, "forms": ["pat", "pats", "patted", "patting"]},
    "wring": {"t": "Wring the {o}.", "mats": "cloth", "forms": ["wring", "wrings", "wrung", "wringing"]},
    "crush": {"t": "Crush the {o}.", "mats": "thin_flexible divisible elastic", "forms": ["crush", "crushes", "crushed", "crushing"]},
    "pinch": {"t": "Pinch the {o}.", "mats": SOFT, "forms": ["pinch", "pinches", "pinched", "pinching"]},
    "bend": {"t": "Bend the {o}.", "mats": "linear thin_flexible elastic plastic", "forms": ["bend", "bends", "bent", "bending"]},
    "tear": {"t": "Tear the {o} in half.", "mats": "thin_flexible divisible plastic", "forms": ["tear", "tears", "tore", "tearing"]},
    "point at": {"t": "Point at the {o}.", "mats": SOLID + " articulated", "forms": ["point", "points", "pointed", "pointing"]},
    "flick": {"t": "Flick the {o}.", "mats": "rigid elastic thin_flexible", "forms": ["flick", "flicks", "flicked", "flicking"]},
    "bounce": {"t": "Bounce the {o} on the table.", "mats": "elastic", "forms": ["bounce", "bounces", "bounced", "bouncing"]},
}
UNSEEN_FAMILY = {"tilt": "rotate", "tap": "press", "poke": "press", "toss": "place", "drop": "place", "crumple": "fold",
                 "squash": "pour", "spin": "rotate", "knock over": "push_pull", "rub": "clean", "pat": "press",
                 "wring": "fold", "crush": "cut_shape", "pinch": "grasp", "bend": "fold", "tear": "cut_shape",
                 "point at": "gesture", "flick": "push_pull", "bounce": "place"}
for _k, _u in UNSEEN.items():
    _u["mats"] = set(_u["mats"].split())
    _u["family"] = UNSEEN_FAMILY[_k]
