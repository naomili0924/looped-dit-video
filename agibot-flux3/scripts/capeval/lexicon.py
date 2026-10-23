"""Hand-curated lexicons for the AgiBot training captions (train split, 2,735 unique captions).

VERBS:   verb phrase -> (primitive, family)
OBJECTS: noun phrase -> (canonical object, semantic category, material class, flags)

Material class = how the object's shape can change under manipulation (one per object):
  rigid         cannot change shape (cups, bottles, trays, boxes, tools)
  articulated   rigid with a joint / slider / button (doors, drawers, lids, switches, faucets)
  cloth         2D textile, foldable (shirts, shorts, towels, curtains)
  thin_flexible thin sheet / film / bag, crumplable, no fixed shape (plastic bags, paper, chip bags)
  elastic       soft, squishable, springs back (pillows, cushions, plush toys, squeeze bottles)
  plastic       permanently mouldable (dough)
  linear        1D deformable (rope, cord, cable, ties)
  liquid        flows, takes the container's shape (water, milk, sauce, wine)
  granular      many small rigid pieces / powder that flow (rice, sugar, pearls, ice cubes)
  divisible     solid that is cut, peeled, torn or split into pieces (produce, bread, meat)
  fixture       static scene surface the hand works on but does not move (table, toilet, wall)
  person        a human or a body part
Flags (any number): fragile transparent reflective container contents heavy sharp hot powered edible squeezable
"""

# ----------------------------------------------------------------------------------------------- verbs
# (phrase, primitive, family). Longest phrase wins; phrases are matched at the start of a clause.
VERBS = [
    # grasp / acquire
    ("pick up", "pick up", "grasp"), ("pick", "pick up", "grasp"), ("picks up", "pick up", "grasp"),
    ("grasp", "grasp", "grasp"), ("grab", "grab", "grasp"), ("grip", "grip", "grasp"),
    ("take out", "take out", "grasp"), ("take off", "take off", "grasp"), ("take", "take", "grasp"),
    ("takes", "take", "grasp"), ("retrieve", "retrieve", "grasp"), ("fetch", "fetch", "grasp"),
    ("hold", "hold", "grasp"), ("clamp", "clamp", "grasp"), ("receive", "receive", "handover"),
    # vertical transport
    ("lift up", "lift", "lift_lower"), ("lift", "lift", "lift_lower"), ("raise", "raise", "lift_lower"),
    ("lower", "lower", "lift_lower"), ("put down", "put down", "place"),
    # place / release
    ("place", "place", "place"), ("put", "place", "place"), ("set", "place", "place"),
    ("return", "place", "place"), ("store", "place", "place"), ("hang", "hang", "place"),
    ("insert", "insert", "place"), ("stack", "stack", "place"), ("release", "release", "place"),
    ("throw", "throw", "place"), ("dump", "dump", "place"), ("discard", "throw", "place"),
    ("arrange", "place", "place"), ("align", "align", "place"), ("adjust", "adjust", "place"),
    # transport / handover
    ("carry", "carry", "transport"), ("move", "move", "transport"), ("bring", "bring", "transport"),
    ("convey", "carry", "transport"), ("deliver", "carry", "transport"),
    ("pass", "pass", "handover"), ("hand over", "hand over", "handover"), ("hand", "hand over", "handover"),
    # non-prehensile planar
    ("push open", "push open", "articulate"), ("pull open", "pull open", "articulate"),
    ("pull out", "pull out", "push_pull"), ("pull down", "pull down", "push_pull"),
    ("push", "push", "push_pull"), ("pull", "pull", "push_pull"), ("slide", "slide", "push_pull"),
    ("drag", "drag", "push_pull"), ("draw", "draw", "push_pull"), ("swipe", "swipe", "push_pull"),
    # articulate
    ("open", "open", "articulate"), ("close", "close", "articulate"), ("cover", "cover", "articulate"),
    ("unlock", "unlock", "articulate"), ("lock", "lock", "articulate"), ("seal", "seal", "articulate"),
    # rotate / screw
    ("turn on", "turn on", "rotate"), ("turn off", "turn off", "rotate"), ("turn", "turn", "rotate"),
    ("twist off", "twist off", "rotate"), ("twist open", "twist open", "rotate"), ("twist", "twist", "rotate"),
    ("rotate", "rotate", "rotate"), ("tighten", "tighten", "rotate"), ("unscrew", "unscrew", "rotate"),
    ("flip", "flip", "rotate"),
    # press / actuate
    ("press", "press", "press"), ("click", "press", "press"), ("touch", "press", "press"),
    ("unplug", "unplug", "place"), ("plug", "insert", "place"),
    # pour / transfer material
    ("pour", "pour", "pour"), ("scoop", "scoop", "pour"), ("sprinkle", "sprinkle", "pour"),
    ("empty", "empty", "pour"), ("dip", "dip", "pour"), ("squeeze", "squeeze", "pour"),
    ("apply", "apply", "pour"), ("fill", "pour", "pour"), ("shovel", "scoop", "pour"),
    ("water", "pour", "pour"), ("scrape", "scrape", "pour"),
    # mix
    ("stir", "stir", "mix"), ("whip", "whip", "mix"), ("whisk", "whisk", "mix"), ("shake", "shake", "mix"),
    # clean / surface contact
    ("wipe", "wipe", "clean"), ("scrub", "scrub", "clean"), ("clean", "clean", "clean"),
    ("sweep", "sweep", "clean"), ("mop", "mop", "clean"), ("vacuum", "vacuum", "clean"), ("wash", "wash", "clean"),
    ("rinse", "rinse", "clean"), ("brush", "brush", "clean"), ("disinfect", "disinfect", "clean"),
    ("remove", "remove", "grasp"), ("roll away", "roll", "clean"), ("tidy", "tidy", "place"),
    # cloth manipulation
    ("fold", "fold", "fold"), ("smooth out", "smooth", "fold"), ("smooth", "smooth", "fold"),
    ("flatten", "flatten", "fold"), ("spread", "spread", "fold"), ("stretch", "stretch", "fold"),
    ("tie", "tie", "tie"), ("untie", "untie", "tie"), ("zip", "zip", "tie"),
    # cut / shape
    ("cut", "cut", "cut_shape"), ("slice", "slice", "cut_shape"), ("chop", "cut", "cut_shape"),
    ("peel", "peel", "cut_shape"), ("knead", "knead", "cut_shape"), ("shape", "shape", "cut_shape"),
    ("roll", "roll", "cut_shape"), ("tear", "tear", "cut_shape"), ("shred", "shred", "cut_shape"),
    # tool use (the tool is the point)
    ("scan", "scan", "tool_use"), ("aim", "aim", "tool_use"), ("stamp", "stamp", "tool_use"),
    ("paint", "paint", "tool_use"), ("iron", "iron", "tool_use"), ("hammer", "hammer", "tool_use"),
    ("strike", "strike", "tool_use"), ("spray", "spray", "tool_use"), ("feed", "insert", "place"), ("pack", "pack", "place"), ("extract", "pull out", "push_pull"),
    # base / locomotion
    ("face", "face", "locomotion"), ("step", "step", "locomotion"), ("approach", "approach", "locomotion"),
    ("leave", "leave", "locomotion"), ("advance", "move", "locomotion"),
    # gestures / social
    ("wave", "wave", "gesture"), ("clap", "clap", "gesture"), ("swing", "swing", "gesture"),
    ("shoot", "throw", "place"), ("cheers", "cheers", "gesture"), ("stretch out", "stretch", "gesture"),
    ("wait", "wait", "wait"), ("coordinate", "coordinate", "other"), ("use", "use", "tool_use"),
    ("confirm", "confirm", "other"),
    # verbs that do not occur in training, kept so probe prompts parse (count 0 by construction)
    ("drop", "drop", "place"), ("crumple", "crumple", "fold"), ("squash", "squash", "pour"), ("nudge", "push", "push_pull"),
    ("tap", "tap", "press"), ("poke", "poke", "press"), ("toss", "toss", "place"), ("tilt", "tilt", "rotate"),
]
FAMILY_LABEL = {
    "grasp": "Grasp / pick up", "lift_lower": "Lift / lower", "place": "Place / insert / hang / throw",
    "transport": "Carry / move", "handover": "Hand over / receive", "push_pull": "Push / pull / slide",
    "articulate": "Open / close / cover", "rotate": "Turn / twist / rotate", "press": "Press / click",
    "pour": "Pour / scoop / squeeze", "mix": "Stir / whisk / shake", "clean": "Wipe / scrub / wash / sweep",
    "fold": "Fold / smooth / spread", "tie": "Tie / untie", "cut_shape": "Cut / peel / knead / roll",
    "tool_use": "Tool use (scan, stamp, iron, …)", "locomotion": "Base: turn / move / face",
    "gesture": "Gesture (wave, clap)", "wait": "Wait", "other": "Other",
}

# --------------------------------------------------------------------------------------------- objects
# phrase[;synonym...] | canonical | category | material | flags
_OBJ = r"""
milk green water bottle;water bottle;bottled water;bottle of water;mineral water;small bottle mineral water;milky white water bottle;gray water bottle|water bottle|drink|rigid|container contents transparent
bottle cap;cap of the held water bottle;cap|bottle cap|packaged|rigid|
bottle;thermos;seasoning bottle;salad dressing bottle;vinegar bottle|bottle|packaged|rigid|container contents
ketchup bottle;ketchup|ketchup bottle|packaged|elastic|container contents squeezable
red wine;wine|red wine|drink|liquid|edible
decanter|decanter|tableware|rigid|container fragile transparent
water;hot water;boiling water|water|drink|liquid|
milk tea|milk tea|drink|liquid|edible
milk|milk|drink|liquid|edible
tea bag|tea bag|food|thin_flexible|edible
tea pi|tea cake|food|rigid|edible
tea;green tea;bucket green tea;bucket tea;oolong;chrysanthemum tea;yellow chrysanthemum tea;iced tea|tea (drink)|drink|liquid|edible
coffee;instant coffee;coffee liquid;packaged concentrated coffee liquid|coffee|drink|liquid|edible
instant coffee powder;coffee powder;powder|coffee powder|food|granular|edible
capsule coffee;coffee capsule;capsule;capsules|coffee capsule|packaged|rigid|
juice;orange juice;plum juice;beverage;drink;cola;soda;yogurt;lactic acid bacteria;lychee flavored beverage concentrate;beverage concentrate;concentrate|beverage|drink|liquid|edible
syrup|syrup|food|liquid|edible
salad dressing;dressing;sauce;sweet bean sauce;light soy sauce;soy sauce;doubanjiang;vinegar;white vinegar;aged vinegar;oil|sauce / condiment|food|liquid|edible
liquid|liquid|drink|liquid|
paint|paint|tool|liquid|
detergent;washing detergent;laundry detergent|detergent|care|liquid|container
seasoning;spices;msg;salt;sugar|seasoning powder|food|granular|edible
canned salt can;canned msg can;canned sugar;can of instant coffee;canned|seasoning can|packaged|rigid|container contents
tapioca pearls;black pearl tapioca pearls;pearls;pearl;crystal ball;original flavor crystal ball;peach-flavored crystal ball|tapioca pearls|food|granular|edible
oatmeal;oatmeal porridge|oatmeal|food|granular|edible
rice;sorghum grain;grain;beans;red beans;mung beans;mung bean;kernels;corn kernels;seeds;sunflower seeds;nuts;peanuts;flaxseed|grains & beans|food|granular|edible
ice|ice|food|granular|
trash;crumbs;dust;waste|trash / crumbs|cleaning|granular|
dough|dough|food|plastic|edible
toast;plain toast;whole wheat toast;bread;bun;pancake;sandwich;hamburger|bread|food|divisible|edible
meat slice;meat slices;meat;steak;bacon;sausage;hot pot sausage;ham;pork;beef;chicken wings;crab stick;shrimp|meat|food|divisible|edible
egg;half an egg|egg|food|divisible|edible fragile
egg tart;dessert;cake;donut|dessert|food|divisible|edible fragile
vegetable leaves;vegetable leaf;leaves;leaf;lettuce|vegetable leaf|food|thin_flexible|edible
potato;carrot;organic carrot;radish;green radish;cucumber;onion;corn;okra;boxed okra;peppers;tomato;tomatoes;garlic;mushroom;shiitake mushroom;shiitake;matsutake;vegetable;vegetables|vegetable|food|divisible|edible
apple;banana;orange;lemon;lemon slice;kiwi fruit;kiwi;mango;pomegranate;grapes;grape;pear;peach;persimmon;pineapple;coconut;lychee;longan;cherry;cherry tomatoes;plum;carambola;fruit;fruits;tangerine|fruit|food|divisible|edible
potato chips;red potato chips;large bag of potato chips;green large bag of potato chips;chips;potato roll;fish fry;cheers crisp;blue cheers crisp;konjac spicy strips;spicy strips;swallow dumplings;dried bean curd sticks;utensil bag;taro chips;purple taro chips;shrimp chips;fries;green fries;red fries;corn snacks;green corn snacks;snack;snacks;crisps;puffs|bagged snack|packaged|thin_flexible|container contents edible
large bucket of potato chips;bucket of potato chips|chips canister|packaged|rigid|container contents edible
chewing gum;bottled chewing gum|bottled gum|packaged|rigid|container
noodles;vermicelli;instant noodles|noodles|food|divisible|edible
peel;outer packaging skin;packaging skin|peel / wrapper|cleaning|thin_flexible|
medicine;medicine box;medicine chest|medicine box|packaged|articulated|container
medicine box partition;medicine box divider;divider;partition|box divider|container|rigid|
medicine box cover;medicine box lid|box lid|container|articulated|
cup;water cup;glass;milk tea cup;measuring cup;shaker cup;empty cup;water-filled cup;paper cup|cup|tableware|rigid|container
shaker|cocktail shaker|tableware|rigid|container
bowl;sauce dish;dish|bowl|tableware|rigid|container
plate;dining plate;porcelain plate;blue and white porcelain plate;blue plastic plate|plate|tableware|rigid|fragile
tray;rectangular tray;baking tray;baking pan;white kitten tray;hepburn-style tray;wooden tray;light-colored wooden tray;meal tray;dining tray|tray|tableware|rigid|
kettle;golden kettle;green kettle;gray kettle;hot water kettle;golden transparent handle kettle;pink flower kettle;teapot;beige teapot;transparent teapot|kettle|appliance|rigid|container contents heavy
kettle lid;green kettle lid|kettle lid|appliance|articulated|
pot;frying pan;black frying pan;pan;inner pot;rice cooker inner pot|pot / pan|tableware|rigid|container heavy
spoon;bar spoon;spoon/fork;fork;chopsticks;disposable chopsticks;cutlery;children's tableware;tableware;utensil|cutlery|tableware|rigid|
knife|knife|kitchen_tool|rigid|sharp
spatula;dessert spatula;dessert shovel;wooden spatula;dark wooden spatula;ice scoop;scoop;shovel|spatula / scoop|kitchen_tool|rigid|
peeler;vegetable scraper;scraper|peeler|kitchen_tool|rigid|sharp
vegetable cutter;cutter;dicer;slicer;hand guard|vegetable cutter|kitchen_tool|articulated|sharp
whisk|whisk|kitchen_tool|rigid|
rolling pin|rolling pin|kitchen_tool|rigid|
oil brush;cup brush|kitchen brush|cleaning|rigid|
cutting board;heat-resistant pad;pad|board / pad|kitchen_tool|rigid|
straw|straw|tableware|rigid|
straw holder;chopstick holder;pen holder;tissue holder;holder|holder|container|rigid|
lunchbox;takeout box;fast food container;food container;container;meal box;takeout box lid|food container|container|rigid|container
takeout bag;plastic bag;white plastic bag;bag;garbage bag;trash bag;shopping bag;packing bag;wrap;film;air column film;tissue bag;empty tissue bag;packaging|bag / film|paper_bag|thin_flexible|container
coin purse;purse;gray felt bag;felt bag;red canvas tote bag;canvas tote;tote bag;tote;non-woven bag;backpack;pink bunny backpack;schoolbag|fabric bag|paper_bag|cloth|container
box;storage box;carton;cardboard box;logistics box;packing box;material box;replenishment box;restocking box;collection box;target box;paper box;box cover;box lid;tissue box;feed box;feed basket;capsule box;used capsule box;discarded capsule box|box / carton|container|rigid|container
basket;laundry basket;storage basket;filter basket;draining basket;fruit bowl;collection basin;residual container;waste bin|basket / bin|container|rigid|container
trash can;trash bin;desktop trash bin;bin;large trash bin;large trash can;large bin|trash can|container|rigid|container
bucket;paint bucket|bucket|container|rigid|container contents
vase;transparent patterned vase;golden vase;flowerpot|vase|container|rigid|container fragile
jar|jar|container|rigid|container fragile
plum blossom;red plum blossom;white plum blossom;flower;flowers;hydrangea;rose;plants|flower|plant|elastic|fragile
paper;scrap paper;paper ball;document;documents;reimbursement form;form;copy original;original copy;original;sheet;sheets;card;cards;toy cards;menu;black menu;brown menu;sticky notes;notepad;padded notepad|paper / card|paper_bag|thin_flexible|
book;books;notebook|book|office|rigid|
tissue;tissues;blue tissues;napkin;used napkin;wet wipes;wipes;toilet paper roll;roll of paper;new roll of paper;empty roll|tissue / napkin|paper_bag|thin_flexible|
nameplate;name tag;name tags;stacked nameplate|nameplate|office|rigid|
marker;marker pen;whiteboard marker;uncapped marker pen;uncapped whiteboard marker;pen;fountain pen;ballpoint pen;pencil;white paint marker;paint marker;sharpener|pen / marker|office|rigid|
eraser;whiteboard eraser;english whiteboard eraser;green english whiteboard eraser;yellow english whiteboard eraser|eraser|cleaning|rigid|
pen cap;corresponding cap|pen cap|office|rigid|
stamp;inkpad;ink pad|stamp|office|rigid|
printer;printer cover;shredder|printer / shredder|appliance|articulated|powered
remote control;remote;silver remote control;dark black remote control|remote control|electronics|rigid|powered
charger;mobile phone charger;power bank;gray power bank;phone;mobile phone;mouse;usb;usb drive;cable;ethernet cable;card reader;camera;battery|small electronics|electronics|rigid|powered
memory module;memory stick;memory card|memory module|electronics|rigid|powered
plug;power strip|plug / socket|electronics|articulated|powered
knob;light-switch;switch;light switch;button;start button;switch button;power button;copy button;open button;lid-opening button;heating button;water heating button;boiling water button;water boiling button;oven start button;bell|switch / button|appliance|articulated|powered
faucet;faucet handle|faucet|fixture|articulated|
showerhead|showerhead|fixture|rigid|
door;door handle;doorknob;cabinet door;wardrobe door;refrigerator door;fridge door;freezer door;oven door;washing machine door;microwave door;opened door|door|furniture|articulated|
lid;cover;rice cooker lid;dishwasher lid;lid of desktop trash bin;ice maker's lid|lid|appliance|articulated|
latch;buckle;lock;door lock;zipper|latch / lock|furniture|articulated|
key|key|tool|rigid|
drawer;storage drawer|drawer|furniture|articulated|container
wardrobe;cabinet;mirror cabinet;refrigerator;fridge;freezer|cabinet / fridge|furniture|articulated|container
washing machine;dishwasher;microwave;orange microwave;black microwave;oven;ice maker;rice cooker;coffee machine;capsule coffee machine;toaster;garment steamer;vegetable cutter lid|appliance|appliance|articulated|powered hot
fan;electric fan;tv;hair dryer;hairdryer;green hair dryer;handheld vacuum;vacuum cleaner;pink vacuum cleaner;floor mopping robot;mopping robot|small appliance|appliance|rigid|powered
steam iron;iron|steam iron|appliance|rigid|powered hot heavy
disinfection gun;disinfectant gun;sanitizing gun;sterilizing gun|disinfection gun|tool|rigid|powered
metal detector;handheld metal detector|metal detector|tool|rigid|powered
barcode scanner;scanner|barcode scanner|tool|rigid|powered
mallet;hammer|hammer / mallet|toy|rigid|
gong|gong|toy|rigid|reflective
dumbbell;dumbbells|dumbbell|toy|rigid|heavy
basketball;ball;hoop|ball|toy|elastic|
plush toy;toy;bear;bunny;doll;pony;rabbit;deer;duck|plush toy|toy|elastic|
rubik's cube;building block;cylinder building block;building blocks;toy car;car;bus;bulldozer;vehicle|rigid toy|toy|rigid|
pillow;sofa cushion;cushion|pillow / cushion|textile|elastic|
quilt;bedsheet|quilt|textile|cloth|
pillowcase;charcoal gray pillowcase;blue pillowcase|pillowcase|textile|cloth|
tablecloth|tablecloth|textile|cloth|
curtain;curtains|curtain|textile|cloth|
towel;cloth;cloths;dishcloth;washcloth;children's exfoliating washcloth;rag;fabric|towel / cloth|textile|cloth|
collar;cuffs;sleeve;left sleeve;hem;bottom seam;seam;short-sleeve;short sleeve;short-sleeve garment;t-shirt;shirt;long-sleeve shirt;blue long-sleeve shirt;garment;clothes;clothing;light-colored clothing;dark clothing;laundry|shirt / garment|textile|cloth|
waistband;pant-leg;shorts;denim shorts;pants;trousers;casual trousers;jeans;athletic shorts;sports shorts|shorts / trousers|textile|cloth|
baseball cap;hat|hat|textile|elastic|
rope;pull cord;cord;curtain tie;curtain ties;tiebacks;velcro curtain tiebacks;curtain sash;lanyard;string|rope / cord|textile|linear|
gauze;bandage|gauze / bandage|care|cloth|
hanger;plastic hanger;clothes hanger;black hanger|hanger|furniture|rigid|
lint roller|lint roller|cleaning|rigid|
brush;paint brush|brush|cleaning|rigid|
broom;dustpan;mop|broom / dustpan|cleaning|rigid|
toothpaste;whitening toothpaste;cotton swab;double-headed cotton swab;swab;swabs;toothbrush;toothbrushes;toothpick;floss;dental floss|swab / toothbrush|care|rigid|
toilet cleaner;cleaner;hand cream;blue hand cream;cream;blue cream;lotion;vitamin e lotion;red vitamin e lotion;facial cleanser;cleanser;soap;shampoo;conditioner;toner;mouthwash;sanitizer;air freshener;lipstick;moisturizing oil;sunscreen;povidone-iodine;alcohol;nasal aspirator;nail clippers;masks;gel|care product|care|rigid|container
permanent magnet ingot;ingot;magnet|magnet ingot|industrial|rigid|heavy
package;item;items;object;product;goods|item (generic)|packaged|rigid|
seasoning box;chicken bouillon|seasoning box|packaged|rigid|container
person;person's;guest;customer;child|person|person|person|
left arm;right arm;arm;leg;legs;shoulders;hands|body part|person|person|
table;desk;desktop;tabletop;counter;countertop;cooking table;operating table;dining table;conference table;cashier counter;checkout counter;cash register;bar;restaurant bar;platform;surface;workstation|table / counter|surface|fixture|
shelf;bookshelf;rack;double-layer shelf;double-layer rack;double-layer storage rack;material frame;clothes rod;rod;hook;wall hook;cart|shelf / rack|surface|fixture|
floor;ground|floor|surface|fixture|
wall|wall|surface|fixture|
whiteboard|whiteboard|surface|fixture|
mirror|mirror|surface|fixture|reflective
toilet;water tank;tank|toilet|surface|fixture|
sink;washstand|sink|surface|fixture|
bed;sofa;couch;windowsill;armrest|bed / sofa|surface|fixture|
stains;stain;water stain;marks;hair|stain / marks|surface|fixture|
slot;gap;base;conveyor belt;belt;handrail;turntable|slot / base|surface|fixture|
"""

CATEGORY_LABEL = {
    "food": "Food & ingredients", "drink": "Drinks", "packaged": "Packaged goods", "tableware": "Tableware & cookware",
    "kitchen_tool": "Kitchen tools", "container": "Boxes, baskets & bins", "paper_bag": "Paper, bags & film",
    "textile": "Clothes & textiles", "furniture": "Doors, drawers & furniture parts", "appliance": "Appliances & switches",
    "electronics": "Small electronics", "tool": "Handheld tools & devices", "cleaning": "Cleaning tools & waste",
    "care": "Personal care & medical", "office": "Office & stationery", "toy": "Toys & sports", "plant": "Plants & flowers",
    "industrial": "Industrial parts", "fixture": "Fixtures (faucet, showerhead)", "surface": "Surfaces & furniture (static)",
    "person": "People & body parts",
}
MATERIAL_LABEL = {
    "rigid": "Rigid — cannot change shape", "articulated": "Articulated — hinge, slider or button",
    "cloth": "Cloth — foldable textile", "thin_flexible": "Thin & flexible — bags, paper, film",
    "elastic": "Soft elastic — squishable, springs back", "plastic": "Plastic — mouldable (dough)",
    "linear": "Linear deformable — rope, cord", "liquid": "Liquid — flows", "granular": "Granular / powder — pours as pieces",
    "divisible": "Divisible — cut, peeled, split", "fixture": "Static surface / fixture", "person": "Person / body",
}

OBJECTS = []
for line in _OBJ.strip().splitlines():
    phrases, canon, cat, mat, flags = line.split("|")
    for p in phrases.split(";"):
        OBJECTS.append((p.strip(), canon, cat, mat, tuple(flags.split())))
