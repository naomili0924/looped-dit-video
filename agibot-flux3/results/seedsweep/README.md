# Random-Depth Seed Sweep — what the model learns and what it does not

- Page (claude.ai artifact): https://claude.ai/artifact/PN27FAM1mM5UsLdamGzzFK
- Same page on GitHub Pages: https://naomili0924.github.io/looped-dit-video/seed-sweep/
- Numbers: `rlong<step>/summary.json` (per clip: `raw.json`); tables in [`../RESULTS.md`](../RESULTS.md#seed-sweep-random-depth-long-run);
  how to rerun: [`../../EVAL.md`](../../EVAL.md#seed-sweep-random-depth-long-run).

## Setup

FLUX 3 Action 7B, fine-tuned on AgiBot World Beta head-camera clips, retrofitted as a Looped-DiT (blocks 12 / 9 / 12,
the 9 core blocks repeated N times) and trained with the loop depth sampled in N 1–8 per step
(`Jinyan0924/flux3-agibot-looped/random_depth_n1-8_long`, EMAs at steps 3000–9500). Each rollout starts from **one real
frame** and **one prompt** and chains four 33-frame windows (133 frames, 8.9 s at 15 fps, 384 × 512; every later window
starts from the model's own last frame). Grid: 8 test cases × 3 prompts × 5 noise seeds × N 1 2 4 8 16, at steps 3000,
4000, 5000, 6000, 7000, 8000, 9500 (840 clips per step). LPIPS/PSNR to the ground truth use the real prompt only and the
frames the episode actually has; "between seeds" is the mean LPIPS over the 10 seed pairs.

## What the eval cases have in common with training

The 8 cases come from the **test** split: held-out *episodes* of tasks that are in training (the split holds out 2.5 %
of each task's episodes; 5 other whole tasks form `test_unseen` and are not used here). Training has 212 task ids
(191 task names), 7,960 episodes, 159,983 windows. Counts below are training windows from `windows/train.jsonl` of
`Jinyan0924/agibot-flux3-eval-splits`.

| case | training task (episodes / windows) | real prompt (p0), training windows with exactly this caption | alternative prompts (ours), training windows |
|---|---|---|---|
| toilet | Clean toilet with a rag (36 / 767) | Wipe the toilet with the held cloth. — **465** | put the cloth down on the toilet lid — 0; lift it up and away — 0 |
| cap | Arrange sofa (39 / 1,624) | Pick up the baseball cap on the sofa. — **174** | push the cap further back — 0; leave it and pull the hand back — 0 |
| chips_bag | Pack in the supermarket cashier (37 / 1,082) | Place the large bucket of potato chips … into the gray felt bag … — **88** | lift it straight up — 0; put it down on the counter — 0 |
| vacuum | Vacuum crumbs with a handheld vacuum (39 / 197) | Use pink vacuum cleaner … with right arm. — **47** | lift the vacuum off the floor — 0; same action with the **left** arm — 3 |
| doorknob | Insert the key and open the door (35 / 375) | Turn the doorknob to open the door with left arm. — **42** | release and pull the arm back — 0; same action with the **right** arm — 39 |
| marker | Insert the pen cap (38 / 527) | Pick up the uncapped whiteboard marker … with left arm. — **33** | slide it across the table — 0; pick it up with the **right** arm — 29 |
| shorts | Roll away stains with a lint roller (38 / 884) | Slide the dark shorts on the desktop upward with the left arm. — **67** | slide them **downward** — 46; upward with the **right** arm — 0 |
| taro_chips | Place goods from the material box onto the shelves (38 / 566) | Pick up the purple taro chips from the restocking box with the right arm. — **260** | push them down into the box — 0; pick them up with the **left** arm — 0 |

So: every **scene, object and task** was seen in training (35–39 training episodes each, the same rooms and props);
every **real caption** was seen verbatim (33–465 windows); 12 of the 16 **alternative prompts** never appear in
training and 4 do (left/right-arm or up/down variants that the task itself contains).

The training captions are dominated by pick-and-place: the first word is *place* (20 % of windows), *pick* (14 %),
*grasp* (7 %), *lift* (6 %), *press*, *open*, *pour*, *use*, *pass*, *grab*, *close*, *turn* (2–4 % each). Household,
kitchen, laundry and retail tasks (serve a meal, make milk tea, fold shorts / T-shirts, restock shelves, pack at the
cashier, wash dishes, iron clothes, …). Several verbs of our alternative prompts are rare or absent as phrased: *put …
down on the counter*, *lift … straight up*, *push … further back*, *slide … across the table* and *leave … and pull the
hand back* occur in 0 training windows; *release* occurs in 724, but never for these objects.

## What the model learns

- **The scene and the camera.** All 133 frames keep the room, the props and the fixed head camera of the context frame;
  backgrounds do not drift or melt even in window 4 (checked by eye on the step-9500 clips; LPIPS to the
  ground truth stays below 0.42 at every step and depth).
- **The robot and the seen action for the seen caption.** With the real prompt the arm does the task's motion in the
  right place: the cloth wipes the toilet tank, the chips bucket goes into the felt bag, the arm reaches toward the
  shelf with the taro chips, the left arm moves along the door edge toward the knob.
- **More of the episode as training goes on (windows 2–3).** N 1 LPIPS to the ground truth, w2 / w3: 0.284 / 0.344 at
  step 3000 → 0.253 / 0.318 at 6000 → 0.248 / 0.305 at 9500; PSNR w3 14.92 → 15.76 dB.
- **Less dependence on the noise seed.** w4 LPIPS between seeds 0.376 (3000) → 0.314 (9500): later checkpoints commit to
  more similar futures.
- **Loop depth is irrelevant at inference.** Trained with random N 1–8, the model is as good at N 1 as at N 8, and N 16
  (twice the trained maximum) does not break it; N 1–2 is best or tied at every step (the one exception, N 4 at step
  6000 in w4, wins by 0.007, inside the seed spread).

## What the model fails to learn

- **Window 4 does not improve.** N 1 LPIPS to the ground truth in w4 stays at 0.379–0.395 from step 3000 to 9500; the
  gain from training stops at about 6.6 s.
- **Motion fades over the rollout.** Mean frame-to-frame change (0–255): ground truth 2.68 / 2.63 / 2.44 / 3.03 per
  window; the model at step 9500 2.41 / 1.95 / 1.95 / 1.89 — about 40 % too little motion by window 4 (model: all prompts and frames; ground truth: frames the
  episode has), and less than
  step 3000 (2.46 / 2.12 / 2.44 / 2.39). Training makes later windows calmer, not more like the episode.
- **Following a prompt it has not seen.** With the same seed, changing the prompt changes the video *less* than
  changing the noise seed in window 1 (LPIPS 0.139 between prompts vs 0.168 between seeds at step 9500) and about as
  much by window 4 (0.312 vs 0.314). Unseen actions come out as the closest trained motion or as artefacts: "lift the
  bucket of chips straight up" grows the bucket into a frame-filling yellow blob; "put it down on the counter" swaps
  in a different object (a red can) with a new arm; "lift the cloth up and away" makes the cloth balloon over the
  camera.
- **Switching arms.** "… with the left arm" (taro chips) or "… with right arm" (doorknob) brings a second arm into view,
  but its shape breaks (bent or duplicated grippers), and the arm that should act often stays where it was.
- **Object identity over long horizons.** Objects can change into neighbours from the same task: the chips bucket in the
  bag turns into a red bag of chips by window 4; held items can change colour or shape after a window boundary.
- **Episodes that end early.** Where the ground truth stops (vacuum 80 frames, doorknob 92, shorts 110) the model keeps
  acting; those frames have no reference and are excluded from the metrics.

## Takeaways

The random-depth long run learns the AgiBot scenes, the robot and the trained motions for trained captions, and makes
the first ~6.6 s closer to the episode and more seed-consistent with training. It does not learn to keep moving for the
whole 8.9 s, to follow prompts outside the caption set of a task, or to keep object identity across windows; extra loop
depth at inference buys nothing. The test split shares every task with training, so these numbers say nothing about
unseen tasks (`test_unseen`).
