# Capability eval: rare actions and held-out tasks

The seed sweep (`results/seedsweep`, `docs/seed-sweep`) measures the model on held-out episodes drawn from the same
mix as training. This eval asks a different question: what does the model do on actions it has barely trained on,
on tasks it has never trained on, and when the instruction is reworded or changed?

Checkpoint `random_depth_n1-8_long` EMA step 9500, k = 1 context frame, 4 chained windows (1 + 4 × 33 frames =
8.9 s at 15 fps), Euler 30 steps, CFG 4, shift 5. 25 cases × 3 prompts × 3 seeds = 225 clips per loop depth, at
N = 1, 2, 4 and 8 (900 clips). The sections up to the per-case table are at N = 1; see Loop depth for the others.
Results page (private claude.ai artifact): https://claude.ai/artifact/4Nh7eAyynXM677TkTqCwwY

## Why these actions

The training mix is dominated by pick and place. Grasp and place segments make up 46 % of the 160,120 training
windows, and fold makes up another 8.6 %. The ten families tested here are the long tail. Each family is
0.5–5.4 % of the training windows, and all ten together are 21.7 % (caption parse of the training windows, same
parser as the caption atlas):

| family | tested as | share of training windows | training tasks with it |
|---|---|---:|---:|
| pour | pour dressing, scoop tapioca | 5.35 % | 27 |
| rotate | twist cap, turn faucet | 3.55 % | 21 |
| tool_use | barcode scan, stamp | 3.03 % | 16 |
| press | oven button, rice cooker | 2.70 % | 22 |
| locomotion | drive forward, turn base | 2.04 % | 26 |
| transport | carry bag, move guard | 1.96 % | 26 |
| cut_shape | cut dough, knead dough | 1.20 % | 5 |
| gesture | clap, wave | 0.79 % | 19 |
| mix | stir coffee, shake decanter | 0.57 % | 5 |
| tie | untie curtain, tie tieback | 0.52 % | 3 |

There are two kinds of case:

- **20 rare-action cases, 2 per family.** Each one is an episode that was in no split (never sampled, never
  fetched, never trained). The task itself was trained, though: training took up to 40 episodes from every one
  of the 212 trainable tasks, so there is no unseen task with these actions. These cases test new episodes of
  rarely trained actions, and new wording of them. They do not test unseen actions in unseen scenes.
- **5 held-out tasks**, one test_unseen episode each: 358 toast, 389 supermarket, 487 bottled water,
  688 curtains, 739 menu. These are tasks the model never saw at all. Their actions (pick, carry, pull) are the
  common ones, so they test new scenes rather than new actions.

The pipeline also accepts edited context images (`"source": "generated"`) for actions with no never-trained
episode. None were needed.

## Prompts

- real: AgiBot's own caption, verbatim. For the 20 rare-action cases this exact sentence also appears in training
  (on other episodes of the task).
- alt1: the same action reworded, without naming an arm.
- alt2: a different action in the same scene, so it has no ground truth.

alt1 and alt2 were written after looking at the frames, and only two alt2 sentences occur in training.

## Judge

Qwen3-VL-32B-Instruct sees 16 frames and the instruction and answers completed (yes / partial / no), attempted
and plausible. It was told that which arm acts does not matter. Calibration on the 25 real episodes:

- With the true caption it says completed for 13 of 25 (52 %). It misses small or slow motions: button presses,
  a cap twist, a slow base approach, a wave. Those actions do happen in the footage (checked on contact sheets).
- With another family's caption it says no for 24 of 25 (96 %).

So a ✓ is reliable and a ✕ is weak evidence. The cross columns re-judge clips against the other prompt. If the
model follows its prompt and the judge can tell, "real clips vs alt2 text" should sit well below "real".

## Results

Mean judge score (yes = 1, partial = 0.5, no = 0) over 3 seeds, at N = 1:

| | real | alt1 | alt2 | real clips vs alt2 text |
|---|---:|---:|---:|---:|
| 13 cases where the judge verifies the GT | 0.87 | 0.85 | 0.59 | 0.67 |
| all 25 cases | 0.60 | 0.57 | 0.51 | 0.48 |

- Rewording barely matters. alt1 scores about the same as the original caption.
- The scene drives the action more than the words do. Clips made for the real caption pass the alt2 instruction
  0.67 of the time.
- Off-script requests fall back to the episode's action. Three alt2 prompts score 0 on every seed: turn left
  away from the curtains (base_turn), squeeze the dressing (pour_dressing), and move the guard away from the
  slicer (move_guard). Judged against the original caption, those same clips pass 1.0, 1.0 and 0.67.
- Held-out tasks score highest (real 0.93 over 15 clips).
- twist_cap, press_oven, knead_dough, wave and carry_bag score 0 for every prompt. The judge also fails their
  real footage, so this is unresolved; watch the videos.

Per case at N = 1 (GT = the judge's verdict on the real footage with its true caption):

| case | family | GT judged ✓ | real | alt1 | alt2 | real clips vs alt2 text | alt2 clips vs real text |
|---|---|:-:|---:|---:|---:|---:|---:|
| pour_dressing | pour | ✓ | 0.33 | 0.67 | 0.00 | 0.33 | 1.00 |
| scoop_tapioca | pour | ✓ | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 |
| twist_cap | rotate | no | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| turn_faucet | rotate | ✓ | 1.00 | 0.33 | 0.67 | 1.00 | 0.67 |
| scan_juice | tool_use | no | 0.00 | 0.33 | 0.67 | 0.33 | 0.33 |
| stamp_form | tool_use | ✓ | 1.00 | 1.00 | 0.67 | 0.67 | 0.67 |
| press_oven | press | no | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| press_ricecooker | press | no | 0.67 | 0.67 | 1.00 | 0.67 | 0.33 |
| base_forward | locomotion | no | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 |
| base_turn | locomotion | ✓ | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| carry_bag | transport | no | 0.00 | 0.00 | 0.17 | 0.00 | 0.00 |
| move_guard | transport | ✓ | 1.00 | 0.33 | 0.00 | 0.33 | 0.67 |
| clap | gesture | ✓ | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 |
| wave | gesture | no | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cut_dough | cut_shape | no | 0.33 | 0.00 | 0.00 | 0.00 | 0.33 |
| knead_dough | cut_shape | no | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stir_coffee | mix | ✓ | 1.00 | 1.00 | 0.67 | 1.00 | 0.67 |
| shake_decanter | mix | ✓ | 0.33 | 1.00 | 0.33 | 0.00 | 0.33 |
| untie_curtain | tie | no | 0.67 | 0.67 | 1.00 | 1.00 | 0.33 |
| tie_tieback | tie | no | 0.33 | 0.00 | 0.33 | 0.67 | 0.00 |
| heldout_toast | heldout | ✓ | 1.00 | 1.00 | 0.67 | 1.00 | 0.33 |
| heldout_supermarket | heldout | ✓ | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| heldout_water | heldout | ✓ | 1.00 | 1.00 | 0.67 | 1.00 | 0.33 |
| heldout_curtains | heldout | no | 0.67 | 0.67 | 1.00 | 0.33 | 1.00 |
| heldout_menu | heldout | ✓ | 1.00 | 1.00 | 1.00 | 0.67 | 0.33 |

## Loop depth

The checkpoint was trained with the core blocks looped N = 1–8 times, picked at random per step. Every case,
prompt and seed was rolled out again at N = 2, 4 and 8 with the same noise, and judged the same way.

| N | real | alt1 | alt2 | real clips vs alt2 text | real, all 25 | held-out real | PSNR | LPIPS | s / clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.87 | 0.85 | 0.59 | 0.67 | 0.60 | 0.93 | 15.6 | 0.323 | 31 |
| 2 | 0.95 | 0.74 | 0.62 | 0.59 | 0.61 | 0.87 | 15.7 | 0.312 | 39 |
| 4 | 0.82 | 0.79 | 0.64 | 0.64 | 0.56 | 0.80 | 15.8 | 0.308 | 54 |
| 8 | 0.77 | 0.74 | 0.54 | 0.46 | 0.52 | 0.67 | 15.8 | 0.308 | 84 |

real / alt1 / alt2 / cross are on the 13 judge-verifiable cases. PSNR / LPIPS are real-prompt rollouts vs the
ground truth over all 25 cases. Time is per clip on one A100.

- Deeper loops do not add capability. Each case has 3 seeds, so one clip shifts a case by 0.33, and 0.1 on the
  verifiable set is about 4 clips. N = 2 and N = 4 are within that noise of N = 1. Only the N = 8 drop is consistent
  across prompts.
- N = 8 is worst on the held-out tasks: real falls from 14 to 10 of 15 clips, mostly heldout_supermarket
  (1.0 → 0.33).
- Pixel fidelity barely moves (PSNR +0.2 dB, LPIPS −0.015), and no clip at any depth is judged implausible
  except one at N = 8.
- The off-script fallback stays: the alt2 prompts of base_turn and move_guard score 0–0.33 at every depth.
- N = 8 costs 2.7× the time of N = 1. N = 1 stays the default, as in the seed sweep.

In summary.json the keys are `<case>/<prompt>/N<d>` (per_case, with `cross` = the same clips judged against the
other prompt) and `<family>/<prompt>/N<d>` (per_family).

## Loop depth: prompt understanding, motion and video quality

Each dimension on its own, 900 clips (25 cases × 3 prompts × 3 seeds × N = 1, 2, 4, 8). "real" is the same metric
on the ground-truth frames. ★ = the paired difference to N = 1 (same case, prompt and seed) is outside a 95 %
bootstrap interval over the 25 cases. Metrics: `ldvflux/cap_quality.py` (definitions in its docstring), files
`rlong9500/quality_*.json`, `rating_*.json`, `quality_summary.json`.

**Answer: depth N does not meaningfully improve any of the three.** The weaknesses that matter, visible jumps where
the chained windows join and blur that grows along the chain, are the same at every N.

1. Prompt understanding: no. Judge scores are flat for N = 1–4 and lower at N = 8 (table above). A changed
   instruction (alt2) is ignored at every depth: changed-prompt clips pass the original caption as often as their
   own prompt (gap −0.03 to −0.10). This is expected from the training objective and not a target.

2. Dynamics: no.

| | real | N = 1 | N = 2 | N = 4 | N = 8 |
|---|---:|---:|---:|---:|---:|
| jerk (flow change / motion, lower = smoother) | 0.59 | 0.51 | 0.52 | 0.52 | 0.52 |
| warp error (change not explained by motion) | 1.76 | 1.25 | 1.21 | 1.23 | 1.19 ★ |
| seam (warp error at window joins / around them) | 1.0 | 3.9 | 4.0 | 3.9 | 4.0 |
| motion (camera-compensated flow, px/frame) | 0.46 | 0.53 | 0.51 | 0.54 | 0.52 |
| direction vs the real motion (cosine) | – | 0.03 | −0.01 | 0.01 | 0.04 |
| VLM object dynamics (1–5) | 5.0 | 4.61 | 4.65 | 4.71 ★ | 4.62 |
| VLM arm motion (1–5) | 5.0 | 4.63 | 4.64 | 4.71 ★ | 4.63 |

   Within a window motion is smooth at every N. The joins between the 4 chained windows jump about 4× more than
   the frames around them (real footage 1.0), at every depth. Motion amount is flat and its direction is unrelated
   to the real episode. N = 4 rates +0.09 higher on the VLM, but ratings are saturated (about 73 % of clips get 5)
   and N = 8 returns to the N = 1 level.

3. Video quality: no. Output resolution is 512 × 384 at every N, so detail is measured instead.

| | real | N = 1 | N = 2 | N = 4 | N = 8 |
|---|---:|---:|---:|---:|---:|
| sharpness (Laplacian variance / context frame) | 0.96 | 0.55 | 0.56 ★ | 0.56 | 0.56 |
| sharpness, last window | 0.94 | 0.41 | 0.43 ★ | 0.43 | 0.42 |
| high-frequency share / context frame | 0.96 | 0.61 | 0.61 | 0.61 | 0.61 |
| VLM visual quality (1–5) | 5.0 | 4.72 | 4.75 | 4.78 ★ | 4.74 |

   Rollouts keep about 55 % of the real frame's sharpness, falling to about 42 % by the last window. Depth moves
   this by about 2 %.

N = 1 stays the default. What to work on is the chaining (seams) and the blur build-up, not the loop depth.
Open questions are in `agibot-flux3/TODO.md`.

## Files and reproduction

- `configs/eval_cases_capability.json` lists the cases, with provenance and a leak check: no episode is in
  train / val / test.
- `ldvflux/cap_eval.py` does the rollouts. `ldvflux/cap_judge.py` produces judge.json, metrics.json (PSNR / LPIPS
  vs GT, motion) and summary.json.
- `scripts/capeval/`: case mining. mine.py takes the full Beta task_info and the splits and writes segs.json.
  select.py builds the candidate pool. fetch_eps.py does HTTP Range downloads of the head-camera mp4 of
  never-sampled episodes, with HF_TOKEN in the environment. frames.py makes contact sheets and build_cases.py
  writes the cases file. setup_env.sh / get_weights.sh set up the environment. make_strips.py encodes the
  page videos (one 3-seed strip per case × prompt × depth), and build_capage.py + capage_template.html build the page. Paths for the episode mirror (/dev/shm/capeval) and task_info are
  hard-coded at the top of the scripts.
- Clips and GT videos are not in git (`*.mp4`). Rerun with:

```
cd flux-action && PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_eval --config ../configs/looped_ft_384_random_n1-8.json \
    --ema /dev/shm/w/looped/random_depth_n1-8_long/ema_step-9500.safetensors --cases ../configs/eval_cases_capability.json \
    --out ../results/capeval/rlong9500 --seeds 0 1 2 --depths 1 2 4 8 --text-encoder outputs/weights/text_encoder \
    --mirror /dev/shm/capeval/mirror --shard 0/2 --device cuda:0      # and --shard 1/2 on cuda:1
PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_judge --run ../results/capeval/rlong9500 \
    --cases ../configs/eval_cases_capability.json --judge /dev/shm/w/qwen3vl32b --device cuda:0
PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_quality --run ../results/capeval/rlong9500 --device cuda:0 --shard 0/2
PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_quality --run ../results/capeval/rlong9500 --device cuda:0 --shard 0/2 \
    --rate --judge /dev/shm/w/qwen3vl32b                # and --shard 1/2 on cuda:1 for both
PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_quality --run ../results/capeval/rlong9500 --summary
```

The rollouts take 31, 39, 54 and 84 s per clip on an A100 at N = 1, 2, 4 and 8, so about 7.5 h for all four depths
on two GPUs. The judge takes about 4 s per call (about 1,550 calls with the cross checks).
