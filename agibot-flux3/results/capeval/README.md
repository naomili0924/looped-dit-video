# Capability eval: rare actions and held-out tasks

The seed sweep (`results/seedsweep`, `docs/seed-sweep`) measures the model on held-out episodes drawn from the same
mix as training. This eval asks a different question: what does the model do on actions it has barely trained on,
on tasks it has never trained on, and when the instruction is reworded or changed?

Checkpoint `random_depth_n1-8_long` EMA step 9500, loop depth N = 1, k = 1 context frame, 4 chained windows
(1 + 4 × 33 frames = 8.9 s at 15 fps), Euler 30 steps, CFG 4, shift 5. 25 cases × 3 prompts × 3 seeds = 225 clips.
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

Mean judge score (yes = 1, partial = 0.5, no = 0) over 3 seeds:

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

Per case (GT = the judge's verdict on the real footage with its true caption):

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

## Files and reproduction

- `configs/eval_cases_capability.json` lists the cases, with provenance and a leak check: no episode is in
  train / val / test.
- `ldvflux/cap_eval.py` does the rollouts. `ldvflux/cap_judge.py` produces judge.json, metrics.json (PSNR / LPIPS
  vs GT, motion) and summary.json.
- `scripts/capeval/`: case mining. mine.py takes the full Beta task_info and the splits and writes segs.json.
  select.py builds the candidate pool. fetch_eps.py does HTTP Range downloads of the head-camera mp4 of
  never-sampled episodes, with HF_TOKEN in the environment. frames.py makes contact sheets and build_cases.py
  writes the cases file. setup_env.sh / get_weights.sh set up the environment, and build_capage.py +
  capage_template.html build the page. Paths for the episode mirror (/dev/shm/capeval) and task_info are
  hard-coded at the top of the scripts.
- Clips and GT videos are not in git (`*.mp4`). Rerun with:

```
cd flux-action && PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_eval --config ../configs/looped_ft_384_random_n1-8.json \
    --ema /dev/shm/w/looped/random_depth_n1-8_long/ema_step-9500.safetensors --cases ../configs/eval_cases_capability.json \
    --out ../results/capeval/rlong9500 --seeds 0 1 2 --depths 1 --text-encoder outputs/weights/text_encoder \
    --mirror /dev/shm/capeval/mirror --shard 0/2 --device cuda:0      # and --shard 1/2 on cuda:1
PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_judge --run ../results/capeval/rlong9500 \
    --cases ../configs/eval_cases_capability.json --judge /dev/shm/w/qwen3vl32b --device cuda:0
```

The rollouts take about 31 s per clip on an A100, so about 1 h on two GPUs. The judge takes about 4 s per clip.
