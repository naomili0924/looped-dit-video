# Action eval: what the model learned about actions

The capability eval (`results/capeval`) asked whether the model does an episode's own action, a reworded one, or one
changed action. This eval is indexed on actions. It takes every action AgiBot World Beta has, tests each one the
same number of times, and gives each initial window 6 different actions, so we see whether the model can do several
actions from the same start, or only the scene's usual one.

Checkpoint `random_depth_n1-8_long` EMA step 9500, k = 1 context frame, 4 chained windows (8.9 s at 15 fps),
Euler 30 steps, CFG 4, shift 5, loop depth N = 1, 2, 4, 8, several noise seeds per probe.

## The questions

1. **A trained action on a new object.** If training had "squeeze the ketchup bottle", can the model squeeze a
   sponge, a bag of chips or a salad dressing bottle? These probes use a verb that occurs in the training captions
   on an object it never occurs with there (verb × object = 0 training windows).
2. **An action training never had.** Verbs no training caption contains in any form (tilt, tap, poke, toss, drop,
   crumple, squash, spin, knock over, rub, pat, wring, crush, pinch, bend, tear, point at, flick, bounce). Failing
   here is expected. They are tested and reported in their own group, never mixed into the other numbers.
3. **A trained action + direction on any object.** Verb × direction pairs that occur in training (push aside,
   slide up / down, fold to the left, carry forward, turn the base left / right, ...), asked on an object that never
   had that verb with that direction.
4. **Many actions from one start.** Each window gets 6 prompts: its own caption (the control, which the real
   footage shows) and 5 probes. How many of the 6 the model completes is reported per window.

## The action index

`action_index.json`, built by `scripts/actioneval/build_index.py` with the Caption Atlas parser (`scripts/capeval/parse.py`,
`lexicon.py`):

- **Seen actions:** the 100 verb primitives (18 families). The full release (1,001,585 labelled segments,
  3,553 captions) has no verb that the training captions lack, so every release action is a seen action. 88 of
  them are probed. The other 12 are not self-contained instructions from a start frame ("use", "coordinate",
  "release", "adjust", "tidy", "align", "leave", "receive", "move arm") or are probed with a direction instead
  (move base, turn base, step).
- **Seen action + direction:** 29 verb × direction pairs that occur in training.
- **Unseen actions:** 19 verbs absent from every training caption. The check is on raw words, so "waving" (121
  windows) makes "wave" seen even though the lexicon has no entry for it.

Every probe is labelled with the Caption Atlas exposure ladder, computed by parsing the probe's prompt:
L0 exact caption, L1 seen verb + object, L3 new direction for this verb + object, L4 new object with the verb seen on
this material, L5 verb never on this material, L6 verb never in training.

## Initial windows (zero-shot)

`pool.json` (`scripts/actioneval/pool.py`) holds 170 windows from 91 tasks. They are episodes of the val, test and
test_unseen splits (never trained; test_unseen holds the 5 held-out tasks), at most 6 per task. Each window starts
at the first labelled segment of its episode, with the robot in its start pose and nothing grasped, so many actions
are possible. A window's objects are the objects named in the episode's own captions that Qwen3-VL-32B sees in the
start frame, plus the other objects it lists there with a material class (`ldvflux/act_vlm.py --mode scene`).

## From index to cases

1. `propose.py`: every action of the index on every scene object its template can act on, at most 2 objects per
   window and action. The prompt is re-parsed, and a candidate is kept only if it parses to the intended verb
   (and direction) and is new in the intended way (pair = 0 for question 1, verb × object × direction = 0 for 3).
2. `ldvflux/act_vlm.py --mode feasible`: the VLM looks at the start frame and keeps a prompt only if its object is in
   view and the action is physically possible on it.
3. `select.py`: balanced selection. Each action gets the same number of probes (`--per-action`, in different tasks
   where possible). Actions with the fewest feasible windows are served first. A window gets 5 probes with distinct
   verbs, plus its own caption. Actions short of the quota are listed in `configs/eval_cases_action.json` (`short`).

## Scoring: the VLM judge only

`ldvflux/act_judge.py` runs the capability eval's judge, unchanged: Qwen3-VL-32B-Instruct sees 16 frames (the real
start frame first) and the instruction, and answers completed (yes / partial / no), attempted and plausible. There
are no pixel metrics: the ground truth shows one action and the probes ask for others. The real footage is judged
against all 6 prompts of its window:

- on its own caption, this is the judge's recall;
- a "yes" on a probe means the robot's real action already satisfies that probe (or the judge is lenient on it). Such
  a probe cannot separate following the prompt from doing the usual action, so it is flagged (`gt_satisfies`) and left
  out of the clean group numbers.

`report.py` gives the score (yes 1, partial 0.5, no 0; seeds averaged per probe, then probes averaged) per group, level,
family, action and depth N, with 95 % bootstrap intervals over probes, seed agreement, and the prompts-done-of-6 per
window.

## Running it

```
scripts/actioneval/run.sh                                   # every step; each skips work already done
STEPS="rollout judge report" DEPTHS="2 4 8" scripts/actioneval/run.sh
PER_ACTION=5 SEEDS="0 1 2 3 4" scripts/actioneval/run.sh    # a bigger balanced set
```

Weights come from `scripts/capeval/get_weights.sh` (into `/dev/shm/w`) plus `Qwen/Qwen3-VL-32B-Instruct` in
`/dev/shm/w/qwen3vl32b`. The environment comes from `scripts/capeval/setup_env.sh`. Episode mp4s go to
`/workspace/loop_dit_eval/data/actioneval/mirror`, about 3.4 GB. Clips are not in git (`*.mp4`); every judge answer and
the summary are.

## Results

(filled in by the run)
