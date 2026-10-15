# Motion-aware evaluation of the AgiBot FLUX 3 Action models — how to run it on another machine

This is the hand-off for evaluating the fine-tuned / looped models **on a different GPU box**. It answers two
questions that the Phase-4 pixel metrics cannot:

1. Does the model move the right things in the right direction (not just keep the background)?
2. When a Looped-DiT runs more loops (N = 1 → 8), does the extra compute go into the static background or into the motion?

Everything below needs one GPU with ≥ 40 GB (A100/H100 class; the 7B DiT in bf16 is 13.6 GB, the Qwen3-VL text
encoder 8 GB, RAFT/LPIPS small), ~60 GB of disk for weights, and a Hugging Face token for the two public weight
repos plus the **private** eval-clip dataset (same HF account as the training machine, user `Jinyan0924`).

## 1. What the model does (inputs and outputs)

| | |
|---|---|
| input image(s) | the **k previous head-camera frames** of an AgiBot World Beta episode, k ∈ {1, 9} (also 0, 5, 17 supported), 384 × 512 RGB, 15 fps (every 2nd source frame) |
| input text | the **task prompt**: AgiBot's per-window action annotation, e.g. an imperative sentence naming the object and the arm (“Pick up … with the left arm.”) |
| output | the **next 33 frames** (2.2 s) at 384 × 512, 15 fps |
| sampler | Euler, 30 steps, classifier-free guidance 4 (empty prompt as the null), timestep shift 5, fixed noise seed per window |
| looped models | the loop depth N is a sampling argument (`--loops N` / `--depths …`); N = 4 is the training depth |

The ground-truth continuation exists for every evaluation window (the real frames that follow), so pixel and flow
metrics are computed against it. Alternative prompts (same frames, different instruction) have no ground truth and
are compared with the real-prompt output instead.

## 2. The evaluation cases

**Gallery set (8 windows, `configs/eval_cases_gallery8.json`).** The first 8 fixed test windows of the Phase-4/5
galleries (`fixed/test_seed0_n1000.json`, picks 0–7, in the eval-splits dataset). Head camera nearly static in all
8 (≤ 0.24 px/frame), so the background is a copy of the context; motion is "hard" (direction changes within the
clip) in cases 01, 05, 07, "easy" (one direction) in 02, 03, 04, 08, nearly absent in 06. Each case has the real
prompt (resolved at run time from the dataset; AgiBot's annotation) and two **alternative prompts** we wrote:
same scene, a different action and/or the other arm (e.g. case 07: real "slide the shorts upward" → "downward" /
"with the right arm"). The file lists episode id, anchor frame and the alternative prompts.

**Statistical set (≈ 50 val windows).** `scripts/select_motion_cases.py` scans the fixed val picks and keeps the
windows with a static camera (≤ 0.3 px/frame), a moving region ≥ 5 % of the pixels and object motion ≥ 1.5
px/frame, labelled hard / easy by whether the motion direction changes. Run it once (CPU is fine, ~25 s per window
with RAFT-large; pass `--device cuda:0` for seconds):

```bash
python scripts/select_motion_cases.py --split val --n-scan 120 --n-keep 50 --mirror $MIRROR --device cuda:0 \
    --out configs/eval_cases_val50.json
```

## 3. Setup

```bash
# code: this directory (agibot-flux3/) + the patched flux-action checkout it expects as ./flux-action
git clone https://github.com/naomili0924/looped-dit-video && cd looped-dit-video/agibot-flux3
git clone https://github.com/black-forest-labs/flux-action && cd flux-action && git checkout 830f1ce178d7aea52f18fde31d8e8e36ce6bae21
git apply ../patches/flux-action.patch                      # our video-only trainer, Looped-DiT retrofit, FSDP fixes
uv venv .venv && uv pip install -e '.[encoders]' lpips torchmetrics av cd-fvd scipy   # python 3.12, torch with CUDA
cd ..
# weights (≈ 25 GB base + 11 GB per EMA)
hf download black-forest-labs/flux-3-action-base --local-dir flux-action/outputs/weights \
    --include 'flux-3-action-base.safetensors' 'video_vae.safetensors' 'text_encoder/*' 'config.json'
hf download Jinyan0924/flux3-agibot-videoft ema_step-6000.safetensors --local-dir weights/ft            # Phase-3 fine-tune
hf download Jinyan0924/flux3-agibot-looped  ema_loop2500_step-2500.safetensors --local-dir weights/loop # looped, 2500 steps
# (the 5000-step looped EMA appears in the same repo as ema_loop5000_step-5000.safetensors when that run has finished)
# evaluation clips + window definitions (private dataset, 14 GB; needs the account's HF token)
hf download Jinyan0924/agibot-flux3-eval-splits --repo-type dataset --local-dir eval-splits
export MIRROR=$PWD/eval-splits
```

The training-time configs point the text encoder at a pre-encoded caption table (`lookup:/dev/shm/...`) that does
not exist elsewhere; the evaluation scripts replace it with the real Qwen3-VL encoder
(`--text-encoder flux-action/outputs/weights/text_encoder`, `TEXT_ENCODER=` for the shell driver), which is also
what alternative prompts need. `--mirror $MIRROR` reads frames from the downloaded clips instead of the streaming
index used on the training box (`data/splits.json` is not in the public repo).

## 4. Run

**A. Loop depth: background vs motion (gallery set, then the val set).** One model at a time, ~40 min per model on
an A100 for the 8 gallery cases (8 cases × 2 k × 3 prompts × 4 depths = 192 samplings), ~4 h for 50 cases:

```bash
MIRROR=$MIRROR TEXT_ENCODER=$PWD/flux-action/outputs/weights/text_encoder \
  scripts/run_depth_eval.sh loop2500 configs/looped_ft_384_r1.json $PWD/weights/loop/ema_loop2500_step-2500.safetensors configs/eval_cases_gallery8.json cuda:0
MIRROR=$MIRROR scripts/run_depth_eval.sh ft6000 configs/video_ft_384_r3.json $PWD/weights/ft/ema_step-6000.safetensors configs/eval_cases_gallery8.json cuda:0   # plain model: depth 1 only
MIRROR=$MIRROR scripts/run_depth_eval.sh base    configs/video_ft_384_r3.json "" configs/eval_cases_gallery8.json cuda:0
# then the same three with configs/eval_cases_val50.json and OUT=results/depth_eval_val50
```

Outputs per model in `results/depth_eval/<name>/`: `clips/<case>/k<k>/{gt,context,p<i>_N<d>}.mp4`, `analysis.json`,
`strips/<case>_k<k>_p<i>.mp4` (GT | N=1 | N=2 | N=4 | N=8 side by side); across models `results/depth_eval/DEPTH.md`.

**B. Motion metrics on train / test / test_unseen (50 windows each, k = 1 and 9).** Generation with clips saved,
then the offline analysis (`results/motion/MOTION.md`):

```bash
scripts/run_motion_gen.sh 50            # edit MODELS / add --mirror as needed; two GPUs by default
scripts/run_motion_metrics.sh cuda:0
```
(`train` needs the streaming index of the full dataset and is only available on the training box; on another
machine run `test` and `test_unseen` from the mirror.)

## 5. What the numbers mean

| metric | computed on | reading |
|---|---|---|
| PSNR bg / PSNR moving | background / moving pixels (moving = changed > 25/255 vs the first future frame in the real clip) | compare with the static baseline (copy the first frame): a model has to beat it on the moving pixels to have predicted any motion |
| dir cos | RAFT flow of output vs real clip, camera motion removed, moving pixels only | 1 = same direction, 0 = no or unrelated motion (chance level ≈ 0.1), negative = opposite |
| mag ratio | same | 1 = moved the right amount; the Phase-3 model is ≈ 0.6 on the gallery set |
| region cos | cosine of the mean motion vectors of the moving region | robust to position offsets |
| delta in / out, concentration, mass in | \|output(N') − output(N)\| inside / outside the moving region | concentration > 1 and mass in > area in: the extra loops work on the motion, not the background |
| motion, LPIPS / dir cos vs real prompt | alternative prompts | dir cos ≈ +1 means the prompt was ignored; ≈ 0 or negative means the instruction changed the motion |

Report the DEPTH.md / MOTION.md tables and the strips; no conclusions are expected from 8 cases alone.

## 6. Reference numbers so far (training box, 8 gallery windows, k = 1, RAFT-large)

| model | dir cos | mag ratio | region cos |
|---|---:|---:|---:|
| chance (another window's real motion) | 0.10 | 0.18 | 0.02 |
| base, zero-shot | 0.18 | 1.02 (flicker) | 0.17 |
| Phase-3 fine-tune | 0.29 | 0.62 | 0.39 |
| looped 2500, N = 1 / 2 / 4 / 8 | 0.30 / 0.36 / 0.41 / 0.35 | 0.53 / 0.71 / 0.82 / 0.60 | 0.40 / 0.43 / 0.46 / 0.35 |

Licence: AgiBot World Beta is CC BY-NC-SA 4.0 (the eval clips stay private); the FLUX weights are under the FLUX
Kommunity License (non-commercial).
