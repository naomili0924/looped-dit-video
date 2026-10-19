# How we evaluate a looped checkpoint

Two evaluations, each a web page with videos. Both are published on GitHub Pages at
**https://naomili0924.github.io/looped-dit-video/** (source: `docs/` in the public repo, built by `scripts/build_pages.sh`).

| say | page | what it measures | cost (2 × H100) | script |
|---|---|---|---|---|
| **normal eval** | [Trained Loop Depth](https://naomili0924.github.io/looped-dit-video/loop-depth/) | test split, 100 windows, k = 1 and 9, loop depth N = 1 2 4 8: PSNR / SSIM / LPIPS, FVD (300 windows), VLM judge (100), validation curve; strips of 8 fixed test windows GT \| base \| fine-tuned \| N = 1 \| 2 \| 4 \| 8 | ≈ 2.5 h | `scripts/normal_eval.sh` |
| **long-rollout eval** | [Long Rollout](https://naomili0924.github.io/looped-dit-video/long-rollout/) | zero-shot: one real frame + one prompt, 4 chained 33-frame windows (8.8 s), 8 cases × 3 prompts × N = 1 2 4 8 16, next to GT / base / Phase-3 fine-tune | ≈ 35 min | `scripts/run_long_rollout.sh` |

Models evaluated so far (registry: `configs/models.json`; weights on the Hub, `Jinyan0924/flux3-agibot-looped`):

| tag | step | trained N | weights | normal eval | long rollout |
|---|---|---|---|---|---|
| `loop2500` | 2500 | 4 | `ema_loop2500_step-2500.safetensors` | yes | no (user decision) |
| `loop4500` | 4500 | 4 | `ema_loop4500_step-4500.safetensors` | yes | yes |
| `rand2500` | 2500 | 1–8 (random per step) | `random_depth_n1-8/ema_step-2500.safetensors` | not yet | yes |

Results text: `results/RESULTS.md` (all splits), `results/<tag>_loops<N>/summary.json`, `results/rollout/<tag>/times.json`.
Lab notebook with the readings: `NOTES.md`.

## Prerequisites (the training box has all of this)

- The repo at `/workspace/loop-dit` with `flux-action/.venv` (uv sync + NATTEN, see `docs/setup.md` in flux-action),
  `PYTHONPATH=/workspace/loop-dit`.
- Weights: Phase-3 EMA `/dev/shm/ema/ema_ft384_step-6000.safetensors` (Hub `Jinyan0924/flux3-agibot-videoft`,
  `ema_step-6000.safetensors`), Qwen3-VL text encoder in `flux-action/outputs/weights/text_encoder`, FLUX 3 Action base
  weights for the base column.
- Data: the streaming index `data/splits.json` (HTTP range reads from the AgiBot World Beta tars on the Hub) or a local
  snapshot of the private eval clips `Jinyan0924/agibot-flux3-eval-splits` (`MIRROR=<dir>` for the rollout).
- `HF_TOKEN` in the environment of the launch command only (tokens are never written to disk). Two GPUs with ≥ 40 GB.
- `tmux` (every job runs in tmux sessions; the launchers print the session names and the log files).

## normal eval

```bash
HF_TOKEN=... scripts/normal_eval.sh <tag> <step> <config> <ema> ["label"] ["trained depths, e.g. 4"]
# e.g. the random-depth model:
HF_TOKEN=... scripts/normal_eval.sh rand2500 2500 configs/looped_ft_384_random_n1-8.json \
    hf://Jinyan0924/flux3-agibot-looped/random_depth_n1-8/ema_step-2500.safetensors "random depth N 1–8, step 2500" 1,2,3,4,5,6,7,8
```

- `<ema>` is a local `.safetensors` or `hf://<user>/<repo>/<path>`; it is fetched to `/dev/shm/ema/ema_<tag>_step-<step>.safetensors`.
- `<config>` is the inference config of the checkpoint (the training config, or an inference-only copy for a model trained
  elsewhere, e.g. `configs/looped_ft_384_random_n1-8.json`). It must load the EMA strictly.
- Runs `scripts/run_loop_eval.sh` (tmux `eval_gpu0`: N = 8 then 1, `eval_gpu1`: N = 4 then 2; `ldvflux/eval_phase4.py`,
  seed 0, Euler 30, CFG 4, shift 5, batch 4) → `results/<tag>_loops<N>/`. Logs `logs/<tag>_eval_gpu{0,1}.log` end with
  `LOOP_EVAL_GPU<g>_DONE`.
- A finisher (tmux `normal_finish_<tag>`) then makes the strips (`scripts/make_looped_strips.sh <tag>` →
  `results/loop_videos_<tag>/`), rebuilds the site (`scripts/build_pages.sh`) and prints `NORMAL_EVAL_<tag>_DONE`.
- The page (`scripts/build_loop_gallery.py --tags <all tags with results>`) shows every model with a complete
  normal eval; the training-depth rows are shaded.

## long-rollout eval

```bash
HF_TOKEN=... scripts/run_long_rollout.sh <tag> <config> <ema> ["0 1"]        # env: STEP, LABEL, TRAINED (defaults: from the EMA name, "looped, step <STEP>", 4)
# e.g.
HF_TOKEN=... STEP=2500 LABEL="random depth N 1–8, step 2500" TRAINED=1,2,3,4,5,6,7,8 \
    scripts/run_long_rollout.sh rand2500 configs/looped_ft_384_random_n1-8.json hf://Jinyan0924/flux3-agibot-looped/random_depth_n1-8/ema_step-2500.safetensors
```

- `ldvflux/rollout.py`: k = 1, 4 windows, fresh noise per window (seed 0 + case index + 1000·window), the last generated
  frame is the next context frame; cases and alternative prompts in `configs/eval_cases_gallery8.json`.
- tmux `roll_gpu0` (depths 1 4 16) and `roll_gpu1` (reference columns, then depths 2 8); the GT / base / ft6000 columns are
  generated once into `results/rollout/{gt,base,ft6000}` and reused. Logs `logs/rollout_<tag>_gpu<g>.log` end with
  `ROLLOUT_<tag>_GPU<g>_DONE`; 120 clips in `results/rollout/<tag>/clips/<case>/p<i>_N<d>.mp4`.
- Finisher tmux `roll_finish_<tag>` rebuilds the site and prints `LONG_ROLLOUT_<tag>_DONE`.
- Pause other GPU jobs meanwhile (`kill -STOP <pid>` / `kill -CONT <pid>`) if the rollout should be fast.

## Publishing

```bash
bash scripts/build_pages.sh                       # docs/ of the public repo: index, loop-depth/, long-rollout/ (+ videos, ~40 MB)
GH_TOKEN=ghp_... bash scripts/push_all.sh "agibot-flux3: eval of <tag>"     # pushes both repos; GitHub Pages serves docs/ of main
```

The claude.ai artifact versions of the pages (private, strips stored as artifact assets) are built from the same results:
`scripts/build_loop_gallery.py ... --urls results/loop_gallery_urls.json` and `scripts/build_rollout_page.py ... --urls ...`.

## seed sweep (random-depth long run)

The "Random-Depth Seed Sweep" artifact page (https://claude.ai/artifact/PN27FAM1mM5UsLdamGzzFK): the long rollout of the
`random_depth_n1-8_long` EMAs with 5 noise seeds per prompt (seed s of case ci: noise 10000·s + ci, so seed 0 is the
single-seed Long Rollout noise), N = 1 2 4 8 16, one 5 × 5 seed-by-depth grid video per case and prompt.

```bash
HF_TOKEN=... scripts/run_seed_sweep.sh "9500 4000 5000 9000 7000 6000 8000" cuda:0    # ~5.5 h per step on one H100
python scripts/build_seed_sweep_page.py --steps 3000 4000 5000 6000 7000 8000 9000 9500 --running <step>
```

- `ldvflux/seed_sweep.py` → `results/seedsweep/rlong<step>/clips/<case>/p<i>_N<d>_s<seed>.mp4`; `ldvflux/seed_metrics.py`
  → grids `results/seedsweep/grids/`, `summary.json` (definitions in its docstring) once all seeds of a step exist.
- The page is `results/seedsweep/page/template.html` with its `DATA` line rebuilt; steps without local clips (3000, the
  reference columns) keep the numbers and videos already on the artifact.
