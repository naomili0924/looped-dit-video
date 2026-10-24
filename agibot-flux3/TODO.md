# TODO for the next session: what does loop depth N improve?

The question: for the looped FLUX 3 Action model (random-depth N 1–8 long run), what does sampling with more loop
iterations (N = 1, 2, 4, 8) improve, and what does it not? Answer it per dimension (prompt understanding, dynamics,
video quality), with evidence, from the three sources below plus the numbers in this repo.

## Sources to analyze

| source | what it holds | how to read it |
|---|---|---|
| Capability eval page, https://claude.ai/artifact/4Nh7eAyynXM677TkTqCwwY | 25 never-trained cases (20 rare-action episodes, 5 unseen tasks) × real / reworded / changed prompt × 3 seeds × N = 1, 2, 4, 8, step 9500. Depth selector, judge scoreboard, a 3-seed video strip per case × prompt × depth | private claude.ai artifact owned by the repo owner: open it with the Artifact tool, `action: "read"` (and `list_files` / `read_file` for videos). Same numbers in `results/capeval/` |
| Seed sweep, https://naomili0924.github.io/looped-dit-video/seed-sweep/ | 8 test-split cases × 3 prompts × 5 seeds × N = 1, 2, 4, 8, 16, at steps 3000–9500: grid videos, LPIPS/PSNR to GT, spread over seeds, motion, between-seed diversity | public GitHub Pages; findings in `results/seedsweep/README.md`, numbers in `results/seedsweep/rlong<step>/summary.json` |
| Caption Atlas, https://claude.ai/artifact/4EQ5Qwet6vPpWvr4roGAxf | caption statistics of the 160,120 training windows: action families (pick+place 46 %, fold 8.6 %, long tail), which captions occur in training | private claude.ai artifact, Artifact tool `read`. Use it to tell "trained wording" from "new wording" when reading the other two |

Also on GitHub Pages: https://naomili0924.github.io/looped-dit-video/capability/ (Changed Prompts and Loop Depth: prompt examples, per-depth dynamics / quality tables, where depth helps or hurts) and
https://naomili0924.github.io/looped-dit-video/loop-depth/ (test-split PSNR / SSIM / LPIPS / FVD per depth for the
earlier fixed-depth runs).

## What is already concluded (2026-10-24, capability eval, step 9500)

Details and tables: `results/capeval/README.md`, sections "Loop depth" and "Loop depth: prompt understanding, motion
and video quality".

- Prompt understanding: not improved. Judge scores flat N = 1–4, lower at N = 8 (real prompt 0.87 / 0.95 / 0.82 / 0.77
  on the 13 judge-verifiable cases; held-out tasks 0.93 → 0.67 at N = 8). Changed prompts are ignored at every N; the
  owner says this is expected from the training objective and is not a goal.
- Dynamics: not improved. Within-window smoothness flat (jerk 0.51–0.52); window joins jump about 4× at every N
  (real footage 1.0); motion amount flat, direction unrelated to GT at every N. VLM object-dynamics / arm-motion
  ratings +0.09 at N = 4 only (saturated scale).
- Video quality: not improved. Resolution fixed (512 × 384); sharpness about 0.55 of the context frame (real 0.96),
  0.42 by the last window, moved about 2 % by depth. VLM visual quality +0.06 at N = 4 only.
- Cost: 31 / 39 / 54 / 84 s per clip at N = 1 / 2 / 4 / 8 on an A100. N = 1 is the default.

## Open items

0. Depth-sensitive patterns (per-case analysis, README "Where depth helps and where it hurts"): N = 8 clearly hurts
   the unseen tasks (1-15 matched wins/losses vs N = 1; the supermarket arm under-reaches). N = 2 may help small
   contact actions (press / rotate / pour, 8-2, p = 0.11) but it is not visible in the frames and the judge is weak
   on presses. Confirm with more seeds (e.g. 10 seeds × N = 1, 2 on press_oven, turn_faucet, pour_dressing, and on
   new press/rotate episodes from the rare-action pool, `scripts/capeval/select.py`), and judge with a closer look
   (more frames around the contact, or crops), before claiming that depth helps fine contact.

1. Cross-check the capability conclusions against the seed sweep: does any metric there (LPIPS to GT, spread over
   seeds, between-seed diversity, motion) change with N at any training step, and is the step-9500 picture the same
   as at steps 3000–8000? Check whether N = 16 (seed sweep only) breaks anything.
2. Use the Caption Atlas to split capability cases by how common their action family and exact caption are in
   training, and check whether depth helps rare actions more than common ones (it does not look like it, but the
   per-family numbers in `results/capeval/rlong9500/summary.json`, keys `<family>/<prompt>/N<d>`, were not analyzed).
3. Per-frame plots of warp error and sharpness (window seams at frames 33/66/99, blur build-up), N = 1 vs N = 8 vs
   real: `python -m ldvflux.cap_quality --series --shard i/2` writes `series_<i>.json`; it was started and stopped
   before finishing, so nothing is committed. Add the plots to `docs/capability/`.
4. The VLM ratings are saturated (about 73 % of clips get 5/5). If a finer quality signal is needed, use pairwise
   comparisons (N = 1 vs N = 8, same case and seed) instead of absolute 1–5 ratings.
5. The seams and the blur, not depth, are the main weaknesses. Worth testing: overlap / more context frames between
   chained windows, and whether blur comes from the VAE decode or from the sampler.

## Rebuilding the data (the VM that produced it is gone)

Videos (`*.mp4`) are not in git, except the Pages videos under `docs/`. To regenerate:
`scripts/capeval/setup_env.sh`, `scripts/capeval/get_weights.sh` (base model, the step-9500 EMA from
`Jinyan0924/flux3-agibot-looped/random_depth_n1-8_long`, Qwen3-VL-32B judge), episode mirror via
`scripts/capeval/fetch_eps.py` (needs `HF_TOKEN`), then the commands at the end of `results/capeval/README.md`.
All judge answers and metrics are committed as JSON in `results/capeval/rlong9500/`, so the analysis above needs no GPU.
