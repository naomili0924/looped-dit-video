# NOTES_h100 — eval on 2×H100 PCIe (80 GB)

## 2026-10-16 00:08 UTC — setup
- repo cloned at fd8eaa3; flux-action @ 830f1ce + patches/flux-action.patch applied cleanly.
- venv (py3.12, uv) installing in tmux `setup` -> logs/setup_venv.log.
- public EMAs (ema_step-6000, ema_loop2500_step-2500) downloading in tmux `dl_ema` -> logs/dl_ema.log.
- loop5000 EMA not present in Jinyan0924/flux3-agibot-looped at this time.
- BLOCKED: HF_TOKEN and GH_TOKEN are not set in the agent shell (not in env, /workspace/.env, /etc/environment, login shell).
  Anonymous access: black-forest-labs/flux-3-action-base files -> 401 (gated); Jinyan0924/agibot-flux3-eval-splits -> 401 (private).
  Stopped per instructions; waiting for tokens.
- 2026-10-16 00:22 UTC: GH token received (stored in /workspace/.env, mode 600); push access to naomili0924/looped-dit-video verified. HF token still missing -> still blocked on base weights + eval dataset.
- 2026-10-16 00:22 UTC: HF token received (user=Jinyan0924, stored in /workspace/.env mode 600). eval-splits dataset: 200, downloading (tmux dl_data). flux-3-action-base: 403 'restricted and you are not in the authorized list' -> blocked (VAE + text encoder + base DiT all come from that repo).
- 00:31 UTC: hf 1.33 'download <repo> a b --include ...' treated quoted names as files and ignored --include -> DiT safetensors skipped; fixed by downloading flux-3-action-base.safetensors by name (size 13972274512 matches).
- 2026-10-16 00:31 UTC: launched tmux gpu0 (logs/gpu0_chain.sh: part A gallery x3 -> select val -> part A val x3) and gpu1 (logs/gpu1_chain.sh: run_motion_gen.sh 50 GPUS=1 -> run_motion_metrics.sh cuda:1). Fixed run_motion_metrics.sh PYTHONPATH (/workspace/loop-dit hard-coded -> $PWD).
- 00:34 UTC: attempt 1 failed at once on both GPUs: ModuleNotFoundError natten (EVAL_MOTION §3 'uv pip install -e' does not pull NATTEN; it is outside uv.lock per flux-action docs/setup.md). Fix: uv pip install 'natten==0.21.6+torch2110cu128' -f https://whl.natten.org/ (env resolved torch 2.11.0+cu128; training box had torch 2.10.0 + natten +torch2100cu128 on 2xA100 -> VAE NATTEN backend here is Hopper vs A100 there; small numeric drift possible). Failed logs in logs/attempt1_natten/.
  select_motion_cases.py succeeded during attempt 1 (RAFT only): mirror has only val_seed0_n40.json -> 40 picks scanned (note text says 120), kept 20 (11 hard / 9 easy) -> configs/eval_cases_val50.json has 20 cases, not 50.
- 00:34 UTC: relaunched tmux gpu0 + gpu1 (attempt 2); gpu0 chain now aborts if a model's analysis.json is missing.
- 00:36 UTC: gpu0 attempt 2 failed: run_depth_eval.sh used relative PY=flux-action/.venv/bin/python after 'cd flux-action' -> not found. Fixed to $PWD/...; relaunched gpu0 (attempt 3). gpu1 motion_gen running since 00:34 UTC.
- 00:44 UTC: measured loop2500 gallery sampling on H100 PCIe (k=1, batch 1): ~6.5 s N=1, 6.4 s N=2, 8.8 s N=4, 13.5 s N=8 per window. Built a private raw-strip page for test01_toilet (claude.ai artifact, frames not in git).
- 00:49 UTC: part B attempt 2: base test_k1 generated + saved 50 clips, then crashed in evaluate._save_grid (ModuleNotFoundError imageio) before writing test_k1.json; driver moved on to ft_step6000. Fix: uv pip install imageio imageio-ffmpeg. Log: logs/motion_gen_gpu1.attempt2_imageio.log.
- 00:49 UTC: reorder per user: k=1 first everywhere, k=9 later. Added K env override (default "1 9") to run_depth_eval.sh and run_motion_gen.sh. depth_eval skips existing clips and seeds by case index (k-independent), and the window anchor is the first future frame, so a K=1 run gives the same k=1 clips as K="1 9". Killed + relaunched gpu0 (71 loop2500 gallery clips kept, all 33 frames, verified) and gpu1 at 00:49 UTC. New chains: logs/gpu0_chain.sh, logs/gpu1_chain.sh.
- 01:01 UTC: gallery loop2500 k=1 done (resumed run 00:49-01:01). Sanity vs EVAL_MOTION §6 (k=1, 8 cases): dir cos N1/2/4/8 = 0.252/0.261/0.198/0.146 here vs 0.30/0.36/0.41/0.35 reference; mag ratio 0.596/0.734/0.751/0.686 vs 0.53/0.71/0.82/0.60; region cos 0.253/0.348/0.234/0.204 vs 0.40/0.43/0.46/0.35. Outside the reference at N=4/8 -> to check (env differences: torch 2.11 vs 2.10, NATTEN hopper vs A100 kernel, real Qwen3-VL text encoder vs training-box caption lookup table). Gallery page (private): https://claude.ai/artifact/AqoexkPcyKo2idYkmcdiTH
- 01:29 UTC: user request: case 04 vacuum with handover prompts (configs/eval_cases_vacuum_handover.json), loop2500 k=1 N=1/2/4/8, --seed 3 to match gallery noise, real-prompt clips copied in (skipped). tmux handover on cuda:0 alongside the chain.
- 01:34 UTC: handover run done 01:32 UTC (~4 min incl. load, alongside chain). Seed check: regenerated p0_N1 bit-identical to gallery (PSNR inf). analysis -> results/depth_eval_handover/loop2500/analysis.json. Added as 'handover 1/2' prompts on case 04 of the gallery page (v3).

## Autoregressive rollout (user request, 2026-10-16 04:05–04:14 UTC)
- New `ldvflux/rollout.py`; launcher `logs/rollout.sh`. 8 gallery cases, real prompt, k=1, 4 chained windows (133 frames = 8.8 s at 15 fps).
  Window 1 = real context + depth_eval noise (seed + case index; mean |diff| vs the gallery clip 0.9/255); window r uses the last
  generated frame as context, seed + case + 1000 r. Models: loop2500 N=1,4 (cuda:0), ft6000 then base (cuda:1). Outputs results/rollout/<model>/.
- Time per 4 windows: ~19.5 s N=1 / 33.5 s N=4 (loop2500), ~21 s ft6000 and base. Wall clock ~9 min. No crashes.
- GT shorter than 8.8 s where the episode ends: vacuum 5.3 s, doorknob 6.1 s, shorts 7.3 s.
- Both chains (gpu0 part A k=1+9, gpu1 part B k=1+9 + metrics) had finished at 02:32 / 03:34; GPUs were idle until 04:05.
- Rollout pass 2 (04:18–05:16 UTC): loop2500 N=8 and N=16 on real + alt1 + alt2 prompts; N=1,4 / ft6000 / base on alt1 + alt2.
  rollout.py gained --prompts and --tag (manifest_<tag>.json, parallel jobs share --out); clips renamed p<prompt>_N<d>.mp4.
  Time per 4 windows: N=8 ~52 s, N=16 ~100 s (loop2500). No crashes. 24 strips (8 cases x 3 prompts) on the rollout page.
- Rollout pass 3 (05:37–05:44 UTC): loop2500 N=2, all 8 cases x 3 prompts, cases split over both GPUs (rollout2.sh ONLY=...). ~22 s per 4 windows. Strips now 8 columns.

## Sigmoid deep-supervision weighting (user request, 2026-10-16)
- flux-action: `deep_supervision: "sigmoid"` + `deep_supervision_center` (None = (N+1)/2) + `deep_supervision_temperature`
  (default 1.0) in PolicyConfig; `deep_supervision_weights(N, "sigmoid", center, T)` in models/looped.py; policy.py passes
  them. Default stays final_plus_mean (existing configs unchanged). patches/flux-action.patch regenerated (diff vs the old
  patch = only these lines). N = loop.num_loops (4), configurable as before.
- Weights at N=4 (sum 2): T=0.5 0.047/0.269/0.731/0.953; T=1 0.182/0.378/0.622/0.818; T=1.5 0.269/0.417/0.583/0.731.
- Not trained yet. Training here needs data/splits.json (train split) and the caption lookup table
  (/dev/shm/agibot/captions.pt); neither is on this box (only the eval-splits mirror).

## Random loop depth in training (user: "go with random N 1-8, implement it"), 2026-10-16
- flux-action (patch regenerated): PolicyConfig.loop_sampler {"min","max","seed","grad_loops"} + monotone_beta.
  Per micro-batch N ~ uniform(min..max), drawn from (seed, update, micro-batch) set by the trainer (`policy.loop_draw`)
  -> same N on every rank, reproducible on resume. Exit r weight = w*(r) / P(exit r supervised), w* =
  deep_supervision_weights(max, deep_supervision, ...) (use "uniform" for flat), so E[weight of exit r] = w*(r).
  grad_loops = k: first N-k loops under no_grad (pre-loop stage gets no grad on such steps); requires k >= min.
  monotone_beta: beta * sum relu(L_{r+1} - sg L_r). Eval keeps loop.num_loops. Trainer logs "loops" per update.
  Fixed-N deep supervision unchanged (same weights as before).
- scripts/test_random_depth.py (CPU): fixed-N = old weights; expectation = w* for (1..8), (2..8), (1..2) x k in
  {None,1,2,4}; draw deterministic and uniform; truncated backprop grads/values. Passes.
- Eval "mismatch" re-examined: pixel metrics (PSNR/SSIM/LPIPS, part B) match the training box within ~0.1-0.4 dB;
  gallery-8 ft6000 dir cos 0.285 vs ref 0.29, mag 0.62 vs 0.62. The training box's own preview run on the same 8
  cases (results/depth_eval_preview) gave loop2500 k=1 dir cos 0.18/0.25/0.22/0.21, not the EVAL_MOTION §6
  0.30/0.36/0.41/0.35 -> 8-case flow metrics swing ~0.1-0.2 between runs; ours (0.25/0.26/0.20/0.15) is in range.
- Caption table: scripts/encode_captions.py on eval-splits/splits.json (has the train split) -> /dev/shm/agibot/captions.pt.
- Smoke test 1 (06:36 UTC, windows_per_rank 4 x accum 4): 2 updates ran (step 1 96 s incl. compile, step 2 26 s,
  draws [8,5,4,3], [3,1,1,5]; peak 54.7 / 76.2 GB), then CUDA OOM at step 3 (VAE encode of the prefetched batch).
  The N=4 run already peaked at 75.9 GB on A100; N=8 roughly doubles stored activations. Fix 1: windows_per_rank 2
  x grad_accumulation 8 (same 32 windows/update, 8 draws/update). Reduce-guard noted 21 no-gradient post_backward
  calls (informational; cross-rank order check passed).
- Smoke test 2 (2 x 8): 20 updates OK, 21.6 s/update mean (first 114 s incl. compile), peak 72.9-75.3 GB.
- Stress test (every micro-batch k=9, N=8, 6 updates): ~37 s/update, peak 75.2 GB -> fits.
- FULL RUN launched 07:10:47 UTC: scripts/launch_looprand.sh -> /dev/shm/runs/looprand, tmux `train`, log
  logs/looprand.log. Steps 10/20 at ~21-23 s/update; ETA ~15 h (~22:30 UTC Oct 16) plus eval every 500 steps.
  compile_blocks on (as on A100). Checkpoints/EMA in /dev/shm (lost on a container restart; not backed up yet).
- Hub backup (user: upload checkpoints to HF under a new folder): scripts/hub_backup.py gained --prefix / --keep-all /
  --squash (default unchanged without --prefix). Sidecar tmux `hub`: --run /dev/shm/runs/looprand --repo
  Jinyan0924/flux3-agibot-looped --prefix random_depth_n1-8 --keep-all, polls every 300 s; uploads
  random_depth_n1-8/ema_step-<N>.safetensors (bf16, 11 GB) + metrics.jsonl, eval/val.jsonl, config.json for every
  complete 500-step checkpoint; never deletes, no history squash. Log logs/hub_looprand.log.
- 21:27 UTC: post-train chain armed (tmux post, scripts/launchers/posttrain_looprand.sh, logs/posttrain.log): waits for
  step-2500 COMPLETE + idle GPUs -> consolidate EMA to weights/looprand/ema_step-2500.safetensors (verified on step-2000:
  395 tensors, same keys/shapes as loop2500) -> copy step-2500 to ckpt_backup/looprand (RAM -> disk) -> rollouts
  N=16,1 (gpu0) / 8,4,2 (gpu1), prompts 0 1 2 -> motion gen k=1 test+test_unseen 50 windows, looprand N=1,2,4,8 +
  loop2500 N=2,8 (same seed/windows as results/motion_gen) -> run_motion_metrics.sh.
- 21:50 UTC: copied pre-cooldown checkpoint /dev/shm/runs/looprand/step-2000 (LR still 1e-5) to ckpt_backup/looprand/step-2000
  (83 GB) before keep_checkpoints=1 deletes it. User will run the long fine-tune later themselves ("until no improvement");
  no long run launched here. Resume recipe: same config, output_dir new, resume=<that step dir>, steps/cooldown_start moved.
- 21:57 UTC: /workspace is NOT a persistent volume (vast-capabilities workspace_is_volume=false) -> full resumable
  checkpoints mirrored to the Hub, never deleted: scripts/hub_full_ckpt.py (tmux fullckpt, logs/hub_full_ckpt.log) ->
  Jinyan0924/flux3-agibot-looped/random_depth_n1-8/checkpoints/step-2000 (uploaded 99 s, 13 files 88.9 GB, sizes verified);
  step-2500 queued (waits for COMPLETE). Restore on any machine: scripts/restore_full_ckpt.sh <step> <run dir>.
  Fix attempt 1: list_repo_tree raises EntryNotFound for a folder not yet on the Hub -> caught.
