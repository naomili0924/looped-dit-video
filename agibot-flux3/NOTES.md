# NOTES — FLUX 3 Action video fine-tune on AgiBot World Beta (+ Looped-DiT retrofit)

Running log of decisions, numbers and problems. Newest entries at the bottom of each phase.

## Layout on this box

```
/workspace/loop-dit/
  flux-action/        black-forest-labs/flux-action @ 830f1ce (uv venv in flux-action/.venv)
  looped-dit-video/   naomili0924/looped-dit-video @ b600f69
  scripts/            our scripts (smoke_base.py, ...)
  logs/               all job logs
  NOTES.md
/dev/shm/flux3/weights   base weights (symlinked as flux-action/outputs/weights)
```

## Phase 0 — Environment (2026-10-11)

### Hardware vs. plan

| | Plan assumed | This box |
|---|---|---|
| GPUs | 2x A100 SXM4 80 GB | 2x A100-SXM4-80GB, driver 580.65.06 (CUDA 13.0), cc 8.0 — **as assumed** |
| RAM | ~516 GB | **2015 GB** |
| CPU | 32 cores | **128** |
| Disk | ~2 TB NVMe | **no NVMe for us.** `/` overlay 32 GB; `/dev/shm` tmpfs **251 GB** (RAM-backed, **wiped on restart**). `/workspace` is NOT a persistent volume. |

The user approved using `/dev/shm` for disk. Consequence: the plan's "keep >= 200 GB free" rule cannot hold
(total 251 GB, 24 GB already used by weights) → raised with the user (see Open questions).

### Software

- uv sync --locked --extra encoders --extra data: torch 2.10.0+cu128, transformers 5.16.1, NATTEN 0.21.6+torch2100cu128.
- Extra: bitsandbytes 0.50.2, torchmetrics 1.9.0, lpips, cd-fvd (I3D/VideoMAE FVD), imageio(+ffmpeg).
- System: git-lfs, ffmpeg, tmux, uv were preinstalled.
- `huggingface-cli whoami`: **not logged in, no HF_TOKEN** → [ASK HUMAN].

### Weights

`black-forest-labs/flux-3-action-base` is **not gated**; downloaded via curl to /dev/shm/flux3/weights (25.4 GB):
trunk 13.97 GB, video_vae 2.51 GB, Qwen3-VL-4B text encoder 8.9 GB. Byte sizes match the Hub.

### Smoke test (scripts/smoke_base.py) — PASSED

Video-only DiT (streams `video`, `video_cond`, text): **5.569B params**, 361 tensors loaded, 0 missing, 0 unexpected
(action streams filtered out at load). One A100, bf16, canvas 480x640 → latent 15x20, 1 observed frame + 8 future
latents, Euler 30 steps, shift 5, CFG 4: **12.2 s sampling, peak 22.8 GB**, load 19 s.
Output (logs/smoke/): a coherent continuation of the SO-101 frame (arm moves toward the box, scene stable).

### Facts learned from the code (needed for Phase 1/2)

- DROID recipe: window = 1 observed + 32 future frames at **15 fps**; canvas 544x736; joint VAE encode
  (latent 0 → `x_video_cond`, latents 1..8 → `x_video`). → **F = 32**.
- SO-101 recipe uses the **history profile**: n_obs_steps 8, 2 snapshots in `x_video_cond`, 42 future frames at 30 fps,
  `video_position_fps` 24, and **each snapshot and the future clip are VAE-encoded independently** (`encode_task`,
  "as in collab's task packer") — i.e. the base trunk has seen independently encoded futures.
- The base model **does have a `video_cond` stream** (variable token count, own RoPE time ids, timestep 0, no loss)
  → k context latents go there (no need to prepend to `x_video`).
- Video VAE (KinoVAE): 96 ch, 32x spatial, **4x temporal, causal encoder** (`enc_causal=True`), non-causal decoder;
  chunked encode = 45-frame chunks with 1-frame overlap. Independent encode of 1+4n frames → 1+n latents;
  decoding 8 latents gives 29 frames, 9 latents give 33.
  → **an independently encoded future of 32 frames does not map to 8 latents.** Phase 1 must pick F on the 1+4n grid
  for an independent future encode (e.g. 33 frames = 9 latents), or encode the future jointly with its last context frame.
- RoPE time ids are in 10 ms units (`times_to_ids`); per-token timesteps are supported, so cond tokens at t=0 are native.

## Phase 1 — Data

### AgiBot World Beta layout check (from the public file tree; downloads need auth → HTTP 401)

- 217 tasks; `task_info/task_<id>.json` per task (content check pending access).
- `observations/<task>/<ep_lo>-<ep_hi>.tar`: **1139 uncompressed tars, ~48 GB each, 46.1 TB total**. Each tar holds whole
  episodes with **all cameras (videos/) and per-frame depth PNGs (depth/)**.
- → "sparse checkout of the head camera only" is **impossible**: the smallest unit is a 48 GB tar. Getting 40 episodes
  per task means streaming whole tars and keeping only `head_color.mp4`.
- Transfer estimate: first tar of every task 9.1 TB; smallest tar of every task 3.7 TB (and the small tars may hold
  < 40 episodes). Both are > the 1.2 TB [ASK HUMAN] threshold. Stored footprint would be small (head videos only).

## Open questions for the human

1. HF login + AgiBot Beta license (gated, auto-approve) + a write token for the private backup repo.
2. Beta transfer of 3.7–9 TB for the plan's subset (vs. the 1.2 TB limit).
3. Disk rule: 251 GB tmpfs only; full-FT resumable checkpoint ≈ 55 GB (fp32 master 22 GB + 8-bit Adam 11 GB + fp32 EMA 22 GB).

## 2026-10-11 — HF login

- Token stored in /workspace/.env (chmod 600) and `hf auth login`: user `Jinyan0924`, fine-grained token,
  canReadGatedRepos=true, write to own repos → fine for the private backup repo.
- AgiBotWorld-Beta still returns 403 "not in the authorized list": the license form has not been submitted/approved
  for this account yet → waiting on the human.
- Decision (user): stream training data instead of downloading. Plan: Range-request tar header index once, then
  stream only head-camera videos per window; encode VAE/context latents in the training loop; cache caption embeddings.

## Phase 1 — Data (streaming, per user decision 2026-10-11)

- Access granted (license accepted); Range requests work (HTTP 206).
- **task_info layout differs from Alpha:** key is `label_info` (not `lable_info`); `action_config[]` has
  `start_frame`, `end_frame`, `action_text`, `skill` (30 fps frame indices). 217 tasks, 168,056 episodes,
  1.0M segments (length p10/p50/p90 = 105/222/464 frames).
- **Tar layout:** `<ep>/videos/{head_color, hand_left/right_color, head_{left,right,center}_fisheye_color,
  back_{left,right}_fisheye_color}.mp4` then `<ep>/depth/head_depth_NNNNNN.png` (~145 KB each, ~1000+/episode).
  head_color.mp4: AV1 640x480, 30 fps, ~27 MB/episode, keyframe every 2nd frame; libdav1d decodes 1342 frames in 1.3 s.
- Tar names are NOT disjoint episode ranges (81% of episodes fall in >1 name range) → must index to locate.
- **Index without download** (scripts/tar_index.py): header walk for the videos + bisection past the depth PNGs
  (members grouped by episode). 5.3 GB tar: 18 episodes, 639 requests, 138 MB, 90 s. Verified: spans tile the tar
  exactly; a Range-read head_color.mp4 decodes and matches task_info frame counts.
- **Subset rule (deviation):** per task, tars are visited in seeded random order and indexed until ≥ 40 labeled
  episodes are known (scripts/build_index.py); then 40 are sampled at random from those (scripts/plan_splits.py).
  Reason: indexing all 1139 tars (46 TB) to sample from the full task would cost ~1 TB of transfer.
- Windows (scripts/plan_splits.py): 15 fps (stride 2), **F = 33** future frames (9 latents; 1+4n grid for an
  independent causal encode — the plan's F = 32 does not land on whole latents), **K_SET = {1, 5, 9, 17}**
  (1/2/3/5 latents), K_MAX = 17. Anchors inside one segment, futures non-overlapping; short segments keep the k
  that fit (`valid_k`). k = 0 (text-only) with p = 0.1 per batch.
- **Streaming loader** (ldvflux/stream.py, flux-action plug `ldvflux.stream:build`): per worker a seeded share of
  episodes; one Range read per episode's head video; decode all its windows into a shuffle buffer; one k per
  batch (cycled over K_SET by global batch index). No raw video / latents on disk; VAE + text encoding happen in
  the training loop (flux-action's native path). Plan's "pre-encode once" therefore does not apply.

## Phase 2 — Code (flux-action branch `video-only`)

- `PolicyConfig.video_only` + `future_frames`; `processing/video_only.py`; `wiring.build_video_dit`
  (video/video_cond/text only; strict: no missing keys). The base trunk HAS a `video_cond` stream → the k context
  latents go there (clean, timestep 0, no loss); loss = velocity MSE on the 9 future latents only.
- Context and future clips VAE-encoded independently with `encode_task` (the history profile's task path), so
  trimming the context never changes the future latents. Time ids: causal latent at its last frame; gap from the
  last context frame to future latent i is (4i+1)/fps for every k.
- Video timesteps: t = sigmoid(N(1.08, 1.0)) (the trunk's `video_logit_mean/std`, used for video tokens by the
  SO-101 history recipe) instead of DROID's shared action timestep (shift 42), which is tuned for actions.
- Trainer: `optimizer: adamw8bit` (training/lowmem.py: bitsandbytes AdamW8bit on each rank's local FSDP2 shard),
  `ema: cpu` (fp32 pinned host EMA, decay 0.999 with warm-up), both saved per rank inside the step directory
  (resume needs the same world size); `dataset_kwargs`; `eval_every`/`eval_fn` hook; LR schedule sliced to one
  param group (no action heads). flux-action's own tests: 208 passed, 2 skipped after the changes.
- Eval hook (ldvflux/evaluate.py): fixed-noise, fixed-t-grid val loss per k ∈ {0,1,5,9,17}; 8 fixed val samples
  with the EMA (Euler 30 steps, CFG 4, shift 5); early stop after 3 consecutive val-loss rises.

## 2026-10-12 — the NaN at update 11 (and 14): root cause

Symptom: deterministic non-finite gradient at update 11 (also 14) on 2 GPUs, loss always finite, same steps at
480x640 and 384x512, with fp32 AdamW and with 8-bit AdamW. Anomaly detection: first NaN in the backward of a
text-mode block residual (`x + mod_gate * output`).

Ruled out (each by a direct test): the 8-bit optimizer; the data (all inputs finite; the same windows replayed
on one GPU are clean at every timestep); a text-only (k = 0) batch last in the update; the side-stream encoder
overlap; bf16 without autocast; activation checkpointing of the text blocks; a single episode.

Decisive test (scripts/replay_dit.py): the exact DiT inputs of update 11 were dumped per rank (FLUX_DUMP_DIR)
and replayed with the step-10 weights:

| configuration                                           | result |
|---------------------------------------------------------|--------|
| 1 GPU, plain bf16 model, with or without AC             | finite |
| 1 rank under FSDP2 + AC                                 | finite |
| 2-GPU FSDP2 + AC, reshard_after_forward=True (trainer)  | NaN at the gradient-sync micro-batch, both ranks |
| 2-GPU FSDP2 + AC, reshard_after_forward=False           | finite |

→ The NaN is produced inside FSDP2's backward re-gather of resharded parameters (triggered by the activation
checkpoint recompute) when it coincides with the last micro-batch's reduce-scatter: a race in that path
(torch 2.10.0+cu128, cc 8.0), not the model, data, precision or optimizer. Which update it hits depends on
the allocation pattern (the k sequence), hence the fixed step numbers across resolutions.

Fix: `reshard_after_forward: false` (+11 GB resident bf16 params per rank; OOMed at 384x512 at first) paid
for by removing the text encoder from the GPU: all 2,786 captions of the splits are pre-encoded once
(scripts/encode_captions.py → /dev/shm/agibot/captions.pt, 9.1 GB host) and served by `LookupTextEncoder`
(`text_encoder_id: lookup:<file>`), saving 8.9 GB per rank. Safety net kept: `nonfinite: skip` (an update
with a non-finite gradient is dropped, logged with diagnostics, and the run aborts after 5 consecutive or > 1%).

Also fixed on the way: NCCL "unhandled cuda error" at the DCP checkpoint gather (empty the CUDA cache first);
VAE OOM at 4 clips per call (`vae_batch_windows: 1`); eval frames not resized to the training canvas.

## Phase 1 — result (2026-10-12 02:24)

Index: 244 tars read by headers only (221.6 GB transferred of ~11 TB of tars; one tar hit the full tmpfs and was
skipped). 4 tasks have < 40 labeled episodes in total (580: 28, 604: 7, 727: 19, 785: 24) — all their
episodes are used. `data/splits.json` (seed 42): 5 whole tasks held out (358, 389, 487, 688, 739), the rest
95 / 2.5 / 2.5 by episode.

| split       | tasks | episodes | windows | unique captions | caption words min/med/max | windows valid at k=17 |
|-------------|------:|---------:|--------:|----------------:|---------------------------|----------------------:|
| train       | 212   | 7,960    | 159,983 | 2,735           | 2 / 12 / 32               | 157,177 |
| val         | 136   | 210      | 4,386   | 954             | 2 / 12 / 30               | 4,282   |
| test        | 134   | 210      | 4,429   | 854             | 2 / 12 / 32               | 4,359   |
| test_unseen | 5     | 200      | 3,112   | 52              | 8 / 10 / 19               | 3,070   |

Head-camera video streamed per epoch: ~188 GB (train). All 2,786 captions encode to 80 tokens.
Phase-1 "decoded sample from each split looks correct": verified visually for the smoke subset (frames decode,
captions align with task_info segments); the eval hook's val sample grids will show val.

## Phase 2 — done (2026-10-12 02:50)

- 2-GPU runs complete without OOM at 384x512 (peak 71.1 GB/GPU), ~24 s per update of 32 windows.
- Exact resume verified (steps 7-10 reproduce the original losses after resuming from step 6; step 10→11 resumes).
- Eval hook verified end to end at step 2 (val loss per k, EMA samples, grids).
- "Loss decreases" not yet observable: the LR is < 1e-7 during the 500-step warmup; judged from the val curve.
- Zero-shot val loss (step 2 = base weights, fixed noise / t-grid, 32 val windows):
  k=0 0.212 | k=1 0.096 | k=5 0.120 | k=9 0.144 | k=17 0.115 | mean 0.137.
- Deviation: the Phase-3 "zero-shot baseline" with the Phase-4 metrics (PSNR/SSIM/LPIPS/FVD) is deferred:
  both GPUs are needed for training, and the base model's metrics do not depend on when they are computed.
  They will be run with the Phase-4 script (pausing training at a checkpoint if needed; resume is exact).

## Phase 3 — launched 2026-10-12 02:50

configs/video_ft_384.json (copy in logs/video_ft_384.launched.json): 384x512, F = 33, K_SET {1,5,9,17}, k0 10%;
effective batch 32 (2 GPUs x 4 windows x 4 accumulation); AdamW fp32 (betas 0.9/0.999, wd 0), lr 1e-5, warmup
500 then constant, clip 1.0; caption dropout 0.1; CPU EMA 0.999; 10,000 steps (~2 epochs); eval every 500,
checkpoint every 1000 (keep 1 locally); reshard_after_forward=false; captions from the lookup table.
Hub backup sidecar (scripts/hub_backup.py) → private repo Jinyan0924/flux3-agibot-videoft: EMA bf16 of the latest
and the best (lowest val loss) checkpoint, plus metrics/val logs. Expected ~24 s/step → ~2.8 days.

### 2026-10-12 07:40 — step 500 eval, profiling plan
- Step-500 val (EMA, fixed noise/t-grid, 32 windows): mean 0.103 | k=0 0.210 | k=1 0.087 | k=5 0.076 | k=9 0.073 | k=17 0.070
  (zero-shot: 0.137 | 0.212 | 0.096 | 0.120 | 0.144 | 0.115). Longer context now helps monotonically, as it should.
  Train loss per 100 steps: 0.136, 0.111, 0.095, 0.100, 0.108 (noisy: random t and k per step). 31 s/step, 71.7 GB peak.
  Eval-500 cost ~230 s (~1.4% of the 500-step interval). 2 episodes skipped on HF CDN 504s (handled).
- Speed question (user): 31 s/step for 32 windows is ~20% of A100 peak for the 5.57B DiT math alone
  (correction: the trained DiT is 5.57B, full trunk 6.99B; not 12B). Candidates for the rest: VAE encode
  (runs on a side stream overlapping the backward, but competes for SMs), AC recompute, FSDP comm.
- Live profiling is impossible here: py-spy needs ptrace (CapEff lacks CAP_SYS_PTRACE, yama scope 1).
  Plan: at the step-1000 checkpoint stop training, run `scripts/profile_step.py` (single GPU, no FSDP, real val
  windows: prepare/forward/backward per k, optional no-AC and kernel trace) and GPU-test `ldvflux.eval_phase4`
  on a tiny subset on the other GPU, then resume (exact). Not stopping at step 500: no checkpoint there, 4.4 h lost.
- Budget decision (user wants speed): decide at step 2000 from the val curve whether to cap Phase 3 at ~5K steps
  (1 epoch of the 160K train windows; 10K = 2 epochs) and give Phase 5 the same budget.

### 2026-10-12 08:10 — speed-up plan for the step-1000 restart (user: "train fast", compile the blocks, shorter context)
- Context set: user asked for ~{1,2,5,10} images; the KinoVAE is causal with 4x temporal compression, so only
  k = 1 + 4n frames encode cleanly (1, 5, 9, 13, 17). New K_SET = {1, 5, 9} (0.07 / 0.33 / 0.6 s at 15 fps); k = 17
  dropped from training (still measured in the val loss for the curve; early-stop mean over {0,1,5,9}; sample grids at k=9).
  Implemented as dataset_kwargs.k_set (stream + eval hook); splits.json untouched so the fixed val windows are unchanged.
- torch.compile per block (flux_action.training.distributed.apply_compile; TrainConfig compile_blocks/compile_mode),
  torchtitan order: AC -> compile -> fully_shard. NO padding needed: one k per batch and 80-token caption buckets make
  every window in a micro-batch the same length, so packed attention already takes the dense reshape path (plain SDPA);
  the attention math is untouched. 4 static shapes (k = 0, 1, 5, 9) -> 4 graphs per block, cache limit raised to 64.
  CPU trace test (reduced DiT, real dumped inputs, fp32): 1 graph / 0 breaks per block, outputs and grads identical to
  eager (7e-7 / 1e-9), with AC too; second shape recompiles fine.
- EMA every 10 updates (decay compounded by the code), removes a per-step 11 GB device->host copy + CPU lerp.
- Restart config configs/video_ft_384_r2.json (= launched config + ema_every 10, compile_blocks true, k_set [1,5,9]);
  scripts/pause_step1000.sh (stop, profile eager vs compiled on GPU 0, Phase-4 evaluator smoke test on GPU 1),
  scripts/launch_ft384_r2.sh (resume from the latest checkpoint, tmux `train`, port 29700).
- Decide from the profile whether to also batch the VAE encode or unwrap some AC blocks.

### 2026-10-12 11:48 — step-1000 pause: profile, evaluator smoke test, restart on r2
- Step-1000 val: mean 0.100 | k0 0.204 | k1 0.084 | k5 0.074 | k9 0.070 | k17 0.068 (all down vs step 500). Checkpoint 52 s;
  Hub sidecar uploaded ema_step-1000.safetensors (11.1 GB) in 97 s.
- Profile (scripts/profile_step.py, 1 GPU, no FSDP, micro-batch of 4 real val windows, seconds; logs/profile/*.json):
    k=0: prepare(VAE) 0.99 | fwd 0.47 | bwd 1.37   -> compiled fwd 0.35 | bwd 1.03  (-25%)
    k=1: prepare      1.07 | fwd 0.50 | bwd 1.51   -> compiled fwd 0.38 | bwd 1.13  (-25%)
    k=9: prepare      2.24 | fwd 0.72 | bwd 1.81   -> compiled fwd 0.46 | bwd 1.36  (-28%)
  Per step (4 micro-batches/rank): DiT ~8.4 s eager, VAE ~6 s (the 9-frame context encode via the chunked
  vae.encode path costs more than the 33-frame future via encode_task: 1.2 vs 0.9 s), so ~15 s of the 31 s step are
  explained; the rest is the CPU EMA (11 GB D2H + CPU lerp over 2.8B values each step with OMP_NUM_THREADS=8),
  FSDP gathers/reduce-scatter (NVLink, small), optimizer, and allocator pressure at 71.7/80 GB. Compile warm-up
  ~30-40 s per packed shape (inductor FX-graph cache makes the 28 identical single blocks cheap).
- Evaluator smoke test (base model, val, 4 windows, 10 steps): recon metrics + grid OK; FVD crashed on scipy>=1.16
  (sqrtm lost `disp`) -> patched in ldvflux/eval_phase4.py (_patch_sqrtm), FVD verified on CPU. Judge not reached yet.
- 11:48 relaunched from step-1000 with configs/video_ft_384_r2.json (compile_blocks, ema_every 10, k_set [1,5,9]);
  tmux `train`, port 29700, log appended to logs/ft384.log. Pause cost: 11:41 -> 11:48 plus compile warm-up.
- Candidate for later: context encode through encode_task with a padded clip (would need a causality/consistency
  check and changes train+eval encode_clip together) — not done, conditioning semantics must not change mid-run.
- 11:50 first r2 launch crashed at resume: torch.compile inserts `_orig_mod.` into parameter names, CPUEMA compared
  names -> `clean_name` now strips it too (DCP already does). Relaunched 11:52.
- Judge path (Qwen3-VL yes/no) exercised on CPU with a random clip: works (7 s/clip on CPU).

### 2026-10-12 12:12 — restart saga + Looped-DiT implementation (CPU-verified)
- 11:52 relaunch sat at 0% GPU for 15+ min: the loader's resume skip replayed 1000 batches/rank through the full
  fetch+decode pipeline. Fixed in ldvflux/stream.py: skipped batches use metadata-only "dry" windows (no fetch, no
  decode), the buffered ones get frames when live batches start (`materialize`). Tested on CPU: 40 skipped + 3 live
  batches in 39 s. Caveat: windows whose labels run past the video end are unknown without decoding, so the window
  order after a resume can differ slightly from the uninterrupted run (still a valid shuffled stream, same seeds).
  12:10 killed + relaunched r2 with the new loader.
- Looped-DiT retrofit implemented (flux_action/models/looped.py; JointSingleSeq.forward refactored into
  _prepare / _run_blocks / _head, bit-identical to the previous forward on real packed and unpacked inputs):
  LoopParams(loop_split (12,9,12) over 5 mode + 28 single -> pre = mode blocks + single[0:7], core = single[7:16]
  shared, post = single[16:28]; num_loops 4; HeadGate per core block: txt/img Linear(hidden->heads), zero weight,
  bias +4 (sigmoid 0.982), XSA optional), exits decoded through post blocks + heads (`out["exits"][r]`),
  deep_supervision_weights (final_plus_mean -> 1/3,1/3,1/3,1 rescaled to sum 2, as the reference). Policy:
  PolicyConfig.loop / deep_supervision / overlay_weights; _forward_video_only sums the weighted exit losses and
  reports loss_exit{r}/loss_final. CPU tests: depth-1 ungated == plain (0.0); gated(+4) differs 5e-4 at scale 0.46;
  packed == unpacked at N=4; grads reach core blocks and gates; AC + compile with the core blocks reused 4x matches
  eager (loss diff 0, out 1e-6); load_overlay of an EMA-style file OK. Open: FSDP2 with a block run 4x per
  forward (expected fine; verify on GPU at the next pause). scripts/export_ema.py consolidates a checkpoint's EMA
  for overlay_weights; configs/looped_ft_384.json drafted (steps TBD with the user).
- Compute note for Phase 5: pre 12 + core 9x4 + post 12x4 (deep supervision) = 96 block passes vs 33 -> ~2.5-3x the
  per-step cost of Phase 3 (post-stage passes on the 3 intermediate exits are the expensive part).
- FSDP2 + AC with the looped core block run 4x per forward: gradients match a plain single-process run to 6e-10
  (scripts/test_fsdp_loop.py, 2 ranks on CPU/gloo). Hub sidecar had died (tmux `hub` gone, cause unknown; last log
  line = the step-1000 upload) -> relaunched 12:13 (state file idempotent).

### 2026-10-12 12:27 — r2 running: 2x faster
- Steps 1010-1030 on r2 (compiled blocks, EMA every 10, K_SET {1,5,9}): 14.3 / 14.3 / 16.3 s per step (was 31 s),
  peak 71.3-72.1 GB, loss 0.10-0.14 and grad norm 0.06-0.09 as before the restart, no non-finite skips.
  The gain exceeds the profile's -25% on the DiT math, so the per-step CPU EMA (11 GB pageable D2H + CPU lerp at
  OMP_NUM_THREADS=8) must have cost far more than the 2-3 s estimated; it now runs every 10th step.
  Compile warm-up at restart: ~3 min. Pause total (11:41 -> first step 12:19, incl. two failed relaunches): 38 min.
- ETA at 15 s/step: 10K steps -> ~Oct 14 02:00 UTC; a 5K cap -> ~Oct 13 05:00 UTC. Decision at step 2000 (~16:30).

### 2026-10-12 14:30 — crash at step ~1080 (FSDP2 mixed gradient dtype), fixed, relaunched from step 1000
- The r2 run died at 12:39 UTC, ~10 steps after the step-1070 log line: `AssertionError: FSDP reduce-scatter expects
  uniform gradient dtype but got {torch.float32, torch.bfloat16}` in the root parameter group's post-backward.
  Noticed only at 14:24 — the background waiter's `pgrep -f 'flux_action.cli train'` also matched the tmux
  session's command line (the `sleep 1d` tail kept it alive), so it reported RUNNING. ~1h45m of GPU idle.
- Root cause (torch 2.10 `_fsdp_param.py:to_accumulated_grad_if_needed` / `_fsdp_param_group.post_backward`):
  with gradient sync disabled on micro-batches 1-3, FSDP2 converts each parameter's gradient to `reduce_dtype`
  (fp32) when it is first accumulated; a parameter whose FIRST gradient arrives in the synced last micro-batch
  still carries a bf16 autograd gradient, and `foreach_reduce` asserts on the mixed set. The root group holds
  both streams' input/output projections; a k = 0 (text-only, `k0_prob` 0.1) micro-batch gives the video_cond
  stream no gradient. Pattern on one rank: micro-batches (k0, k0, k0, k>0) -> p = 0.1^3 x 0.9 x 2 ranks
  = 0.18 %/step: run 1 survived its 1000 steps with ~16 % luck; r2 hit it after ~80. Not caused by compile.
  Reproduced on CPU with a 2-rank toy (scripts/test_fsdp_accum_dtype.py): (0,0,0,5) asserts, (5,0,0,0) is fine.
- Fix (trainer.py `_update`): micro-batches of an update are run k > 0 first, k = 0 last (micro[0] stays first
  because its encoder outputs were prepared ahead; if micro[0] is the only text-only batch preceding a k > 0 last
  one — accumulation 2 — it is moved last and re-encoded). The parameter set used by the last micro-batch is
  then always a subset of what came before, so every grad reaching the reduce is fp32. Same batches, same
  windows, only the order inside an update changes (summation order only).
- Also: `checkpoint_every` 1000 -> 500 (52 s per checkpoint, +0.1 s/step, halves the loss on a crash; keep 1,
  /dev/shm has 136 GB free), stop_train.sh/waiters now match the python processes only (anchored pgrep).
- Relaunched 14:29 UTC from step-1000 (steps 1001-1080 redone, ~20 min lost + the idle gap). New ETA: step 2000
  ~18:45, 5K cap ~Oct 13 07:30, 10K ~Oct 14 04:30 UTC.

### 2026-10-12 15:40 — checkpoints mirrored to the Hub in full, local copies deleted after upload (user request)
- `scripts/hub_backup.py --full --delete-local` (tmux `hub`): for every complete step-N it uploads the consolidated
  EMA (`ema_step-N.safetensors`, latest + best kept) and the whole step-N directory (sharded model / optimizer /
  CPU-EMA, 83 GB) under `checkpoints/step-N/` (latest only, older folders deleted, history squashed), then
  deletes the local step-N once every file is listed in the repo. step-1000: full upload 334 s; /dev/shm
  33 GB used (was 116 GB).
- Consequence: `resume: auto` would start from step 0 with no local checkpoint, so `scripts/launch_ft384_r2.sh`
  now refuses to launch without one; `scripts/restore_checkpoint.sh <step>` pulls `checkpoints/step-N` back
  from the Hub (~7 min at 200 MB/s) before a relaunch.
- Repo: Jinyan0924/flux3-agibot-videoft (private). The token is in /workspace/.env and was pasted in chat again —
  rotate it after the experiment.

### 2026-10-12 19:00 — evaluation splits published (user request: reproduce val on another machine)
- `scripts/export_eval_dataset.py` → private HF dataset `Jinyan0924/agibot-flux3-eval-splits` (13 GB, 635 files):
  byte-identical head-camera mp4 of every val/test/test_unseen episode (210/210/200; sha256 in episodes/*.json),
  windows/*.jsonl (incl. train definitions), fixed/{val_seed0_n40, test_seed0_n1000, test_unseen_seed0_n1000}.json
  (the exact eval picks), splits.json, meta.json (protocol constants incl. logit-normal t), load_windows.py
  (standalone av+numpy+torch loader; verified bit-identical to ldvflux on a val window), README (CC BY-NC-SA 4.0
  inherited from AgiBot World Beta, non-commercial). Two fetch timeouts on the first pass, fixed on a rerun.
  Local export deleted after upload. Pre-encoded captions.pt (Qwen3-VL, 8.6 GB) not included.

### 2026-10-12 20:03 — step-2000 decision: Phase 3 capped at 5K steps with a 1K-step LR cooldown
- Val (EMA, 32 windows, 8-point t grid), comparable mean over k 0/1/5/9: step 500 0.1115, 1000 0.1082,
  1500 0.1067, 2000 0.1056 (per k at 2000: k0 0.1995, k1 0.0823, k5 0.0721, k9 0.0686, k17 0.0663). Still
  improving but the gain per 500 steps fell 0.0033 -> 0.0015 -> 0.0011 (~1 %), i.e. the "flattens by 2000"
  condition of the user's rule ("train fast, cap at ~5K") holds. Base zero-shot was 0.137.
- Config `video_ft_384_r2.json`: steps 10000 -> 5000, cooldown_start 4000, cooldown_steps 1000 (trunk LR 1e-5
  constant until 4000, then linear to zero at 5000 — the DROID-style cooldown overlay in schedule.py). The
  schedule is unchanged before step 4000, so resuming at 2000 is equivalent to having launched this way.
- Relaunched 20:03 UTC from the local step-2000 checkpoint (steps 2001-2003 redone; the Hub sidecar was stopped
  first so it could not delete step-2000 during the resume, restarted once the trainer was stepping).
  ETA at ~16.5 s/step: step 5000 ≈ Oct 13 10:00 UTC (+ ~10 min relaunch overhead).
- EMA note: CPUEMA uses `min(0.999, (1+n)/(10+n))` per optimizer update, compounded over ema_every 10, so its
  horizon is ~230 steps at step 2000 and ~550 steps at 5000 — the exported EMA is not base-dominated.
- 22:34 UTC step 2500 val (EMA): mean k0/1/5/9 0.1047 (k0 0.1975, k1 0.0817, k5 0.0716, k9 0.0680, k17 0.0658);
  -0.0009 vs 2000, consistent with the cap decision. Checkpoint 2500 handed to the Hub sidecar.
- 2026-10-13 01:05 UTC step 3000 val (EMA): mean k0/1/5/9 0.1042 (k0 0.1966, k1 0.0813, k5 0.0712, k9 0.0677,
  k17 0.0655); -0.0005 vs 2500. Checkpoint 3000 handed to the Hub sidecar.

### 2026-10-13 03:20 — NCCL hang at step ~3175 (30 min) → root cause: rank-dependent FSDP reduce-scatter sequence; fix = gradient parity tie
- 01:55 UTC the trainer stopped logging; GPUs 100 %, no error. The 30-min process-group watchdog (FLUX_ACTION_PG_TIMEOUT_MINUTES
  default) fired at 02:25: rank 1 timed out on the 3-element metrics ALLREDUCE (seq 266730), rank 0 on a 122,683,648-element
  _REDUCE_SCATTER_BASE at the same seq (one mode block; it had enqueued 5 more reduce-scatters than rank 1). Trainer killed
  at 02:28, step-3000 restored from the Hub (7 min), relaunched 03:20 after the investigation below. ~1 h 40 min lost
  (hang 30 min + diagnosis), steps 3001-3175 redone.
- Mechanism (verified with a 2-rank gloo toy, scripts/test_fsdp_rank_patterns.py, a traced run of the real model on the
  GPUs, scripts/trace_fsdp_cli.py, and raw NCCL tests): a text-only (k = 0) window runs the video_cond mode blocks on a
  2-token dummy whose output is discarded, so those 5 blocks (and emb_in/modulations of video_cond in the root group) get
  no gradient from it. FSDP2 then reduce-scatters them from its final autograd callback — after root and time_in — while
  a rank whose last micro-batch has k > 0 reduces them in backward order (interleaved with the video and txt mode blocks).
  NCCL matches collectives by issue order and never checks sizes: with equal byte totals the two ranks' sequences complete
  and the WRONG buffers are added together (verified: swapped-order reduce-scatters return 21 = 1 + 20 instead of 3 and
  30); with unequal totals the ranks deadlock (the hang: rank 1 issued 5 fewer reduce-scatters — the video_cond blocks had
  nothing to reduce in that update; the exact data-dependent trigger was not reproduced, see below).
- Consequence for the run so far: whenever the ranks' last micro-batches differed in k-type, the mode blocks (15 x 122.7M),
  root (264M) and time_in received gradients mixed with the peer's other groups. All 28 single blocks (3.4B of 5.55B
  params) were always reduced correctly. Frequency: ~18 % of updates in run 1 (steps 0-1000, loader order) and ~40 % after
  the 14:29 reorder "fix" (steps 1000-3175; k = 0 micro-batches were deliberately moved last, which made the asymmetry
  more frequent). The val curve kept improving through it (0.1115 → 0.1042), i.e. the corruption acted as gradient noise
  on those groups, not as divergence — but steps 0-3000 are not clean. Steps 3001-5000 are.
- Fix (JointSingleSeq._prepare/_head, training only): the dummy content of every inactive modality and that modality's
  single/late modulations are tied into the first output with weight 0 (``grad_tie``). Every trainable parameter now has a
  gradient in every window, so all ranks issue identical reduce-scatter sequences (traced: 45 groups per update, identical
  order on both ranks, 0 groups without gradient, for both the observed pattern [5,1,9,5]/[0,0,1,9] and the extreme
  [5,5,5,5]/[0,0,0,0], which previously deadlocked). The micro-batch reorder of 14:29 is removed (its dtype hazard cannot
  arise any more: no parameter ever receives its first gradient in the synced micro-batch).
- Guard (training/reduce_guard.py, ``check_reduce_order: true`` default): records the groups reduce-scattered in the synced
  micro-batch via a wrapper on FSDPParamGroup.post_backward and compares the sequence across ranks after every update
  (one all_gather_object); raises with the first differing position. Launch script now sets FLUX_ACTION_PG_TIMEOUT_MINUTES=10.
- Open question: why rank 1 had NO video_cond gradient at ~3175 although its micro-batches were [0,0,1,9]/[0,0,9,5] (k-draw
  is deterministic: ldvflux.stream.batch_k; data position 12000 at step 3000 confirms the mapping). Not reproduced; with
  the tie the condition is impossible by construction (every group reduces every update), and the guard would stop the run
  before NCCL could deadlock... except that a count mismatch deadlocks inside backward — hence the 10-min timeout.
- Debug runs wrote 2 x 83 GB checkpoints to /dev/shm and filled it (the all-k0 run's checkpoint write failed with
  "inline_container.cc unexpected pos"); cleaned up, 126 GB free. Hub sidecar stopped during the restore (it would have
  deleted the restored step-3000); restarted once the trainer resumed.

## 2026-10-13 05:40 UTC — decision: keep the run, conditional extension to 6000

User reviewed the step-3175 hang / gradient-mixing finding (artifact https://claude.ai/artifact/Eeq2cp4ggDEHq23NG6hPsH)
and decided: continue from the restored step 3000 (no from-scratch rerun now; a clean retrain can come later),
go on to Phase 4/5 after this run, and **extend to 6000 steps if the clean segment keeps improving** and the extra
time is modest (+1000 steps ≈ 4.5 h). Rule: decide at the step-4000 eval; extend if mean val loss over k ≥ 1
still drops by ≥ 0.0002 per 500 steps (recent trend 0.0003–0.0004). Mechanics: `configs/video_ft_384_r3.json`
(steps 6000, cooldown 5000→6000) is ready; at step 4000 the LR is still 1e-5 under both configs, so the switch is
a plain resume: stop the hub sidecar (so it cannot delete local step-4000), kill the trainer after step-4000's
COMPLETE + eval, `scripts/launch_ft384_r2.sh ../configs/video_ft_384_r3.json`, restart the sidecar after the
resume event. Steps 4999+ roll into epoch 1 (trainer re-seeds the stream per epoch; 4998 updates per epoch).
Step time unchanged by the fix (16.0–16.3 s mean before and after). Looped model inherits the grad tie
(LoopedJointSingleSeq calls `_prepare`/`_head`), so Phase 5 keeps grad parity.

- step 3500 (first clean checkpoint, 05:58 UTC): val k0 0.1951, k1 0.0807, k5 0.0707, k9 0.0672, k17 0.0651.
  Drop vs 3000: k0 −0.0016, k1 −0.0006, k5 −0.0005, k9 −0.0005, k17 −0.0004 — about 1.5× the 2500→3000 drop
  (k0 −0.0009, k ≥ 1 −0.0003/−0.0004). Consistent with the gradient mixing having slowed steps 0–3000.

- step 4000 (08:30 UTC): val k0 0.1939, k1 0.0802, k5 0.0702, k9 0.0668, k17 0.0647. Drop vs 3500: k0 −0.0011,
  k ≥ 1 −0.0004/−0.0005 — still above the 0.0002 bar, so the run is **extended to 6000 steps**: trainer stopped
  right after the step-4000 checkpoint + eval, relaunched 08:32 with `configs/video_ft_384_r3.json` (steps 6000,
  cooldown 5000→6000; LR unchanged at 1e-5 until 5000), resume "exact" from step-4000 at 08:34. Sidecar restarted
  08:35. First relaunch attempt failed with EADDRINUSE on port 29700 because the old torchrun launcher outlived
  its workers for ~30 s; killed it and relaunched (lesson: wait for `pgrep torchrun` to clear before launching).
  Lost ~5 min. New ETA for step 6000: ≈ 17:45 UTC (2000 steps × 16.3 s + 4 evals/checkpoints).

- step 4500 (11:00 UTC): val k0 0.1927, k1 0.0800, k5 0.0700, k9 0.0665, k17 0.0644. Drop vs 4000: k0 −0.0012,
  k ≥ 1 −0.0003 each. Step time 16.2 s; 1500 steps to go → step 6000 ≈ 17:50 UTC. Epoch 1 starts at step 4999.

- step 5000 (13:39 UTC): val k0 0.1920, k1 0.0796, k5 0.0697, k9 0.0663, k17 0.0643. Drop vs 4500: k0 −0.0007,
  k1 −0.0004, k5 −0.0003, k9 −0.0002, k17 −0.0001. Epoch 1 began at 13:34 (19,992 batches/rank, fresh shuffle).
  Cooldown 5000→6000 now running (LR 1e-5 → 0). Step 6000 ≈ 18:10 UTC.

### Phase 4 plan (prepared 13:50 UTC while the cooldown runs)
`scripts/run_phase4.sh [6000]` — exports the step-6000 EMA (or fetches the sidecar's upload), then in parallel:
GPU 0 fine-tuned EMA, GPU 1 base zero-shot; each: test + test_unseen (k 0 1 5 9 17, n-recon 100, FVD 300 at k 1/17,
judge 100 at k 1), then val recon-only (50 windows); then the zero-shot loop-depth probe of the EMA (gate-free
retrofit (12,9,12) at N = 2, 4, 8, 10; test, k 1 9, 50 windows, PSNR/SSIM/LPIPS) split over the two GPUs, plus
`probe_loops.py` (val velocity loss, depths 1 2 4 8 10) on GPU 1. ≈ 5.5 h wall at ~8 s/clip. Patches:
`eval_phase4.py` now merges earlier `*_k*.json` of the same --out into summary.json (so the val-light pass does not
overwrite the test entries) and builds the gate-free retrofit when `--loops` is given with a plain config.
Then `scripts/build_results.py --results results --out results/RESULTS.md`.

### 2026-10-13 15:30 UTC — Hub storage quota hit; repos deleted on the user's instruction
The sidecar's EMA upload for step 5000 was refused five times (BadRequestError: "You need to set up automatic
credit recharge to use more storage"): the account's private storage was full (flux3-agibot-videoft alone was
111 GB: the 89 GB step-5000 checkpoint + two 11 GB EMAs, one of them a stale step-1000 "best"). The full
checkpoint had uploaded, so local step-5000 was stuck (the sidecar deletes local only after both uploads), which
would have overflowed /dev/shm at the step-6000 write. The user named repos to delete; deleted from the Hub:
looped-dit-t2v-b-webvid50k (14.4 GB), looped-dit-t2v-webvid50k (9.6 GB), sdxl-turbo-onnx (7.0 GB),
chatterbox-turbo-onnx (1.0 GB), hunyuan-dit-onnx (22.3 GB) — 54 GB freed. The step-5000 EMA then uploaded
(15:35), ema_step-4500 pruned, local step-5000 deleted (/dev/shm 43 GB used). Fix in `hub_backup.py`: `best_step`
scores checkpoints by the mean of the per-k (k ≥ 1) val losses, not the headline `val_loss` whose weighting changed
at step 1500 (it had been keeping ema_step-1000 as "best"); sidecar restarted 15:36 with it, so ema_step-1000
will be pruned at the next retention pass. Steady-state footprint: one full checkpoint (89 GB) + one EMA (11 GB).

- step 5500 (16:07 UTC, mid-cooldown, LR ≈ 5e-6): val k0 0.1908, k1 0.0795, k5 0.0695, k9 0.0661, k17 0.0640.
  Drop vs 5000: k0 −0.0011, k1 −0.0001, k5 −0.0002, k9 −0.0003, k17 −0.0003. Step 6000 ≈ 18:10 UTC.

## 2026-10-13 18:46 UTC — Phase 3 complete at step 6000
- step 6000 (18:45 UTC, end of cooldown): val k0 0.1904, k1 0.0792, k5 0.0693, k9 0.0659, k17 0.0639
  (drop vs 5500: k0 −0.0004, k1 −0.0003, k5 −0.0002, k9 −0.0002, k17 −0.0001). Base zero-shot was 0.137 mean.
  Full curve (k1 / k17): 500 0.0871/0.0700 → 3000 0.0813/0.0655 → 6000 0.0792/0.0639. Trainer exited cleanly;
  step-6000 checkpoint COMPLETE (45.8 s write); sidecar uploading it + ema_step-6000 to the Hub.
- EMA exported: /dev/shm/ema/ema_ft384_step-6000.safetensors (5.550B params, 11.1 GB).
- Phase 4 launched 18:47 (`scripts/run_phase4.sh 6000`): tmux eval_ft (cuda:0, fine-tuned), eval_base (cuda:1);
  logs/phase4_ft.log, logs/phase4_base.log; results/ft_step6000, results/base, results/ft_step6000_loops{2,4,8,10},
  results/probe_loops_step6000.json. ETA ≈ 00:30 UTC Oct 14.
- Run summary: steps 0–3000 with the reduce-order gradient mixing (mode blocks/root/time_in, 18–40 % of updates),
  3000–6000 clean; r2 config to 4000, r3 (steps 6000, cooldown 5000→6000) from 4000; 1.2 epochs of data.

- 22:05 UTC: GitHub repo created and pushed: https://github.com/naomili0924/loop-dit (private). First commit =
  configs, ldvflux, scripts, data, NOTES, README (Phase-3 curve + caveat), patches/flux-action.patch, results so
  far (test split). Phase-4 completion → rebuild RESULTS.md, second commit.
- 22:15 UTC: the user's intended repo is https://github.com/naomili0924/looped-dit-video (public, MIT): pushed our
  work as `agibot-flux3/` (configs, ldvflux, scripts, patches, NOTES, README, results tables + JSON) and a README
  section with the headline table. Dataset-derived content (mp4 grids, data/splits.json) stays out of the public
  repo; the private loop-dit repo holds the full mirror. Keep both in sync at the Phase-4/5 milestones.

### Phase 5 plan (prepared 01:05 UTC Oct 14; user: "then we can continue with phase 5")
`configs/looped_ft_384_r1.json`: Looped-DiT retrofit of the step-6000 EMA (overlay), split (12, 9, 12), N = 4,
head-wise gated attention (bias +4), deep supervision final_plus_mean; 2500 steps, trunk warmup 200, LR 1e-5,
cooldown 2000→2500; K_SET {1,5,9} + 10 % k0; 2 × 4 × 4 windows/step; checkpoint/eval every 500, keep_checkpoints 1.
Launch: `scripts/launch_loop384.sh` (tmux train, logs/loop384.log). First GPU run of the looped model: watch peak
memory (Phase 3 peaked 72 GB; the loop adds ~66 block-call checkpoints) — on OOM relaunch with windows_per_rank 2 /
grad_accumulation 8. Expected ~40–45 s/step → 2500 steps ≈ 30 h. Hub: EMA-only sidecar (no --full) to a new
private repo (Phase-3 repo holds 100 GB; quota); local retention by the trainer (keep_checkpoints 1).
Then evaluate at N = 1, 2, 4, 8 with eval_phase4 --config looped_ft_384_r1.json --ema <looped EMA> --loops N.

## 2026-10-14 02:48 UTC — Phase 4 complete; Phase 5 launched
- Phase 4 finished 02:48 (both chains, no errors; 8 h wall). results/RESULTS.md: fine-tuned step-6000 EMA beats the
  base zero-shot in all 30 model×k cells of test / test_unseen / val. test k1: PSNR 19.83 vs 17.03, SSIM 0.805 vs
  0.664, LPIPS 0.166 vs 0.288, FVD 167 vs 408, judge 0.46 vs 0.38; test k17: FVD 129 vs 219. test_unseen k1: FVD
  174 vs 628, judge 0.40 vs 0.27; k17 FVD 130 vs 234 → no overfitting to seen tasks. k17 (never trained) is the
  best k for the fine-tune on every split.
- Zero-shot loop-depth probe (gate-free retrofit (12,9,12), test, 50 windows): N=1 → 2 → 4 → 8 → 10 at k1 LPIPS
  0.166 → 0.183 → 0.203 → 0.357 → 0.514 (PSNR 19.83 → 19.46 → 19.46 → 18.67 → 17.07); k9 LPIPS 0.123 → 0.132 →
  0.148 → 0.259 → 0.360. Val velocity loss (16 windows): k1 0.0771 / 0.0798 / 0.0937 / 0.1472 / 0.1920; k9 0.0655
  / 0.0678 / 0.0800 / 0.1285 / 0.1700. Depths 2–4 degrade mildly, 8–10 strongly. Videos: results/loop_videos/
  (scripts/make_loop_strips.sh: GT | base | N=1 | 2 | 4 | 8 | 10 per window), artifact
  https://claude.ai/artifact/YEtv4D6nkgQ2FSx4BK4rwh.
- Phase 5 launched 02:48 (`scripts/launch_loop384.sh`, configs/looped_ft_384_r1.json, logs/loop384.log): fresh
  start from the step-6000 EMA overlay; first GPU run of the looped model.
- 02:52: first Phase-5 launch died at step 1 on the reduce-order guard: "rank 0 reduce-scattered without a gradient"
  for 42 entries = the 12 post blocks ×2 and the 9 core blocks ×2. Cause: a block used N times per forward gets one
  FSDP2 post_backward per use, and the calls without a gradient are no-ops (FSDP2 returns before foreach_reduce:
  `_fsdp_param_group.py:530`), not collectives. The guard's "missing" rule was wrong for multi-use blocks. Fix in
  reduce_guard.py: the compared sequence is the calls that actually reduced; no-op calls are reported once on
  rank 0 and raise only if they make the sequences differ across ranks. Relaunched 02:54 (fresh start, same config).
- 03:03 Phase 5 step 10: loss 0.200 (weighted exits, weights sum 2), video_mse 0.107, grad_norm 0.42 (warmup, LR
  4.5e-7), step_time 31.4 s (better than the 40–45 s estimate), peak_mem 75.9 GB (nvidia-smi 81.0 / 77.7 GB used).
  Guard: 66 groups reduced per update in identical order on both ranks, 42 no-op post_backward calls. ETA 2500
  steps ≈ 22 h → ~01:00 UTC Oct 15.

### 2026-10-14 06:40 UTC — Phase 5 step 380; budget question (user: why 2500 steps, not 5000 = 1 epoch?)
- Status: step 380/2500, 31.4 s/step steady, loss 0.195 → 0.183 (last 20), video_mse 0.100 → 0.092, grad norm 0.13–0.2,
  peak 76.5 GB; no non-finite steps, guard clean (66 groups, identical order). Step-500 eval ≈ 07:30, end ≈ 00:55 Oct 15.
- Why 2500: the budget was matched to Phase 3 in compute, not steps (a looped step = 106 block passes vs 43 →
  2500 × 106 ≈ 6000 × 43; planned at 40–45 s/step ≈ 30 h vs Phase 3's 27 h). Data-wise it is 0.5 epoch (80K of the
  160K train windows; Phase 3 saw 1.2 epochs, 5000 steps = 1 epoch). Nothing in the data says 0.5 epoch suffices;
  the val curve decides, as in Phase 3 (extended 4000 → 6000 at the step-4000 eval).
- Prepared configs/looped_ft_384_r2.json (steps 5000, cooldown 4500→5000, otherwise identical). Switching is
  lossless at any checkpoint ≤ 2000 (LR constant 1e-5 until cooldown_start 2000): `scripts/launch_loop384.sh
  /workspace/loop-dit/configs/looped_ft_384_r2.json` resumes from the latest checkpoint in /dev/shm/runs/loop384.
  Deadline: the step-2000 checkpoint (≈ 20:30 UTC Oct 14; keep_checkpoints 1 deletes it when step 2500 is written).
  Extension cost: +2500 × 31.4 s ≈ +22 h (end ≈ 23:00 UTC Oct 15 instead of 01:00).

### 2026-10-14 07:25 UTC — decision (user): compare a 2500-step and a 5000-step looped model
- Plan: r1 runs on unchanged to 2500 (cooldown 2000→2500) = the 2500 model. The 5000 model branches from the
  step-2000 checkpoint (last constant-LR one): configs/looped_ft_384_r2.json = r1 with steps 5000, cooldown
  4500→5000, output_dir /dev/shm/runs/loop384_r2. Both share steps 0–2000 and the data stream (resume replays the
  stream position), so they differ only in "500 cooldown steps now" vs "2500 more constant steps + 500 cooldown".
- Branch point: a watcher hard-links /dev/shm/runs/loop384/step-2000 → /dev/shm/runs/loop384_r2/step-2000 (cp -al,
  no extra space; survives r1's prune at 2500). Launch after r1 ends: scripts/launch_loop384.sh
  /workspace/loop-dit/configs/looped_ft_384_r2.json (script now derives run dir / log from the config).
- Disk: a full checkpoint is ~89 GB and /dev/shm fits two beside the 54 GB of weights/captions (54+178+11 = 243 of
  251). r1's final step-2500 full checkpoint must therefore leave /dev/shm before r2 writes its own step-2500
  (4.4 h after r2's launch): upload to the Hub (needs quota → public repos, see below) or keep only its EMA.
- Timeline: r1 ends ≈ 01:00 Oct 15 → Phase-4-style eval of the 2500 EMA (N = 1/2/4/8, ~2–3 h) → r2 2000→5000
  (3000 steps ≈ 26.5 h with evals) → ends ≈ 06:30 Oct 16 → eval of the 5000 EMA.
- HF storage (user: private quota nearly full, use public?): FLUX Kommunity License §3 allows distributing
  Derivatives for non-commercial use with the license copy, the Attribution Notice, a "modified" statement and
  no implied endorsement; AgiBot World Beta is CC BY-NC-SA 4.0. Weights-only repos (flux3-agibot-videoft 100 GB,
  flux3-agibot-looped) can be public with a NOTICE + non-commercial model card; agibot-flux3-eval-splits (13.9 GB
  of AgiBot clips) stays private.
- 07:25 UTC (user: "make the two weight repos public"): added LICENSE.md (FLUX Kommunity License copy), NOTICE
  (attribution notice + modified-derivative statement + AgiBot CC BY-NC-SA note) and a model card to
  Jinyan0924/flux3-agibot-videoft and Jinyan0924/flux3-agibot-looped, then set both public. 100 GB of private
  quota freed; agibot-flux3-eval-splits stays private. Sidecar hub2 keeps uploading looped EMAs to the public repo.
- 07:37 UTC Phase 5 step-500 eval (EMA, N = 4 unless noted; val, 32 fixed windows): k0 0.1929 | k1 0.0806 | k5 0.0707 |
  k9 0.0672 | k17 0.0653; k9 at N = 1/2/8: 0.0670/0.0662/0.0716. Reference: Phase-3 EMA (N = 1) k1 0.0792, k9 0.0659;
  zero-shot gate-free retrofit at N = 4 k1 0.0937, k9 0.0800, N = 8 k9 0.1285. After 500 steps the N = 4 loss is
  0.013 lower than zero-shot and 0.0013 above the N = 1 baseline; N = 8 recovered most of its 0.063 gap.
  Step-500 EMA uploaded to the public looped repo (141 s). Next eval at step 1000 (~12:00 UTC).
- 07:45 UTC (user): after training, a NEW artifact (private, AgiBot frames) with the evaluation: 2500 vs 5000 looped
  models vs Phase-3 fine-tune vs base, N = 1/2/4/8 sweep (+ zero-shot probe), both val curves, side-by-side videos.
  First version after the 2500 model's eval (~04:00 Oct 15), updated after the 5000 model's.

### 2026-10-14 12:27 UTC — Phase 5 step-1000 eval
- Val (EMA, N = 4): k0 0.1916 | k1 0.0799 | k5 0.0700 | k9 0.0666 | k17 0.0647 (step 500: 0.1929 | 0.0806 | 0.0707 |
  0.0672 | 0.0653). k9 at N = 1/2/8: 0.0668/0.0660/0.0699 (500: 0.0670/0.0662/0.0716). Phase-3 N = 1 reference: k1
  0.0792, k9 0.0659. Gap of N = 4 to the reference: k9 +0.0013 → +0.0007 in 500 steps; N = 2 is the best depth so far
  (k9 0.0660, within 0.0001 of the reference); N = 8 still improving fastest (−0.0017).
- Train: loss 0.1894 (steps 0–500) → 0.1827 (500–1000), video_mse 0.0961 → 0.0916; 31.4 s/step; peak 75.7 GB; no
  non-finite steps. Step-1000 EMA on the Hub (94 s); /dev/shm 137 GB used (one full checkpoint). Next eval ~16:50.

### 2026-10-14 17:12 UTC — Phase 5 step-1500 eval
- Val (EMA, N = 4): k0 0.1902 | k1 0.0795 | k5 0.0696 | k9 0.0662 | k17 0.0642 (1000: 0.1916 | 0.0799 | 0.0700 | 0.0666 |
  0.0647). k9 at N = 1/2/8: 0.0665/0.0657/0.0691 (1000: 0.0668/0.0660/0.0699). N = 2 (0.0657) is now below the
  Phase-3 N = 1 reference (0.0659); N = 4 is 0.0003 above it; per-500-step drop at N = 4 k9: 0.0006 → 0.0004.
- Train: loss 0.1746, video_mse 0.0874 (steps 1000–1500), 32.1 s/step, peak 75.9 GB, no non-finite steps.
  Hub: step-1500 EMA uploaded (latest + best kept; ema_step-500 pruned by the sidecar). Cooldown starts at 2000.
- 21:57 UTC: step-2000 checkpoint hard-linked to /dev/shm/runs/loop384_r2/step-2000 (13 files, 83 GB shared, link
  count 2; no extra space, /dev/shm still 137 GB used). Branch point for the 5000-step run secured; r1 cooldown starts.

### 2026-10-14 22:03 UTC — Phase 5 step-2000 eval (cooldown starts)
- Val (EMA, N = 4): k0 0.1891 | k1 0.0791 | k5 0.0691 | k9 0.0658 | k17 0.0639 (1500: 0.1902 | 0.0795 | 0.0696 | 0.0662 |
  0.0642). Phase-3 N = 1 reference: 0.1904 | 0.0792 | 0.0693 | 0.0659 | 0.0639 → the looped model at N = 4 now matches
  or beats the reference at every k. k9 at N = 1/2/8: 0.0663/0.0654/0.0683 (1500: 0.0665/0.0657/0.0691).
- Train 1500–2000: 31.4 s/step, peak 76.3 GB, no non-finite steps. Step-2000 EMA on the Hub. Cooldown 2000→2500
  (linear to 0); step 2500 ≈ 02:30 UTC Oct 15.

## 2026-10-15 02:46 UTC — Phase 5 run r1 (2500 steps) DONE
- Final val (EMA, N = 4): k0 0.1879 | k1 0.0787 | k5 0.0688 | k9 0.0654 | k17 0.0635 (2000: 0.1891 | 0.0791 | 0.0691 |
  0.0658 | 0.0639; Phase-3 N = 1 reference 0.1904 | 0.0792 | 0.0693 | 0.0659 | 0.0639). k9 at N = 1/2/8:
  0.0660/0.0651/0.0678. Below the reference at every k and at N = 2/4; N = 1 (0.0660) ≈ reference; N = 8 0.0678.
  Monotone through all 5 evals. Trainer exited cleanly after the step-2500 checkpoint; no non-finite steps in the run.
- 02:48: EMA exported to /dev/shm/ema/ema_loop2500_step-2500.safetensors (5.551B params). hub2 now does a one-off
  `--once --full --delete-local` upload of step-2500 to Jinyan0924/flux3-agibot-looped (checkpoints/step-2500),
  then deletes the local copy after verification (needed: /dev/shm at 219 GB with r1's 2500 + the r2 branch 2000).
- 02:48: light loop-depth eval of the 2500 model (scripts/run_loop_eval.sh 2500; test split, k 1 9, recon 100,
  FVD 300 at k 1, judge 100 at k 1; N = 8 → 1 on cuda:0, N = 4 → 2 on cuda:1; results/loop2500_loops{1,2,4,8}).
  Full Phase-4 protocol for both looped models at the end. r2 (2000 → 5000) launches when the evals finish.

### 2026-10-15 05:55 UTC — 2500-model eval half done; new evaluation artifact prepared
- 03:01: hub2 finished: step-2500 full checkpoint on the Hub (checkpoints/step-2500, 809 s), local copy deleted
  (13 files verified). /dev/shm 147 GB used, 105 GB free. Phase-3 stale EMA ema_step-1000 still present (11 GB).
- Eval timings (test, k 1: 300 gens + 100 caption-swapped for the judge; k 9: 100 gens), per-window generation cost
  scales with depth: N=1 7.9 s, N=2 9.9 s, N=4 13.8 s, N=8 21.7 s (k 1, batch 4, 30 steps × 2 CFG passes). ETA: GPU 1
  (N 4 → 2) ≈ 07:05, GPU 0 (N 8 → 1) ≈ 08:00 UTC; r2 launches then (≈ 27 h → ends ≈ 11:00 UTC Oct 16).
- Results so far (test, k 1, 100 windows; FVD 300; judge 100) — Phase-3 ft: PSNR 19.83 / SSIM 0.805 / LPIPS 0.166 /
  FVD 166.9 / judge 0.46; base: 17.03 / 0.664 / 0.288 / 407.6 / 0.38.
  loop2500 N=4: 20.00 / 0.807 / 0.162 / 157.1 / 0.39 (k 9: 21.78 / 0.849 / 0.120 vs ft 21.77 / 0.848 / 0.123).
  loop2500 N=8: 20.03 / 0.803 / 0.164 / 159.2 / 0.42 (zero-shot N=8 was LPIPS 0.357: the trained loop holds at depth 8).
  Judge differences (±0.07 at n = 100) are within noise.
- User request: after the eval, a NEW artifact with the same eval cases as the zero-shot gallery
  (https://claude.ai/artifact/YEtv4D6nkgQ2FSx4BK4rwh: 8 fixed test windows, k 1/9, seed 0) and a description of the
  fine-tuning and the inference procedure. Prepared: scripts/make_looped_strips.sh (GT | base | ft N=1 | looped
  N=1/2/4/8 strips from the eval grids) and scripts/build_loop_gallery.py (page "Trained Loop Depth": strips,
  test tables for all models/depths, zero-shot-vs-trained LPIPS, val curves from metrics.jsonl cached as
  results/<tag>_val_curve.json, Phase-3 and Phase-5 recipe, sampling procedure and cost per depth). Takes
  --tags loop2500 loop5000 for the update after r2. Private artifact (AgiBot frames, CC BY-NC-SA).

## 2026-10-15 18:03 UTC — 2500-model eval DONE; r2 launched late (watcher died)
- Eval incident: the N=1 job on cuda:0 crashed at 06:36 during the test-set decode (huggingface.co read timeout
  fetching a dataset tar, transient); relaunched 06:38, finished ≈ 08:05 cleanly. The chained launcher for r2 (a
  background shell waiting on both eval logs) died silently without firing, so the GPUs sat idle 08:05 → 18:02
  (~10 h lost). r2 launched by hand 18:02 (scripts/launch_loop384.sh configs/looped_ft_384_r2.json, resume from
  /dev/shm/runs/loop384_r2/step-2000; EMA sidecar in tmux hub2). r2 ends ≈ 21:30 UTC Oct 16. Lesson: use a tmux
  session (not a tool background shell) for anything that must fire hours later.
- Test results, 2500 model (EMA; test, 100 windows; FVD 300; judge 100; seed 0; results/loop2500_loops{1,2,4,8}):
  | N | k1 PSNR | SSIM | LPIPS | FVD | judge | k9 PSNR | SSIM | LPIPS | s/window k1 |
  | ft (Phase 3) | 19.83 | 0.805 | 0.166 | 166.9 | 0.46 | 21.77 | 0.848 | 0.123 | 7.9 |
  | 1 | 19.97 | 0.805 | 0.166 | 164.2 | 0.43 | 21.82 | 0.848 | 0.121 | 7.9 |
  | 2 | 20.02 | 0.808 | 0.161 | 162.0 | 0.43 | 21.75 | 0.848 | 0.121 | 9.9 |
  | 4 | 20.00 | 0.807 | 0.162 | 157.1 | 0.39 | 21.78 | 0.848 | 0.120 | 13.9 |
  | 8 | 20.03 | 0.803 | 0.164 | 159.2 | 0.42 | 21.73 | 0.846 | 0.122 | 21.9 |
  Reading: after 0.5 epoch of looped training every depth is a valid exit (zero-shot N=8 LPIPS was 0.357, now 0.164);
  depths 2–8 beat the Phase-3 model on PSNR/LPIPS/FVD by small margins (best FVD 157 at the training depth N=4,
  best LPIPS 0.161 at N=2); N=1 ≈ Phase-3 (the gates at N=1 keep its function). Gains are modest (LPIPS −0.005,
  FVD −10): the looped model matches the plain model at 1× cost and improves slightly at 1.3–1.8× cost. Judge
  differences (0.39–0.46, n=100) are within noise. The 5000 model will show whether a full epoch widens the gap.
- Artifact "Trained Loop Depth" (private, AgiBot frames): https://claude.ai/artifact/DQGteiPLtakyHLP4ia8BEA —
  same 8 test windows/k/seed as the zero-shot gallery; strips GT | base | ft | looped N=1/2/4/8; tables; recipe and
  sampling description. results/RESULTS.md rebuilt with the loop2500 rows.
- 18:07: the r2 EMA sidecar (latest + best retention over ema_step-N.safetensors) uploaded r2's step-2000 EMA and
  pruned r1's ema_step-2500.safetensors from the Hub. checkpoints/step-2500 (full, incl. EMA shards) is intact; the
  consolidated r1 EMA is re-uploaded as ema_loop2500_step-2500.safetensors (outside the sidecar's regex; tmux
  hub_r1ema). r2 resumed exactly from step 2000 (seed 42) at 18:05; first steps running.

### 2026-10-15 21:15 UTC — evaluation critique (user) and motion-aware evaluation code
- User: the Phase-4/5 metrics show "not blurry" but do not test semantics/dynamics. Confirmed with numbers on the 8 saved
  test windows (k 1, full res): copy-first-future-frame baseline beats every model on all pixel metrics (PSNR 20.46 vs
  ft 18.86; SSIM 0.866 vs 0.790; LPIPS 0.102 vs 0.159); on moving pixels (8%) models are only ~1.1 dB above static;
  judge adherence is dominated by scene mismatch of other-task captions. Toy-image study (scripts in scratchpad,
  artifact https://claude.ai/artifact/X8rLgDaa8EoZEEkyUkyp19, Chinese): PSNR/SSIM reward a smear over a sharp but
  offset object and cannot tell "moved to a plausible B" from "did not move"; LPIPS punishes blur but still favours
  the smear slightly. Decision (user): no action head / IDM for now; evaluate pixel metrics AND optical flow on the
  moving region, on train, test and test_unseen windows, after r2 finishes.
- Code (pushed, public 9ebe9e7): eval_phase4 --save-clips (near-lossless clips + meta), --no-fvd, train split;
  ldvflux/motion_eval.py (RAFT flow, camera-motion removal via per-frame median, dir_cos / mag_ratio / region_cos /
  EPE / spurious + references real/static/reversed/other-window); ldvflux/clip_eval.py (pixel + flow analysis of
  saved clips, static baseline, per-frame curves, summary + MOTION.md); scripts/run_motion_gen.sh (post-training
  generation: base, ft, loop2500 N1/N4, loop5000 N1/N4 if present; 50 windows x {train,test,test_unseen} x k{1,9})
  and scripts/run_motion_metrics.sh (analysis, GPU or CPU).
- Flow preview, 8 saved test windows, RAFT large full res (results/motion/flow_8win_k{1,9}.json):
  k1 dir_cos / mag_ratio / region_cos: base 0.18/1.02/0.17 (spurious 0.15 = flicker) | ft 0.29/0.62/0.39 |
  loop2500 N1 0.30/0.53/0.40 | N2 0.36/0.71/0.43 | N4 0.41/0.82/0.46 | N8 0.35/0.60/0.35; references: real 1/1/1,
  static 0/0/0, reversed -0.20/0.59/-0.34, other window 0.10/0.18/0.02. k9: all models 0.11-0.19 dir_cos (near the
  0.11 chance level), looped N2/N4 best (0.19/0.18). Reading (8 windows, preliminary): motion direction agreement is
  modest for every model; looped N4 best at k1 (0.41 vs ft 0.29) and moves ~80% as much as the real clip (ft 62%);
  k9 agreement is near chance for all — to be checked on 50 windows per split.

### 2026-10-15 22:30 UTC — loop-depth "background vs motion" evaluation (user request) + hand-off doc
- User hypothesis to test: with more loops the Looped-DiT spends its extra compute on the static background rather
  than on motion accuracy. Design: cases with a static camera (background = copy) and clear object/arm motion
  ("background-easy, motion-hard"), the same noise at N = 1/2/4/8, metrics split into background vs moving region,
  the change between consecutive depths localised (concentration = per-pixel change inside / outside the moving
  region; mass_in vs area share), and alternative prompts (same context frames, different action / arm) without GT.
- Code (pushed): ldvflux/depth_eval.py (generate / --analyze / --summarize, strips GT | N1 | N2 | N4 | N8),
  configs/eval_cases_gallery8.json (the 8 gallery test windows by episode/anchor + 2 alt prompts each; captions
  resolved at run time), scripts/select_motion_cases.py (val scan: camera <= 0.3 px/fr, moving >= 5%, object >= 1.5
  px/fr, hard if direction cosine first/last third < 0.5), scripts/run_depth_eval.sh, EVAL_MOTION.md (full hand-off
  for another machine: setup, weights, inputs = k past frames + task prompt, commands, metric reading).
  Alternative prompts need the real Qwen3-VL encoder (--text-encoder), not the caption lookup table.
- Preview from the already-saved 8-window grids (seed 0 shared across depths; RAFT small on CPU; results/
  depth_eval_preview/): change between consecutive depths concentrates in the moving region — concentration 4.4–7.0,
  mass_in 0.27–0.35 vs area 0.08 — i.e. per pixel the extra loops change the moving region 4–7x more than the
  background. Background PSNR 21.5–24 for every depth vs 32.6 for the static copy (re-synthesis noise); moving-region
  PSNR 10.1–11.0 vs static 9.2. k9 dir_cos N1/2/4/8 = 0.31/0.34/0.40/0.36 (small RAFT; large-RAFT k1 numbers in the
  21:15 entry). 8 cases, no conclusion drawn (user: "this is just an experiment, no need to jump to conclusion").
- Case selector test on the first 12 val picks: 5 pass (3 hard, 2 easy) → ~50 cases from 120 picks.
- Planned on the other machine (user: "another claude and another gpu"): run_depth_eval.sh for loop2500 / ft6000 /
  base on gallery8 then val50; run_motion_gen/metrics on test + test_unseen. Here: after r2 ends, the same for
  loop5000 and the train split.

### 2026-10-15 22:40 UTC — r2 (5000-step run) step-2500 eval
- Val (EMA, N = 4): k0 0.1879 | k1 0.0787 | k5 0.0689 | k9 0.0655 | k17 0.0636; k9 at N = 1/2/8: 0.0661/0.0652/0.0678.
  r1's final (cooled-down) 2500: 0.1879 | 0.0787 | 0.0688 | 0.0654 | 0.0635; 0.0660/0.0651/0.0678 — identical within
  0.0001 although r2 is still at the constant LR (EMA 0.999 smooths over ~1000 steps, so the cooldown's effect on the
  EMA is small at this point). Steps 2030–2500: 29.5–32 s/step, peak 76.4 GB, no non-finite steps. Sidecar uploaded
  r2's ema_step-2500 and pruned ema_step-2000 (r1's EMA stays under ema_loop2500_step-2500). Next eval 3000 ≈ 03:00 UTC.

### 2026-10-16 00:20 UTC — portability fixes for the off-box evaluation (user opened a 2×H100 box with its own Claude)
- Found while writing the hand-off: (1) `ldvflux.eval_phase4` (motion-gen stage) had no `--mirror` / `--text-encoder`,
  so part B of EVAL_MOTION.md could not run without data/splits.json and the caption lookup table; (2) the looped
  configs carry `policy.overlay_weights=/dev/shm/ema/ema_ft384_step-6000.safetensors` (init-time overlay) which the
  policy loads at construction → part A would also have crashed off-box.
- Fixes: `load_policy(..., text_encoder=)` overrides `text_encoder_id` and drops `overlay_weights` whenever an `--ema`
  is given (the EMA replaces the whole DiT); `--mirror` in eval_phase4 reads fixed picks (`fixed/<split>_seed0_n*.json`,
  verified identical order to `stream.fixed_windows` for the first 60 test / test_unseen picks), `meta.json` and the
  frames from the eval-splits snapshot; depth_eval uses the same path (no temp config). `run_motion_gen.sh` is
  path-relative with env overrides (MIRROR, TEXT_ENCODER, FT_EMA, LOOP2500_EMA, LOOP5000_EMA, GPUS, MODELS_ONLY,
  SPLITS); `run_motion_metrics.sh` path-relative. EVAL_MOTION.md updated.
- Checks here (GPUs busy with r2): mirror window decode OK (50, 3, 384, 512); CPU load of the looped policy with a
  bogus overlay path + EMA + real Qwen3-VL: loaded OK in 61 s (overlay ignored, encoder overridden).

## 2026-10-16 05:20 UTC — A100 box lost; r2 (5000-step run) resumed on a 2×H100 box

- The 2×A100 box (Vast 55451460) went offline between 00:30 and 03:00 UTC Oct 16 (last Hub upload 22:56 UTC Oct 15 =
  r2's step-2500 EMA; the step-3000 EMA due ≈ 03:00 never came). r2 was at ≈ step 2700–2900. Its step-2500 *full*
  checkpoint lived only on that box (/dev/shm; the r2 sidecar was EMA-only). Lost: ≈ 3 h of r2 compute.
- New box (user: "continue training"): Vast 56199722, 2×H100 80GB SXM (cc 9.0), driver 535.309.01 (CUDA 12.2 max),
  504 GB RAM, 256 CPUs, **/dev/shm 125 GB** (half the old box), / 32 GB, no volume. torch 2.10.0+cu128 (uv.lock),
  NATTEN 0.21.6+torch2100cu128 (separate wheel, `docs/setup.md`), transformers 5.16.1, bnb 0.50.2, torchmetrics 1.9.0,
  lpips, cd-fvd, imageio(+ffmpeg), scipy. cu128 wheels run fine on the 12.2 driver (matmul / SDPA / compile / NCCL
  all-reduce smoke tests passed; no PTX JIT needed on sm_90). Same layout as before (`/workspace/loop-dit`, weights in
  /dev/shm/flux3/weights symlinked to flux-action/outputs/weights, overlay in /dev/shm/ema, run dir
  /dev/shm/runs/loop384_r2). captions.pt re-encoded (2783 captions, 80 tokens, 9.1 GB, 15 s on one H100) — kept on /
  (`/workspace/agibot/captions.pt`, symlinked from /dev/shm/agibot) because /dev/shm is tight.
- **Resume point = r1's final step-2500 full checkpoint** (Hub `checkpoints/step-2500`, 83 GB), the only resumable state
  left. Caveat for the 2500-vs-5000 comparison: this r2 now shares steps 0–2500 with r1 *including r1's LR cooldown
  2000→2500* (the planned r2 kept LR 1e-5 there); at step 2500 the two EMAs were identical within 0.0001 at every k
  (22:40 UTC Oct 15 entry), so the branch point moved from 2000 to 2500 at a small cost in "constant-LR steps"
  (≈ 250 effective steps). Steps 2500→5000 follow the r2 config unchanged (LR 1e-5, cooldown 4500→5000). Resume
  passed the contract checks (`mode: exact`, seed 42, global batch 32, data position replayed: 10000 batches consumed).
  The Hub copy of r1's full checkpoint was first copied to `r1/checkpoints/step-2500/` (server-side) because the
  sidecar's `--full` retention prunes `checkpoints/step-N` below the latest.
- Disk plan (125 GB): a full checkpoint is 83 GB, so two cannot coexist beside the 25 GB of weights. The sidecar
  (tmux hub2, `--full --delete-local`, state seeded with step 2500 as uploaded) deletes each local step-N once its EMA
  and full checkpoint are verified on the Hub (step-2500 deleted at 05:19 after the trainer had loaded it; /dev/shm
  26 GB used). A crash therefore costs a ~15 min restore (`scripts/restore_checkpoint.sh <step> /dev/shm/runs/loop384_r2`
  with repo flux3-agibot-looped).
- 05:15 UTC launched (`scripts/launch_loop384.sh configs/looped_ft_384_r2.json`; the script now passes HF_TOKEN into the
  tmux session — tokens are not stored on this box). Resume event 05:16; step 2501 guard: 66 groups reduced, 42 no-op
  post_backward calls (as before). Step 2510: loss 0.180, video_mse 0.090, grad_norm 0.47, **step_time 12.95 s**
  (2.4× the A100 box's 31.4 s), peak 76.1 GB; step 2520: loss 0.165, grad_norm 0.13. ETA: 2480 steps × 13 s ≈ 9 h +
  evals → step 5000 ≈ 14:45 UTC Oct 16. Evals at 3000 (≈ 07:05), 3500, 4000, 4500, 5000.

### 2026-10-16 08:35 UTC — r2 step-3000 eval (first val point of the resumed run)
- Trainer healthy on the H100 box: step 3320 at 08:31, loss 0.12–0.19, 12.8 s/step, peak 76 GB, no non-finite steps.
  Sidecar uploaded `ema_step-3000` + `checkpoints/step-3000` in 194 s and deleted the local copy (/dev/shm 26 GB used).
  ETA for step 5000 still ≈ 15:10 UTC.
- Val (EMA, N=4): `val_loss 0.10035 | k0 0.1884 | k1 0.0787 | k5 0.0688 | k9 0.0655 | k17 0.0636`; k9 at N=1/2/8:
  0.0660 / 0.0652 / 0.0678. Versus the step-2000 point (k9 0.0658, k17 0.0639) every k improved by 0.0003–0.0007,
  so constant LR 1e-5 keeps buying a little; versus r1's cooled-down step 2500 (k9 0.0654, k17 0.0635) it is identical
  within 0.0001 — as expected, since this r2 branched from that exact checkpoint 500 steps ago and has not cooled
  down. Still below the Phase-3 N=1 reference at every k (0.1904 | 0.0792 | 0.0693 | 0.0659 | 0.0639), and the
  N=2 < N=4 < N=1 < N=8 ordering at k9 is unchanged. Next evals: 3500 (≈ 09:00), 4000, 4500, 5000.
