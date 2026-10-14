# loop-dit — FLUX 3 Action video fine-tune on AgiBot World Beta, and a Looped-DiT retrofit

Full fine-tune (no LoRA) of the video + text streams of `black-forest-labs/flux-3-action-base` (7B DiT) on
AgiBot World Beta humanoid head-camera clips, then a Looped-DiT retrofit (Chng et al., 2026) of the fine-tuned
model. Task: *k* past frames + caption → 33 future frames at 15 fps, 384×512.

| Phase | What | Status |
|---|---|---|
| 1–2 | data index + own train/val/test/test_unseen splits (`data/splits.json`, seed 42) | done |
| 3 | plain fine-tune, 6000 steps × 32 windows (1.2 epochs), K_SET {1, 5, 9} + 10 % text-only | done 2026-10-13 |
| 4 | evaluation: fine-tuned vs base zero-shot (PSNR/SSIM/LPIPS, FVD, VLM judge), zero-shot loop-depth probe | running |
| 5 | Looped-DiT training (split 12 / 9 / 12, N = 4, gated attention, deep supervision), eval at N = 1, 2, 4, 8 | next |

Results: [`results/RESULTS.md`](results/RESULTS.md) (one table per split, model × k). Running lab notebook:
[`NOTES.md`](NOTES.md). Weights: `Jinyan0924/flux3-agibot-videoft` (private, Hub; EMA `ema_step-6000.safetensors`).
Eval splits: `Jinyan0924/agibot-flux3-eval-splits` (private; AgiBot World Beta is CC BY-NC-SA 4.0).

## Phase 3 — validation velocity loss (EMA weights, 32 fixed val windows, fixed timesteps and noise)

| step | k0 | k1 | k5 | k9 | k17 |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.2100 | 0.0871 | 0.0765 | 0.0727 | 0.0700 |
| 1000 | 0.2041 | 0.0844 | 0.0740 | 0.0703 | 0.0678 |
| 2000 | 0.1995 | 0.0823 | 0.0721 | 0.0686 | 0.0663 |
| 3000 | 0.1966 | 0.0813 | 0.0712 | 0.0677 | 0.0655 |
| 4000 | 0.1939 | 0.0802 | 0.0702 | 0.0668 | 0.0647 |
| 5000 | 0.1920 | 0.0796 | 0.0697 | 0.0663 | 0.0643 |
| 6000 | 0.1904 | 0.0792 | 0.0693 | 0.0659 | 0.0639 |

Trunk LR 1e-5 (500-step warmup) to step 5000, linear cooldown to 0 at 6000; CPU EMA; FSDP2 on 2 × A100-80GB,
grad accumulation 4 × 4 windows per rank, activation checkpointing + `torch.compile` per block; ~16 s/step.

**Caveat on steps 0–3000.** A text-only (k = 0) micro-batch left the context-stream blocks without gradient, which
made FSDP2 issue their reduce-scatters in a different order on the two ranks; NCCL matches collectives by order
only, so in roughly 18–40 % of those updates the gradients of the 15 mode blocks, the input/modulation root and the
timestep embedder were summed with the wrong peer buffers (the 28 joint blocks were unaffected). The run hung at
step 3175 when the counts differed; it was restored from step 3000 with a fix that ties every stream into the loss
(`patches/flux-action.patch`: `JointSingleSeq` grad tie + `training/reduce_guard.py`). Steps 3000–6000 are clean;
the validation curve was monotone throughout. A clean re-train is planned later.

## Layout

```
configs/      training / eval configs (video_ft_384_r2.json → steps 0–4000, _r3.json → 4000–6000, looped_ft_384.json)
ldvflux/      streaming AgiBot dataset for the flux-action trainer, val hook, Phase-4 evaluator, judge, sampler
scripts/      index building, splits, launch/restore/backup scripts, FSDP debugging tools, Phase-4 runner
patches/      our changes to black-forest-labs/flux-action (apply with `git apply` on the base commit noted)
data/         tar index + splits
results/      Phase-4 outputs (per split/k JSON, grids) and RESULTS.md
```

Vendored checkouts (`flux-action/`, `looped-dit-video/`) are not included; `patches/flux-action.patch` applies to the base commit in `patches/flux-action.base.txt`.
