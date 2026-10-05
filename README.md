# Looped-DiT for Text-to-Video

This repo extends **[Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305)** (Chng et al., 2026; [code](https://github.com/OpenSenseNova/Looped-DiT)) from text-to-image to **text-to-video**. The backbone is the **FLUX 3 Action 7B** transformer layout ([black-forest-labs/flux-action](https://github.com/black-forest-labs/flux-action)), trained **from scratch**, with its robot-action and state parts removed.

The Looped-DiT idea: the shared transformer blocks run several times within every denoising step. This adds depth without adding parameters.

```
h_0 = A(x)          pre-loop:  5 per-stream (mode) blocks + 7 joint single blocks   (run once)
h_r = B(h_{r-1})    looped:    9 joint single blocks, shared weights                (r = 1..N, N = 4)
x0_hat(r) = C(h_r)  post-loop: 12 joint single blocks + final layer                 (every exit decodable)
```

## What is kept from the paper and what changed

| | Looped-DiT (T2I) | this repo (T2V) |
|---|---|---|
| Backbone | MiniT2I pixel MMDiT, 260M, 17 blocks `[6,5,6]` | FLUX 3 Action layout, hidden 3072, 24 heads, 5 mode + 28 single blocks `[12,9,12]` (same 6:5:6 ratio), **4.81B** params |
| Removed from FLUX | – | action / action_cond / state streams, video_cond (history) stream, pooled vector input, PAG / packing / FP8 inference code |
| Loop depth | N=4 train, variable at inference | same |
| Deep supervision | `final_plus_mean` (1/3,1/3,1/3,1), rescaled to sum 2 | same |
| Self-modulating attention | XSA in looped blocks (gate was an ablation) | **head-wise sigmoid gated attention** in looped blocks (modality-specific, zero-bias init); XSA kept as a flag |
| Objective | x0-prediction, velocity loss with `max(1-t, 0.05)`, logit-normal(-0.8, 0.8), CFG drop 0.1 | same |
| Noise scale | 2.0 (pixels) | **1.0** (Wan latents are unit-normalized) |
| Space | pixels, 2D RoPE + 2D sincos abs. pos. | Wan2.1 VAE latents (4x temporal, 8x spatial, 16 ch), patch (1,2,2), FLUX 4-axis RoPE (t,h,w,l); optional 3D sincos abs. pos. (`abs_pos_embed`) |
| Timestep conditioning | none | FLUX adaLN (kept, part of the base model) |
| Text encoder | FLAN-T5-Large, 256 tokens | FLAN-T5-Large, **128** tokens (WebVid captions are short) |
| Ridge probe | probe 2D patch coordinates per loop (analysis only) | probe **3D (t,h,w)** coordinates per loop: R² per axis, space, joint (analysis only, never in the loss) |
| Optimizer | AdamW (0.9, 0.95), wd 0, clip 0.1, warmup 5K, lr 4e-4, batch 1024, EMA 0.99995 | same, except 8-bit AdamW (memory), **lr 1e-4** (4.8B vs 260M), **batch 64** and **warmup 1K** for the 50K run, **EMA 0.9999** for short runs (CPU-resident) |

> **Note on the probe and FLUX:** FLUX is RoPE-only, so absolute position never enters the hidden states. It is injected only through attention logits. A freshly initialised model therefore has R²≈0 at every depth, unlike the paper's MiniT2I, which adds a sincos position embedding. Set `model.abs_pos_embed: true` to probe the paper's question ("does looping erode local positional information?") under the paper's conditions.

## Model sizes

Every size uses the same code, blocks and training recipe; only the config file differs, so moving between a small model and the full FLUX layout is a change of `--config`.

| Config | Width / heads | Blocks (pre / looped / post) | Parameters | Step time, batch 64, one H200 |
|---|---|---|---|---|
| `configs/t2v_s_webvid50k.yml` | 768 / 12 | 17 (6 / 5 / 6) | 156M | not measured |
| `configs/t2v_b_webvid50k.yml` | 1024 / 16 | 17 (6 / 5 / 6), the paper's split and size | 276M | ~3.6 s, 21 GB |
| `configs/t2v_l_webvid50k.yml` | 1536 / 24 | 23 (8 / 7 / 8) | 836M | not measured |
| `configs/t2v_7b_webvid50k.yml` | 3072 / 24 | 33 (12 / 9 / 12), FLUX 3 Action layout | 4.81B | ~34 s, 71 GB |

```bash
python -m ldv.train --config configs/t2v_b_webvid50k.yml  --output-dir $DATA_ROOT/outputs/webvid50k_b    # small
python -m ldv.train --config configs/t2v_7b_webvid50k.yml --output-dir $DATA_ROOT/outputs/webvid50k      # full FLUX layout
torchrun --nproc_per_node=8 -m ldv.train --config configs/t2v_7b_webvid50k.yml --output-dir ...           # more GPUs
```

The data shards are shared by all sizes. Checkpoints are not interchangeable between sizes.

## Layout

```
ldv/model.py        LoopedFluxT2V: FLUX blocks, loop stages, gated attention / XSA, exits
ldv/diffusion.py    flow-matching loss with deep supervision, Euler + CFG sampler
ldv/probe.py        3D ridge probe (closed-form, lambda per axis by clip-level validation)
ldv/encoders.py     frozen Wan2.1 VAE and FLAN-T5
ldv/data.py         WebVid download/decode, latent WebDataset stream
ldv/train.py        training loop (single GPU or DDP), CPU EMA, sampling, periodic probe
ldv/sample.py       generate videos at any loop depth
scripts/prepare_webvid.py   download -> decode -> VAE encode -> latent shards (no raw mp4 kept)
scripts/probe.py            standalone probe of a checkpoint -> R² vs loop depth (+ plot)
configs/            t2v_{s,b,l,7b}_webvid50k.yml, t2v_7b_webvid1m.yml, tiny.yml
```

## Quickstart

```bash
pip install -r requirements.txt
export DATA_ROOT=/dev/shm/ldv          # large tmpfs on this box; RAM-backed and wiped on restart

# 1. 50K-clip end-to-end subset (17 frames @ 8 fps, 256², ~160 KB latents per clip)
python scripts/prepare_webvid.py --out $DATA_ROOT/webvid50k --num-clips 50000

# 2. train (resumes from the newest checkpoint in --output-dir)
python -m ldv.train --config configs/t2v_7b_webvid50k.yml --output-dir $DATA_ROOT/outputs/webvid50k

# 3. sample at different loop depths, and probe
python -m ldv.sample --checkpoint $DATA_ROOT/outputs/webvid50k/ema_latest.pt --prompt "Waves at sunset" --loops 1 4 8
python scripts/probe.py --checkpoint $DATA_ROOT/outputs/webvid50k/ema_latest.pt --shards $DATA_ROOT/webvid50k/shards

# Scale-up: ~1M public WebVid clips, shards streamed to the HF Hub
python scripts/prepare_webvid.py --out $DATA_ROOT/webvid1m --num-clips 1000000 --start-partition 5 \
    --hub-repo <user>/webvid1m-wan-latents --delete-after-upload
torchrun --nproc_per_node=8 -m ldv.train --config configs/t2v_7b_webvid1m.yml --output-dir ...
```

To back checkpoints up off the box, set `HF_TOKEN` and pass `--set hub_repo=<user>/<repo>`: the EMA weights, config and logs are pushed to a private Hugging Face model repo every `hub_every` steps.

Ablations use `--set`, e.g. `--set model.use_attn_gate=false model.use_xsa=true` (the paper's XSA), `model.num_loops=1 deep_supervision=false` (no looping), `model.share_loop_weights=false` (compute-matched untied baseline) or `model.abs_pos_embed=true`.

## Measured on one H200 (17x256x256 clips)

For the 4.81B model, a step at batch 64 (4 micro-batches of 16, gradient checkpointing, 4 loops with deep supervision) takes about 34 s and peaks at 71 GB of GPU memory: roughly 160K clips per day. The 20K-step 50K-clip config is therefore about 8 days on a single GPU; use `torchrun` on more GPUs, fewer steps, or a smaller model size to shorten it. The 276M model takes about 3.6 s per step (micro-batch 64), about 20 hours for 20K steps. Preparing the 50K clips (download, decode, VAE encode) takes about 75 minutes with 72 CPU workers.

## Tests

```bash
pytest -q
```

## Licenses

MIT (see `LICENSE`). Contains code adapted from Looped-DiT (MIT) and FLUX 3 Action (Apache-2.0); see `NOTICE`. WebVid data and Wan2.1 / FLAN-T5 weights are under their own licenses.
