"""Ridge-probe a checkpoint: R^2 of (t, h, w) token coordinates vs loop depth.

    python scripts/probe.py --checkpoint /dev/shm/ldv/outputs/run/ema_latest.pt \
        --shards /dev/shm/ldv/webvid50k/shards --clips 64 --max-loops 8 --out probe/
Writes probe.jsonl and probe.png (R^2 per axis vs loop depth, one panel per t).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldv.data import list_shards, make_loader  # noqa: E402
from ldv.encoders import TextEncoder  # noqa: E402
from ldv.probe import run_probe  # noqa: E402
from ldv.sample import load_model  # noqa: E402


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = sorted({r["t"] for r in rows})
    fig, axes = plt.subplots(1, len(ts), figsize=(4 * len(ts), 3.2), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], ts):
        sub = sorted((r for r in rows if r["t"] == t), key=lambda r: r["loop"])
        loops = [r["loop"] for r in sub]
        for key in ("r2_t", "r2_h", "r2_w", "r2_joint"):
            ax.plot(loops, [r[key] for r in sub], marker="o", label=key[3:])
        ax.set_title(f"flow time t = {t}")
        ax.set_xlabel("loop depth (0 = after pre-loop)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("ridge probe R²")
    axes[0][-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--clips", type=int, default=64)
    p.add_argument("--max-loops", type=int, default=8)
    p.add_argument("--t", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--tokens-per-clip", type=int, default=256)
    p.add_argument("--out", default="probe")
    args = p.parse_args()
    device = torch.device("cuda")
    model, cfg = load_model(args.checkpoint, device)
    batch = next(iter(make_loader(list_shards(args.shards)[:2], args.clips, 0, 0, seed=1234)))
    text, mask = TextEncoder(cfg.text_encoder, cfg.prompt_length, device)(batch["caption"])
    rows = run_probe(model, batch["latents"].to(device, torch.float32), text.to(torch.bfloat16), mask,
                     num_loops=args.max_loops, t_values=tuple(args.t), tokens_per_clip=args.tokens_per_clip,
                     noise_scale=cfg.noise_scale)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))
    plot(rows, out / "probe.png")
    print("wrote", out / "probe.png")


if __name__ == "__main__":
    main()
