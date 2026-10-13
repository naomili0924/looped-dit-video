"""Pre-encode every caption of data/splits.json (plus the empty CFG caption) with Qwen3-VL once.

Writes {caption: (L, 20480) bf16} to --out; training then uses ``text_encoder_id: "lookup:<out>"`` and keeps
no text encoder on the GPU (8.9 GB per rank).

    cd flux-action && PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/encode_captions.py \
        --splits ../data/splits.json --out /dev/shm/agibot/captions.pt
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from flux_action.models.text_encoder import load_text_encoder, text_contexts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["/workspace/loop-dit/data/splits.json"])
    p.add_argument("--text-encoder", default="outputs/weights/text_encoder")
    p.add_argument("--out", default="/dev/shm/agibot/captions.pt")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    captions = set()
    for path in args.splits:
        data = json.load(open(path))
        captions |= {w["caption"] for eps in data["splits"].values() for e in eps for w in e["windows"]}
    captions = sorted(captions)
    captions = [""] + captions
    print(f"{len(captions)} unique captions (incl. the empty one)", flush=True)
    enc = load_text_encoder(args.text_encoder, args.device)
    table: dict[str, torch.Tensor] = {}
    t0 = time.time()
    for i in range(0, len(captions), args.batch):
        chunk = captions[i : i + args.batch]
        for c, ctx in zip(chunk, text_contexts(enc, chunk, args.device)):
            table[c] = ctx[0].to("cpu", torch.bfloat16).contiguous()
        if (i // args.batch) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(captions)}  {time.time() - t0:.0f}s", flush=True)
    lengths = sorted({v.shape[0] for v in table.values()})
    total = sum(v.numel() * 2 for v in table.values()) / 1e9
    torch.save(table, args.out)
    print(json.dumps({"captions": len(table), "lengths": lengths, "GB": round(total, 2), "out": args.out,
                      "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
