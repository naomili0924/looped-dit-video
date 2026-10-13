"""Sidecar: back up a run's EMA weights (bf16 safetensors) to a private HF model repo as checkpoints appear.

Runs next to training (``--full`` also mirrors each complete step-N directory, 83 GB, latest only), touches only files training has finished writing (``step-N/COMPLETE``). For every new
checkpoint: consolidate the per-rank CPU-EMA shards (training/lowmem.py, FSDP2 dim-0 sharding) into one
``ema_step-N.safetensors`` (11 GB), upload it with the run's config/metrics/eval logs, then delete superseded
uploads so the repo holds only the latest and the best (lowest val loss in eval/val.jsonl) EMA.

    PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/hub_backup.py --run /dev/shm/runs/ft384 \
        --repo Jinyan0924/flux3-agibot-videoft
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


def token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        for line in Path("/workspace/.env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip()
    return tok


def complete_steps(run: Path) -> list[int]:
    return sorted(int(p.name[5:]) for p in run.glob("step-*") if (p / "COMPLETE").exists() and p.name[5:].isdigit())


def consolidate_ema(step_dir: Path) -> dict[str, torch.Tensor]:
    """Per-rank CPU-EMA files -> full bf16 tensors (FSDP2 shards parameters on dim 0, rank order)."""
    ema_dirs = [p for p in step_dir.iterdir() if p.is_dir() and p.name.startswith("ema")]
    assert len(ema_dirs) == 1, ema_dirs
    ranks = sorted(ema_dirs[0].glob("rank-*.pt"))
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in ranks]
    names = parts[0]["names"]
    out = {}
    for i, name in enumerate(names):
        full = torch.cat([p["tensors"][i] for p in parts], dim=0) if len(parts) > 1 else parts[0]["tensors"][i]
        shape = tuple(parts[0]["global_shapes"][i])
        if full.shape != shape:  # FSDP2 pads dim 0 to a multiple of the world size
            full = full.reshape(-1)[: int(torch.tensor(shape).prod())].reshape(shape)
        out[name.replace("_checkpoint_wrapped_module.", "")] = full.to(torch.bfloat16).contiguous()
    return out


def best_step(run: Path) -> int | None:
    path = run / "eval" / "val.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("step", 0) % 500 == 0]  # only steps that have a checkpoint

    def score(r: dict) -> float:  # the headline val_loss changed weighting at step 1500; the per-k columns did not
        ks = [v for k, v in r.items() if k.startswith("val_loss_k") and k != "val_loss_k0"]
        return sum(ks) / len(ks) if ks else r["val_loss"]

    return min(rows, key=score)["step"] if rows else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--poll", type=int, default=120)
    p.add_argument("--once", action="store_true")
    p.add_argument("--full", action="store_true",
                   help="also upload each complete step-N directory (sharded model/optimizer/EMA, resumable) "
                        "under checkpoints/step-N, keeping only the latest one on the Hub")
    p.add_argument("--delete-local", action="store_true",
                   help="with --full: delete a local step-N directory once its EMA and full checkpoint are on the "
                        "Hub (every local file verified present in the repo). Restore with scripts/restore_checkpoint.sh")
    args = p.parse_args()
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    api.create_repo(args.repo, private=True, exist_ok=True)
    run = Path(args.run)
    state_path = run / "hub_backup.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"uploaded": []}
    state.setdefault("full_uploaded", [])
    while True:
        for step in complete_steps(run):
            if args.full and step not in state["full_uploaded"]:
                t0 = time.time()
                try:
                    api.upload_folder(folder_path=str(run / f"step-{step}"), path_in_repo=f"checkpoints/step-{step}",
                                      repo_id=args.repo, commit_message=f"full checkpoint step {step}")
                    state["full_uploaded"].append(step)
                    state_path.write_text(json.dumps(state))
                    print(f"[hub] full checkpoint step {step} uploaded in {time.time() - t0:.0f}s", flush=True)
                    for f in api.list_repo_files(args.repo):
                        m = re.fullmatch(r"checkpoints/step-(\d+)/.*", f)
                        if m and int(m.group(1)) < max(state["full_uploaded"]):
                            api.delete_folder(f"checkpoints/step-{m.group(1)}", repo_id=args.repo)
                            print(f"[hub] deleted checkpoints/step-{m.group(1)}", flush=True)
                except Exception as e:  # retried on the next poll
                    print(f"[hub] full checkpoint step {step} failed: {e!r}", flush=True)
            if (args.full and args.delete_local and step in state["full_uploaded"] and step in state["uploaded"]
                    and (run / f"step-{step}").exists()):
                local = sorted(str(f.relative_to(run)) for f in (run / f"step-{step}").rglob("*") if f.is_file())
                remote = set(api.list_repo_files(args.repo))
                absent = [f for f in local if f"checkpoints/{f}" not in remote]
                if absent:
                    print(f"[hub] step {step}: {len(absent)} files not on the Hub, keeping local: {absent[:3]}", flush=True)
                else:
                    shutil.rmtree(run / f"step-{step}")
                    print(f"[hub] deleted local step-{step} ({len(local)} files verified on the Hub)", flush=True)
            if step in state["uploaded"]:
                continue
            t0 = time.time()
            try:
                ema = consolidate_ema(run / f"step-{step}")
                with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
                    f = Path(tmp) / f"ema_step-{step}.safetensors"
                    save_file(ema, str(f), metadata={"step": str(step), "run": str(run)})
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=args.repo)
                for name in ("metrics.jsonl", "eval/val.jsonl"):
                    if (run / name).exists():
                        api.upload_file(path_or_fileobj=str(run / name), path_in_repo=name, repo_id=args.repo)
                cfg = run / f"step-{step}" / "config.json"
                if cfg.exists():
                    api.upload_file(path_or_fileobj=str(cfg), path_in_repo="config.json", repo_id=args.repo)
                state["uploaded"].append(step)
                state_path.write_text(json.dumps(state))
                print(f"[hub] step {step} uploaded in {time.time() - t0:.0f}s", flush=True)
            except Exception as e:  # retried on the next poll
                print(f"[hub] step {step} failed: {e!r}", flush=True)
                continue
            # retention: keep the latest and the best
            keep = {max(state["uploaded"])}
            best = best_step(run)
            if best in state["uploaded"]:
                keep.add(best)
            for f in api.list_repo_files(args.repo):
                m = re.fullmatch(r"ema_step-(\d+)\.safetensors", f)
                if m and int(m.group(1)) not in keep:
                    api.delete_file(f, repo_id=args.repo)
                    print(f"[hub] deleted {f}", flush=True)
            try:
                api.super_squash_history(args.repo)  # drop LFS history of deleted uploads (storage quota)
            except Exception as e:
                print(f"[hub] squash failed: {e!r}", flush=True)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
