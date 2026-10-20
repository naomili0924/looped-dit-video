"""Sidecar: back up a run's EMA weights (bf16 safetensors) to a private HF model repo as checkpoints appear.

Runs next to training (``--full`` also mirrors each complete step-N directory, 83 GB, latest only), touches only files training has finished writing (``step-N/COMPLETE``). For every new
checkpoint: consolidate the per-rank CPU-EMA shards (training/lowmem.py, FSDP2 dim-0 sharding) into one
``ema_step-N.safetensors`` (11 GB), upload it with the run's config/metrics/eval logs, then delete superseded
uploads so the repo holds only the latest and the best (lowest val loss in eval/val.jsonl) EMA.

    PYTHONPATH=/workspace/loop-dit .venv/bin/python ../scripts/hub_backup.py --run /dev/shm/runs/ft384 \
        --repo Jinyan0924/flux3-agibot-videoft

``--prefix <folder>`` puts every upload under that folder of a shared repo; with it, nothing outside the folder is
touched, ``--keep-all`` skips the retention deletes and history squashing is off unless ``--squash``.
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
    if not tok and Path("/workspace/.env").exists():  # tokens normally come from the environment
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
    p.add_argument("--prefix", default="", help="folder in the repo for every upload (shared repos)")
    p.add_argument("--keep-all", action="store_true", help="never delete earlier EMA uploads")
    p.add_argument("--keep-steps", type=int, nargs="*", default=[],
                   help="steps whose EMA and full checkpoint are never deleted from the Hub (e.g. the pre-cooldown step)")
    p.add_argument("--keep-every", type=int, default=0,
                   help="keep every EMA whose step is a multiple of this (replaces keeping the best by val loss)")
    p.add_argument("--squash", action="store_true", help="squash the repo history after deletes (default without --prefix)")
    args = p.parse_args()
    if not args.prefix:
        args.squash = True  # the original behaviour for a run's own repo
    pre = f"{args.prefix.strip('/')}/" if args.prefix else ""
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    api.create_repo(args.repo, private=True, exist_ok=True)
    run = Path(args.run)
    state_path = run / "hub_backup.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"uploaded": []}
    state.setdefault("full_uploaded", [])
    state.setdefault("full_in_progress", [])

    def tree_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size

    def local_files(p: Path) -> list[str]:
        return [str(f.relative_to(run)) for f in (p.rglob("*") if p.is_dir() else [p]) if f.is_file()]

    while True:
        # steps whose COMPLETE marker is hidden (full upload in progress) are tracked in the state file
        steps = sorted(set(complete_steps(run)) | {s for s in state["full_in_progress"] if (run / f"step-{s}").exists()})
        for step in steps:
            sd = run / f"step-{step}"
            if step not in state["uploaded"]:  # 1) EMA + logs first: small, and the deliverable
                t0 = time.time()
                try:
                    ema = consolidate_ema(sd)
                    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
                        f = Path(tmp) / f"ema_step-{step}.safetensors"
                        save_file(ema, str(f), metadata={"step": str(step), "run": str(run)})
                        api.upload_file(path_or_fileobj=str(f), path_in_repo=pre + f.name, repo_id=args.repo)
                    del ema
                    for name in ("metrics.jsonl", "eval/val.jsonl"):
                        if (run / name).exists():
                            api.upload_file(path_or_fileobj=str(run / name), path_in_repo=pre + name, repo_id=args.repo)
                    cfg = sd / "config.json"
                    if cfg.exists():
                        api.upload_file(path_or_fileobj=str(cfg), path_in_repo=pre + "config.json", repo_id=args.repo)
                    state["uploaded"].append(step)
                    state_path.write_text(json.dumps(state))
                    print(f"[hub] step {step} uploaded in {time.time() - t0:.0f}s", flush=True)
                except Exception as e:  # retried on the next poll
                    print(f"[hub] step {step} failed: {e!r}", flush=True)
                    continue
                if not args.keep_all:
                    # retention: keep the latest and the best
                    keep = {max(state["uploaded"]), *args.keep_steps}
                    if args.keep_every:  # a regular series instead of a "best" pick
                        keep |= {st for st in state["uploaded"] if st % args.keep_every == 0}
                    else:
                        best = best_step(run)
                        if best in state["uploaded"]:
                            keep.add(best)
                    for f in api.list_repo_files(args.repo):
                        m = re.fullmatch(re.escape(pre) + r"ema_step-(\d+)\.safetensors", f)
                        if m and int(m.group(1)) not in keep:
                            api.delete_file(f, repo_id=args.repo)
                            print(f"[hub] deleted {f}", flush=True)
                    if args.squash:
                        try:
                            api.super_squash_history(args.repo)  # drop LFS history of deleted uploads (storage quota)
                        except Exception as e:
                            print(f"[hub] squash failed: {e!r}", flush=True)
            if (args.full and args.delete_local and step in state["full_uploaded"] and sd.exists()):
                # a restored (or leftover) copy of a checkpoint that is already on the Hub: verify, then free /dev/shm
                local = sorted(local_files(sd))
                remote = set(api.list_repo_files(args.repo))
                absent = [f for f in local if f"{pre}checkpoints/{f}" not in remote]
                if absent:
                    print(f"[hub] step {step}: {len(absent)} files not on the Hub, keeping local: {absent[:3]}", flush=True)
                else:
                    shutil.rmtree(sd)
                    print(f"[hub] deleted local step-{step} ({len(local)} files verified on the Hub)", flush=True)
                continue
            # 2) full checkpoint, staged: biggest entries first, each local copy deleted as soon as it is on the Hub
            # (/dev/shm is RAM and counts against the container's memory limit; the 83 GB must not sit there for the
            # whole upload). COMPLETE is renamed locally while entries go missing so resume=auto and the restore
            # script never see a partial directory; it is uploaded last.
            if args.full and step in state["uploaded"] and step not in state["full_uploaded"] and sd.exists():
                t0 = time.time()
                try:
                    if step not in state["full_in_progress"]:
                        (sd / "COMPLETE").rename(sd / "COMPLETE.partial")
                        state["full_in_progress"].append(step)
                        state_path.write_text(json.dumps(state))
                    entries = sorted((p for p in sd.iterdir() if p.name != "COMPLETE.partial"), key=tree_size, reverse=True)
                    for p in entries:
                        size_gb = tree_size(p) / 1e9
                        dest = f"{pre}checkpoints/step-{step}/{p.name}"
                        if p.is_dir():
                            api.upload_folder(folder_path=str(p), path_in_repo=dest, repo_id=args.repo,
                                              commit_message=f"full checkpoint step {step}: {p.name}")
                        else:
                            api.upload_file(path_or_fileobj=str(p), path_in_repo=dest, repo_id=args.repo,
                                            commit_message=f"full checkpoint step {step}: {p.name}")
                        if args.delete_local:
                            remote = set(api.list_repo_files(args.repo))
                            absent = [f for f in local_files(p) if f"{pre}checkpoints/{f}" not in remote]
                            if absent:
                                raise RuntimeError(f"{p.name}: {len(absent)} files missing on the Hub after upload: {absent[:3]}")
                            shutil.rmtree(p) if p.is_dir() else p.unlink()
                        print(f"[hub] step {step}: {p.name} ({size_gb:.1f} GB) on the Hub"
                              f"{', local copy deleted' if args.delete_local else ''} ({time.time() - t0:.0f}s)", flush=True)
                    api.upload_file(path_or_fileobj=str(sd / "COMPLETE.partial"), path_in_repo=f"{pre}checkpoints/step-{step}/COMPLETE",
                                    repo_id=args.repo, commit_message=f"full checkpoint step {step}: COMPLETE")
                    state["full_uploaded"].append(step)
                    state["full_in_progress"].remove(step)
                    state_path.write_text(json.dumps(state))
                    print(f"[hub] full checkpoint step {step} uploaded in {time.time() - t0:.0f}s", flush=True)
                    if args.delete_local:
                        shutil.rmtree(sd)
                        print(f"[hub] deleted local step-{step}", flush=True)
                except Exception as e:  # retried on the next poll; entries already on the Hub are deduplicated
                    print(f"[hub] full checkpoint step {step} failed: {e!r}", flush=True)
                    continue
                try:  # keep only the latest full checkpoint on the Hub
                    old = set()
                    for f in api.list_repo_files(args.repo):
                        m = re.fullmatch(re.escape(pre) + r"checkpoints/step-(\d+)/.*", f)
                        if m and int(m.group(1)) < max(state["full_uploaded"]) and int(m.group(1)) not in args.keep_steps:
                            old.add(int(m.group(1)))
                    for old_step in sorted(old):  # one delete per folder
                        api.delete_folder(f"{pre}checkpoints/step-{old_step}", repo_id=args.repo)
                        print(f"[hub] deleted {pre}checkpoints/step-{old_step}", flush=True)
                except Exception as e:
                    print(f"[hub] prune failed: {e!r}", flush=True)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
