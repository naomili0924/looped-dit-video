"""Mirror full resumable checkpoints (model + optimizer + EMA + RNG, ~83 GB each) to a private HF repo folder and
verify every file by size. Unlike hub_backup.py --full, nothing on the Hub is ever deleted.

    python scripts/hub_full_ckpt.py --repo Jinyan0924/flux3-agibot-looped --prefix random_depth_n1-8 \
        ckpt_backup/looprand/step-2000 /dev/shm/runs/looprand/step-2500

A step dir that does not exist yet is waited for (until its COMPLETE file appears). Restore with
scripts/restore_full_ckpt.sh.
"""
import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError

sys.path.insert(0, str(Path(__file__).parent))
from hub_backup import token  # noqa: E402


def verify(api: HfApi, repo: str, src: Path, dst: str) -> list[str]:
    try:
        remote = {f.path: f.size for f in api.list_repo_tree(repo, path_in_repo=dst, recursive=True) if hasattr(f, "size")}
    except EntryNotFoundError:  # nothing uploaded under dst yet
        remote = {}
    bad = []
    for f in src.rglob("*"):
        if f.is_file():
            rel = f"{dst}/{f.relative_to(src).as_posix()}"
            if remote.get(rel) != f.stat().st_size:
                bad.append(rel)
    return bad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("steps", nargs="+", help="step-N directories")
    args = p.parse_args()
    api = HfApi(token=token())
    for s in map(Path, args.steps):
        while not (s / "COMPLETE").exists():
            time.sleep(60)
        dst = f"{args.prefix}/checkpoints/{s.name}"
        if not verify(api, args.repo, s, dst):
            print(f"[full] {dst} already on the Hub", flush=True)
            continue
        for attempt in (1, 2):
            t0 = time.time()
            try:
                api.upload_folder(repo_id=args.repo, folder_path=str(s), path_in_repo=dst,
                                  commit_message=f"full resumable checkpoint {s.name}")
            except Exception as e:
                print(f"[full] {dst} attempt {attempt} failed: {e!r}", flush=True)
                continue
            bad = verify(api, args.repo, s, dst)
            print(f"[full] {dst} uploaded in {time.time() - t0:.0f}s; "
                  + ("verified (every file, by size)" if not bad else f"MISSING/SIZE MISMATCH: {bad}"), flush=True)
            if not bad:
                break


if __name__ == "__main__":
    main()
