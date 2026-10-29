"""Action eval judge: the VLM decides whether each clip completes its action. No pixel metrics: the ground truth shows
one action, and the probes ask for others.

Judge = ldvflux.cap_judge.Judge unchanged (Qwen3-VL-32B-Instruct, greedy, 16 frames spread over the clip with the real
context frame first, "which arm does not matter", JSON completed yes / partial / no, attempted, plausible), so the
numbers are on the capability eval's scale. Jobs:
  <case>/<p>_N<d>_s<seed>   every generated clip against its own prompt
  gt/<case>/<p>             the real footage against all 6 prompts of its window: p0 (its own caption) measures
                            the judge's recall; a "yes" on a probe means the real robot already does what the probe asks
                            (or the judge is lenient on it), so that probe cannot tell following from fallback (report.py
                            flags it)

    cd flux-action && PYTHONPATH=.. .venv/bin/python -m ldvflux.act_judge --run ../results/actioneval/rlong9500 \
        --cases ../configs/eval_cases_action.json --judge /dev/shm/w/qwen3vl32b --shard 0/2 --device cuda:0
Writes <run>/judge_<shard>.json (resumable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .cap_judge import Judge
from .depth_eval import read_clip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    run = Path(a.run)
    shard, n_shards = map(int, a.shard.split("/"))
    cases = {c["id"]: c for c in json.loads(Path(a.cases).read_text())["cases"]}
    gt_meta = json.loads((run / "gt" / "meta.json").read_text()) if (run / "gt" / "meta.json").exists() else {}
    jobs = []
    for cid, c in cases.items():
        for path in sorted((run / "clips" / cid).glob("*.mp4")):
            jobs.append((f"{cid}/{path.stem}", path, c["prompts"][path.stem.split("_")[0]], None))
        if (run / "gt" / f"{cid}.mp4").exists():
            for pk, text in c["prompts"].items():
                jobs.append((f"gt/{cid}/{pk}", run / "gt" / f"{cid}.mp4", text, gt_meta.get(cid, {}).get("gt_frames")))
    out_p = run / f"judge_{shard}.json"
    answers = json.loads(out_p.read_text()) if out_p.exists() else {}
    done = set(answers)  # answered in any shard file: the job list grows with each depth, so shards shift
    for p_ in run.glob("judge_*.json"):
        if p_ != out_p:
            done |= set(json.loads(p_.read_text()))
    jobs = [j for j in jobs if j[0] not in done][shard::n_shards]
    todo = jobs
    print(f"{len(todo)} judge calls to do", flush=True)
    if not todo:
        return
    judge = Judge(a.judge, torch.device(a.device))
    for i, (key, path, prompt, n) in enumerate(todo):
        v = read_clip(path)
        if n:
            v = v[:n]
        answers[key] = {**judge(v, prompt), "prompt": prompt}
        if i % 10 == 0 or i == len(todo) - 1:
            out_p.write_text(json.dumps(answers, indent=1, ensure_ascii=False))
        print(f"[judge] {key}: {answers[key]['completed']} / {answers[key]['attempted']} / {answers[key]['plausible']}", flush=True)
    out_p.write_text(json.dumps(answers, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
