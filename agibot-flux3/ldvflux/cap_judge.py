"""Scores of the capability eval (ldvflux.cap_eval): a VLM judge on every clip plus pixel metrics against the ground truth.

Judge: Qwen3-VL-32B-Instruct (greedy) sees 16 frames spread over the clip and the instruction, told that which arm acts
does not matter, and answers JSON {"completed": yes|partial|no, "attempted": yes|no, "plausible": yes|no, "reason"}.
score = 1 / 0.5 / 0 for completed yes / partial / no. Calibration on the real episodes (ground-truth clips): the real
caption should get "yes", a caption of another case (a different family) "no"; both rates are reported so the judge's
own error is visible next to the model's numbers. Cross check (prompt discrimination): every real clip is also judged
against the case's alt2 instruction and every alt2 clip against the real one (keys cross/<case>/<clip>); a model that
follows its prompt, and a judge that can tell, give the own-prompt score clearly above the cross score.

Pixel metrics (real cases, real prompt, frames the episode has): per window PSNR and LPIPS to the ground truth
(ldvflux.seed_metrics definitions), and motion (mean |frame_t - frame_t-1|) for every clip and the ground truth.

    cd flux-action && PYTHONPATH=.. .venv/bin/python -m ldvflux.cap_judge --run ../results/capeval/rlong9500 \
        --cases ../configs/eval_cases_capability.json --judge /dev/shm/w/qwen3vl32b --device cuda:0

Outputs: <run>/judge.json {clip: answer}, <run>/metrics.json {clip: {...}}, <run>/summary.json (per case × prompt
and per family: mean judge score over seeds, metrics, calibration). Resumable (answers already in judge.json are kept).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from .depth_eval import read_clip
from .seed_metrics import Lpips, frame_psnr, motion, windows

N_FRAMES = 16
QUESTION = (
    "These {n} frames are evenly spaced over a {sec:.1f}-second video from the head camera of a two-armed robot. "
    "The robot was asked: \"{prompt}\"\n"
    "Judge only what happens in the video. Which arm (left or right) does it does not matter. Answer with JSON only: "
    "{{\"completed\": \"yes\" | \"partial\" | \"no\", \"attempted\": \"yes\" | \"no\", \"plausible\": \"yes\" | \"no\", "
    "\"reason\": \"<one sentence>\"}} where completed = the instruction is carried out by the end, partial = clearly "
    "started but not finished, attempted = the robot visibly moves to do this instruction (not some other action), "
    "plausible = arms, hands and objects keep a physically possible shape and identity (no melting, duplicating or "
    "objects appearing from nowhere).")
SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


class Judge:
    def __init__(self, path: str, device):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.device = device

    @torch.no_grad()
    def __call__(self, video: np.ndarray, prompt: str, fps: float = 15.0) -> dict:
        idx = np.linspace(0, len(video) - 1, N_FRAMES).round().astype(int)
        frames = [torch.from_numpy(video[i]).permute(2, 0, 1) for i in idx]
        text = QUESTION.format(n=N_FRAMES, sec=len(video) / fps, prompt=prompt)
        messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": text}]}]
        inputs = self.proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", fps=N_FRAMES * fps / len(video)).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
        raw = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            ans = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            ans = {}
        ans = {k: str(ans.get(k, "")).strip().lower() for k in ("completed", "attempted", "plausible", "reason")}
        ans["score"] = SCORE.get(ans["completed"])
        ans["raw"] = raw.strip()
        return ans


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-judge", action="store_true", help="pixel metrics and summary only")
    a = p.parse_args()
    run, dev = Path(a.run), torch.device(a.device)
    cases = {c["id"]: c for c in json.loads(Path(a.cases).read_text())["cases"]}
    judge_p, metrics_p = run / "judge.json", run / "metrics.json"
    answers = json.loads(judge_p.read_text()) if judge_p.exists() else {}
    clips = sorted((run / "clips").glob("*/*.mp4"))
    gts = {c: run / "gt" / f"{c}.mp4" for c in cases if (run / "gt" / f"{c}.mp4").exists()}
    gt_meta = json.loads((run / "gt" / "meta.json").read_text()) if (run / "gt" / "meta.json").exists() else {}

    if not a.no_judge:
        judge = Judge(a.judge, dev)
        real_ids = sorted(gts)
        jobs = [(f"{p.parent.name}/{p.stem}", p, cases[p.parent.name]["prompts"][p.stem.split("_")[0]]) for p in clips]
        for i, cid in enumerate(real_ids):  # calibration: the ground truth with its own and with a foreign caption
            other = next(cases[o] for o in real_ids[i + 1:] + real_ids[:i] if cases[o]["family"] != cases[cid]["family"])
            jobs += [(f"gt/{cid}/real", gts[cid], cases[cid]["prompts"]["real"]),
                     (f"gt/{cid}/foreign", gts[cid], other["prompts"]["real"])]
        for p_ in clips:  # cross: every non-real clip vs the real caption (fallback); a real clip vs alt2 (v1 cases)
            pk, prompts = p_.stem.split("_")[0], cases[p_.parent.name]["prompts"]
            other_pk = "real" if pk != "real" else "alt2" if "alt2" in prompts else None
            if other_pk:
                jobs.append((f"cross/{p_.parent.name}/{p_.stem}", p_, prompts[other_pk]))
        for key, path, prompt in jobs:
            if key in answers:
                continue
            v = read_clip(path)
            if key.startswith("gt/"):
                v = v[: gt_meta.get(key.split("/")[1], {}).get("gt_frames", len(v))]
            answers[key] = {**judge(v, prompt), "prompt": prompt}
            judge_p.write_text(json.dumps(answers, indent=1))
            print(f"[judge] {key}: {answers[key]['completed']} / {answers[key]['attempted']} / {answers[key]['plausible']}", flush=True)
        del judge
        torch.cuda.empty_cache()

    lp = Lpips(dev)
    metrics = {}
    gt_cache = {c: read_clip(p) for c, p in gts.items()}
    for path in clips:
        cid, name = path.parent.name, path.stem
        v = read_clip(path)
        wins = windows(len(v))
        m = {"motion": [motion(v, w) for w in wins]}
        if name.startswith("real_") and cid in gt_cache:
            g, n = gt_cache[cid], gt_meta[cid]["gt_frames"]
            m["gt_psnr"], m["gt_lpips"] = [], []
            for w in wins:
                sl = slice(w.start, min(w.stop, n))
                ok = sl.stop > sl.start
                m["gt_psnr"].append(frame_psnr(v[sl], g[sl]) if ok else None)
                m["gt_lpips"].append(lp(v[sl], g[sl]) if ok else None)
        metrics[f"{cid}/{name}"] = m
    for cid, g in gt_cache.items():
        n = gt_meta[cid]["gt_frames"]
        metrics[f"gt/{cid}"] = {"motion": [motion(g, slice(w.start, min(w.stop, n))) if min(w.stop, n) - w.start > 1 else None
                                           for w in windows(len(g))]}
    metrics_p.write_text(json.dumps(metrics, indent=1))

    # summary: per case × prompt (mean over seeds), per family × prompt, calibration
    rows = {}
    for key, ans in answers.items():
        if key.startswith(("gt/", "cross/")):
            continue
        cid, name = key.split("/")
        pk, d = name.split("_")[:2]  # prompt, loop depth ("N1")
        r = rows.setdefault((cid, pk, d), {"score": [], "attempted": [], "plausible": []})
        r["score"].append(ans["score"] if ans["score"] is not None else 0.0)
        r["attempted"].append(ans["attempted"] == "yes")
        r["plausible"].append(ans["plausible"] == "yes")
    mean = lambda x: float(np.mean(x)) if len(x) else None
    per_case = {f"{c}/{pk}/{d}": {k: mean(v) for k, v in r.items()} | {"n": len(r["score"])} for (c, pk, d), r in rows.items()}
    fam = {}
    for (c, pk, d), r in rows.items():
        f = fam.setdefault(f"{cases[c]['family']}/{pk}/{d}", {"score": [], "attempted": [], "plausible": []})
        for k in f:
            f[k] += r[k]
    calib = {k: [a_["completed"] == ("yes" if k == "real" else "no") for key, a_ in answers.items()
                 if key.startswith("gt/") and key.endswith("/" + k)] for k in ("real", "foreign")}
    cross = {}
    for key, ans in answers.items():
        if key.startswith("cross/"):
            _, cid, name = key.split("/")
            cross.setdefault(f"{cid}/{'/'.join(name.split('_')[:2])}", []).append(ans["score"] if ans["score"] is not None else 0.0)
    for k, v in cross.items():  # per_case[case/real/N<d>]["cross"] = the real clips judged against alt2, and vice versa
        if k in per_case:
            per_case[k]["cross"] = mean(v)
    summary = {"per_case": per_case, "per_family": {k: {kk: mean(vv) for kk, vv in v.items()} for k, v in fam.items()},
               "calibration": {"gt_real_caption_yes_rate": mean(calib["real"]), "gt_foreign_caption_no_rate": mean(calib["foreign"]),
                               "n": len(calib["real"])}}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["calibration"]), flush=True)


if __name__ == "__main__":
    main()
