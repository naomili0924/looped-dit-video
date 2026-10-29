"""Action eval, VLM steps on the initial frame (Qwen3-VL-32B-Instruct, the judge model, greedy):

--mode scene     which of the episode's caption objects are visible in the context frame, and which other objects the
                 hands could act on (name + material class). -> <out>/scene.json {window: {...}}
--mode feasible  for every candidate prompt of a window (scripts/actioneval/propose.py): could the robot plausibly do
                 this from this frame (object in view, action physically possible on it)? -> <out>/feasible.json

    cd flux-action && PYTHONPATH=.. .venv/bin/python -m ldvflux.act_vlm --mode scene --pool ../results/actioneval/pool.json \
        --mirror /workspace/loop_dit_eval/data/actioneval/mirror --vlm /dev/shm/w/qwen3vl32b --out ../results/actioneval \
        --shard 0/2 --device cuda:0
Resumable: windows already answered are skipped; shards write scene_<i>.json / feasible_<i>.json, merged by propose.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from .rollout import load_episode_frames

MATERIALS = {
    "rigid": "cannot change shape (cup, bottle, box, tray, tool, toy car)",
    "articulated": "rigid part with a hinge, slider or button (door, drawer, lid, switch, faucet handle, oven knob)",
    "cloth": "foldable textile (shirt, towel, curtain, tablecloth)",
    "thin_flexible": "thin sheet, film or bag that crumples (paper, plastic bag, chip bag, cardboard sheet)",
    "elastic": "soft and squishable, springs back (sponge, plush toy, pillow, squeeze bottle)",
    "plastic": "mouldable, keeps the new shape (dough, clay)",
    "linear": "rope, cord, cable, ribbon",
    "liquid": "a liquid in an open container (water in a cup)",
    "granular": "loose pieces or powder (rice, beans, sugar, crumbs)",
    "divisible": "food that can be cut, peeled or torn (fruit, vegetable, bread, meat)",
    "fixture": "static surface the hand works on but does not move (table top, counter, wall, shelf, sink)",
}
SCENE_Q = (
    "This is the first frame from the head camera of a two-armed robot standing at a work area.\n"
    "1. List up to 12 objects in view that the robot's hands could act on (pick up, push, open, press, wipe, fold, "
    "pour from, ...), including furniture parts (drawer, door) and surfaces (table). Give each a short name as a "
    "person would say it (\"red cup\", \"paper bag\") and one material class:\n{mats}\n"
    "2. For each of these names, say whether that object is visible in the frame: {names}\n"
    "Answer with JSON only: {{\"objects\": [{{\"name\": \"...\", \"material\": \"...\"}}], \"visible\": {{\"<name>\": true | false}}}}")
FEAS_Q = (
    "This is the first frame from the head camera of a two-armed robot. For each numbered instruction, answer \"yes\" "
    "if the robot could plausibly carry it out starting from this scene within about 9 seconds: the object it names is "
    "visible here, and the action is physically possible on that object (an action that is unusual but possible counts "
    "as yes). Answer \"no\" if the object is not in view or the action makes no physical sense for it.\n{items}\n"
    "Answer with JSON only, one key per number and \"y\" or \"n\": {{\"1\": \"y\", \"2\": \"n\", ...}}")


class VLM:
    def __init__(self, path: str, device):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.device = device

    def ask(self, image: np.ndarray, text: str, max_new_tokens: int) -> tuple[dict, str]:
        return self.ask_many(image, [text], max_new_tokens)[0]

    @torch.no_grad()
    def ask_many(self, image: np.ndarray, texts: list[str], max_new_tokens: int) -> list[tuple[dict, str]]:
        """Several questions about one image in one batched generate (left-padded): decoding is memory-bound, so a
        batch of 8 costs about what one question does."""
        from PIL import Image

        img = Image.fromarray(image)
        convs = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": t}]}] for t in texts]
        self.proc.tokenizer.padding_side = "left"
        inputs = self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        res = []
        for raw in self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True):
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                res.append(((json.loads(m.group(0)) if m else {}), raw))
            except json.JSONDecodeError:
                res.append(({}, raw))
        return res


def context_frame(w: dict, mirror: Path) -> np.ndarray:
    """The k = 1 context frame of a window: uint8 (H, W, 3) at the rollout resolution."""
    return load_episode_frames(w, 2, None, mirror, None)[0].transpose(1, 2, 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["scene", "feasible"], required=True)
    p.add_argument("--pool", required=True, help="pool.json (scene) or candidates.json (feasible)")
    p.add_argument("--mirror", required=True)
    p.add_argument("--vlm", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk", type=int, default=30, help="feasible: prompts per question")
    p.add_argument("--batch", type=int, default=8, help="feasible: questions per batched generate")
    a = p.parse_args()
    shard, n_shards = map(int, a.shard.split("/"))
    out = Path(a.out)
    res_p = out / f"{a.mode}_{shard}.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    windows = [w for i, w in enumerate(json.loads(Path(a.pool).read_text())["windows"]) if i % n_shards == shard]
    vlm = VLM(a.vlm, torch.device(a.device))
    mats = "\n".join(f"  {k}: {v}" for k, v in MATERIALS.items())
    for w in windows:
        if w["id"] in res:
            continue
        img = context_frame(w, Path(a.mirror))
        if a.mode == "scene":
            names = [o["span"] for o in w["objects"]]
            ans, raw = vlm.ask(img, SCENE_Q.format(mats=mats, names=json.dumps(names)), 600)
            res[w["id"]] = {"objects": ans.get("objects", []), "visible": ans.get("visible", {}), "raw": raw}
            print(f"[scene] {w['id']} {len(res[w['id']]['objects'])} objects, visible {res[w['id']]['visible']}", flush=True)
        else:
            prompts = [c["prompt"] for c in w["candidates"]]
            got = {}
            parts = [prompts[i:i + a.chunk] for i in range(0, len(prompts), a.chunk)]
            for b in range(0, len(parts), a.batch):
                qs = [FEAS_Q.format(items="\n".join(f"{j + 1}. {t}" for j, t in enumerate(part))) for part in parts[b:b + a.batch]]
                for part, (ans, raw) in zip(parts[b:b + a.batch], vlm.ask_many(img, qs, 8 * a.chunk + 20)):
                    for j, t in enumerate(part):
                        v = str(ans.get(str(j + 1), "")).strip().lower()
                        got[t] = "yes" if v in ("y", "yes") else "no" if v in ("n", "no") else ""
            res[w["id"]] = got
            print(f"[feasible] {w['id']} {sum(v == 'yes' for v in got.values())}/{len(got)} yes", flush=True)
        res_p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("done", res_p, flush=True)


if __name__ == "__main__":
    main()
