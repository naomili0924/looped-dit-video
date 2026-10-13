"""Prompt-adherence judge: Qwen3-VL (the FLUX text encoder's own checkpoint) watches a generated clip and answers
whether the robot's motion follows an instruction. Score = yes-rate with the true caption minus yes-rate with a
swapped caption from another task (a model that ignores the text scores ~0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_MODEL = None


def _load(text_encoder_dir: str, device):
    global _MODEL
    if _MODEL is None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = str(text_encoder_dir)
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        proc = AutoProcessor.from_pretrained(path)
        _MODEL = (model, proc)
    return _MODEL


@torch.no_grad()
def ask(model, proc, video: np.ndarray, caption: str, fps: float, device) -> bool:
    """video (T, H, W, 3) uint8 -> True if the judge answers yes."""
    frames = [video[i] for i in range(0, len(video), max(1, len(video) // 8))][:8]  # 8 evenly spaced frames
    messages = [{"role": "user", "content": [
        {"type": "video", "video": [torch.from_numpy(f).permute(2, 0, 1) for f in frames]},
        {"type": "text", "text": f"This is a first-person video from a robot's head camera. Does the robot's motion in "
                                 f"this video carry out the instruction \"{caption}\"? Answer with one word: yes or no."}]}]
    inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt", fps=fps).to(device)
    out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
    return text.startswith("yes")


def judge_prompt_adherence(videos_true, captions_true, videos_swapped, captions_swapped, *, fps, device,
                           text_encoder_dir) -> dict:
    model, proc = _load(text_encoder_dir, device)
    yes_true = [ask(model, proc, v, c, fps, device) for v, c in zip(videos_true, captions_true)]
    yes_sw = [ask(model, proc, v, c, fps, device) for v, c in zip(videos_swapped, captions_swapped)]
    # the swapped generations are also asked about their *own* (swapped) caption: did the model follow the text?
    return {"n": len(yes_true), "yes_rate_true_caption": float(np.mean(yes_true)),
            "yes_rate_swapped_caption": float(np.mean(yes_sw)),
            "adherence": float(np.mean(yes_true) - np.mean(yes_sw))}
