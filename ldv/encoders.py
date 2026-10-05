"""Frozen encoders: Wan2.1 video VAE (latents) and FLAN-T5 (text)."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, T5EncoderModel

WAN_VAE = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class WanVAE:
    """Wan2.1 causal video VAE: 4x temporal / 8x spatial compression, 16 latent channels.
    A clip of 1 + 4k frames maps to 1 + k latent frames. Latents are normalized with the
    VAE's per-channel mean/std so they are roughly unit variance."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16, repo: str = WAN_VAE):
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
        self.vae = self.vae.to(device).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        c = self.vae.config
        self.mean = torch.tensor(c.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(c.latents_std, device=device).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """video [B, 3, T, H, W] in [-1, 1] -> normalized latents [B, 16, T', H/8, W/8] (float32)."""
        video = video.to(self.device, torch.float32)
        with torch.autocast("cuda", dtype=self.dtype):
            mu = self.vae.encode(video).latent_dist.mean
        return (mu.float() - self.mean) / self.std

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """normalized latents -> video [B, 3, T, H, W] in [-1, 1]."""
        z = latents.to(self.device, torch.float32) * self.std + self.mean
        with torch.autocast("cuda", dtype=self.dtype):
            video = self.vae.decode(z).sample
        return video.float().clamp(-1, 1)


class TextEncoder:
    """Frozen FLAN-T5 encoder (Looped-DiT's text encoder); prompts padded to prompt_length."""

    def __init__(self, name: str, prompt_length: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.prompt_length, self.device = prompt_length, device
        self.tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=prompt_length)
        self.model = T5EncoderModel.from_pretrained(name, torch_dtype=dtype).to(device).eval().requires_grad_(False)

    def tokenize(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(prompts, max_length=self.prompt_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        return tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids.to(self.device),
                          attention_mask=attention_mask.to(self.device)).last_hidden_state

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self.tokenize(prompts)
        return self.encode(ids, mask), mask
