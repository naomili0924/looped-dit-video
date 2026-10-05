"""Training configuration: a YAML file with a `model:` section (ModelConfig) plus flat
training keys (TrainConfig). Strings may reference ${VAR} environment variables."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .model import ModelConfig

DEFAULT_ROOTS = {"DATA_ROOT": "/dev/shm/ldv", "OUTPUT_ROOT": "/dev/shm/ldv/outputs"}


def expand_vars(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name = m.group(1)
            if name in os.environ:
                return os.environ[name]
            if name in DEFAULT_ROOTS:
                return DEFAULT_ROOTS[name]
            raise KeyError(f"undefined variable ${{{name}}} in {value!r}")
        return re.sub(r"\$\{(\w+)\}", sub, value)
    if isinstance(value, list):
        return [expand_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_vars(v) for k, v in value.items()}
    return value


def _coerce(obj, values: dict[str, Any]):
    names = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"unknown config key for {type(obj).__name__}: {key}")
        default = getattr(obj, key)
        if type(default) in (int, float) and isinstance(value, str):  # YAML reads "2e-4" as str
            value = type(default)(float(value))
        elif type(default) is float and type(value) is int:
            value = float(value)
        setattr(obj, key, value)
    return obj


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)

    # Frozen encoders
    text_encoder: str = "google/flan-t5-large"
    prompt_length: int = 128
    vae: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # Data: latent shards from scripts/prepare_webvid.py
    train_shards: list[str] = field(default_factory=lambda: ["${DATA_ROOT}/webvid50k/shards"])
    probe_shards: list[str] = field(default_factory=list)  # held-out shards for the ridge probe
    shuffle_buffer: int = 2000
    num_workers: int = 4

    # Looped-DiT training components
    deep_supervision: bool = True
    deep_supervision_weighting: str = "final_plus_mean"

    # Flow matching (Looped-DiT values except noise_scale, see README)
    noise_scale: float = 1.0
    t_logit_mean: float = -0.8
    t_logit_std: float = 0.8
    label_drop_rate: float = 0.1

    # Optimization
    batch_size: int = 64
    micro_batch_size: int = 8
    num_steps: int = 20_000
    warmup_steps: int = 5_000
    learning_rate: float = 1e-4
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    max_grad_norm: float = 0.1
    optimizer: str = "adamw8bit"  # adamw8bit | adamw
    ema_decay: float = 0.9999
    ema_every: int = 10
    seed: int = 42
    compile: bool = False

    # Logging, sampling, probing, checkpoints
    log_every: int = 10
    sample_every: int = 2_000
    sample_steps: int = 50
    sample_cfg: float = 6.0
    sample_prompts: list[str] = field(default_factory=lambda: [
        "Aerial shot of a winter forest covered in snow.",
        "Waves crashing on a rocky beach at sunset.",
        "A woman walking a dog in a park, slow motion.",
        "Timelapse of clouds moving over a city skyline.",
    ])
    probe_every: int = 2_000
    probe_clips: int = 32
    probe_max_loops: int = 8
    ckpt_every: int = 1_000
    keep_last: int = 1
    hub_repo: str = ""  # optional HF Hub model repo to push EMA weights to at each checkpoint

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainConfig:
        values = dict(values)
        model = _coerce(ModelConfig(), values.pop("model", {}) or {})
        model.__post_init__()
        cfg = _coerce(cls(model=model), values)
        cfg.train_shards = expand_vars(cfg.train_shards)
        cfg.probe_shards = expand_vars(cfg.probe_shards)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key, value in (overrides or {}).items():  # "model.num_loops=2" style keys
            target = data
            *parents, leaf = key.split(".")
            for p in parents:
                target = target.setdefault(p, {})
            target[leaf] = value
        return cls.from_dict(expand_vars(data))

    def validate(self) -> None:
        if self.deep_supervision and self.model.num_loops < 2:
            raise ValueError("deep_supervision needs model.num_loops >= 2")
        if self.optimizer not in ("adamw8bit", "adamw"):
            raise ValueError(f"unknown optimizer {self.optimizer!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"] = self.model.to_dict()
        return d
