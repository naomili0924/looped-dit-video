"""Looped text-to-video DiT on the FLUX 3 Action backbone.

The backbone is FLUX 3 Action's ``JointSingleSeq`` (black-forest-labs/flux-action,
Apache-2.0) with the robot parts removed: no action / action_cond / state streams,
no video_cond (history-frame) stream, and no pooled-vector input. What remains is one
``video`` content stream plus text:

    mode blocks    per-stream self-attention (video and text separately), adaLN on t
    single blocks  joint attention over [text; video], adaLN on t
    final layer    adaLN + linear back to latent patches

The Looped-DiT recipe (Chng et al., 2026, arXiv 2609.40305) then partitions the
blocks (mode blocks first, then single blocks) into three stages:

    pre-loop   A   blocks[:pre]            run once  (all mode blocks + some single)
    looped     B   the next `core` blocks  run N times with shared weights
    post-loop  C   the last `post` blocks  run once, followed by the final layer

    h_0 = A(x),   h_r = B(h_{r-1}),   x0_hat(r) = C(h_r),   r = 1..N

Every loop state can be decoded through C (deep supervision, variable inference depth).
Self-modulating attention acts on the looped blocks only: a head-wise sigmoid gate
(Qiu et al., 2025) by default; exclusive self-attention (XSA) is available as an ablation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    latent_channels: int = 16  # Wan2.1 VAE
    patch_size: tuple[int, int, int] = (1, 2, 2)  # (t, h, w) patch over latents
    context_in_dim: int = 1024  # FLAN-T5-Large
    hidden_size: int = 3072
    num_heads: int = 24
    mlp_ratio: float = 3.0
    depth: int = 5  # mode (per-stream) blocks
    depth_single_blocks: int = 28
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])  # RoPE over (t, h, w, l)
    theta: int = 10000
    qkv_bias: bool = False
    # Looped-DiT
    loop_split: tuple[int, int, int] = (12, 9, 12)  # (pre, core, post) over [mode blocks, single blocks]
    num_loops: int = 4
    share_loop_weights: bool = True
    use_attn_gate: bool = True
    use_xsa: bool = False
    # Absolute 3D sincos position embedding added to video tokens, as MiniT2I/Looped-DiT add a 2D one.
    # FLUX itself is RoPE-only (position never enters the hidden states directly), so this is off by default.
    abs_pos_embed: bool = False
    # FLUX 3 Action conventions (used when loading its pretrained weights):
    text_timestep_zero: bool = False  # text tokens are modulated with t = 0, video tokens with t
    time_id_stride: int = 1  # RoPE time id per latent frame (FLUX: 10 ms units, 4 frames / fps -> 50 at 8 fps)
    attn_gate_init: float = 0.0  # attention-gate bias at init; 0 = half open (paper), >0 = mostly open
    grad_checkpointing: bool = False

    def __post_init__(self):
        self.patch_size = tuple(self.patch_size)
        self.loop_split = tuple(self.loop_split)
        pre, core, post = self.loop_split
        if min(self.loop_split) < 1:
            raise ValueError(f"loop_split must be three positive block counts, got {self.loop_split}")
        if pre + core + post != self.depth + self.depth_single_blocks:
            raise ValueError(
                f"loop_split {self.loop_split} must cover all {self.depth} mode + "
                f"{self.depth_single_blocks} single blocks"
            )
        if pre < self.depth:
            raise ValueError("the looped stage must consist of single (joint) blocks: pre >= depth")
        if self.hidden_size % self.num_heads or sum(self.axes_dim) != self.hidden_size // self.num_heads:
            raise ValueError("axes_dim must sum to the head dim")
        if self.num_loops < 1:
            raise ValueError("num_loops must be >= 1")

    @property
    def in_channels(self) -> int:
        return self.latent_channels * math.prod(self.patch_size)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["patch_size"], d["loop_split"] = list(self.patch_size), list(self.loop_split)
        return d


# ---------------------------------------------------------------------------
# FLUX building blocks (ported from flux_action/models/transformer.py)
# ---------------------------------------------------------------------------


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.double(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta, self.axes_dim = theta, list(axes_dim)

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat([rope(ids[..., i], d, self.theta) for i, d in enumerate(self.axes_dim)], dim=-3)
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x.to(dtype) * self.scale.to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class Modulation(nn.Module):
    """shift, scale, gate from the timestep vector (one per stage and stream, as in FLUX)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        return tuple(out.chunk(3, dim=-1))


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale) * self.norm_final(x) + shift)


def exclusive_self_attention(out: Tensor, v: Tensor) -> Tensor:
    """XSA: remove from each token's attention output its component along the token's own value.
    out, v: [B, H, L, D]."""
    v_hat = F.normalize(v.float(), dim=-1)
    out_f = out.float()
    return (out_f - (out_f * v_hat).sum(dim=-1, keepdim=True) * v_hat).to(out.dtype)


class _AttnMLPBlock(nn.Module):
    """Shared body of FLUX's ModeBlock / SingleStreamBlock: parallel attention + SwiGLU MLP
    on one adaLN-modulated pre-norm input, with a gated residual."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.mlp_in = nn.Linear(hidden_size, self.mlp_hidden_dim * 2, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(self.head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def _qkv(self, x_mod: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v = (rearrange(p(x_mod), "B L (H D) -> B H L D", H=self.num_heads)
                   for p in (self.q_proj, self.k_proj, self.v_proj))
        q, k = self.norm(q, k, v)
        return q, k, v

    def _mlp(self, x_mod: Tensor) -> Tensor:
        a, b = self.mlp_in(x_mod).chunk(2, dim=-1)
        return self.mlp_out(F.silu(a) * b)


class ModeBlock(_AttnMLPBlock):
    """Per-stream block: self-attention within one stream only."""

    def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        shift, scale, gate = mod
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        q, k, v = self._qkv(x_mod)
        q, k = apply_rope(q, k, pe)
        attn = rearrange(F.scaled_dot_product_attention(q, k, v), "B H L D -> B L (H D)")
        return x + gate * (self.attn_out(attn) + self._mlp(x_mod))


class SingleStreamBlock(_AttnMLPBlock):
    """Joint block over [text; video] with per-stream adaLN. In the looped stage it may carry
    self-modulating attention: a modality-specific head-wise sigmoid gate on the attention
    output, G = sigmoid(W_g u + b_g) with u the block's normalized input, and/or XSA."""

    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias, use_attn_gate=False, use_xsa=False,
                 attn_gate_init=0.0):
        super().__init__(hidden_size, num_heads, mlp_ratio, qkv_bias)
        self.use_attn_gate, self.use_xsa = use_attn_gate, use_xsa
        if use_attn_gate:
            # Zero bias: gates start half open on average (as in Looped-DiT). A pretrained model
            # starts them mostly open (attn_gate_init > 0) so its function is preserved at init.
            self.img_gate = nn.Linear(hidden_size, num_heads)
            self.txt_gate = nn.Linear(hidden_size, num_heads)
            nn.init.constant_(self.img_gate.bias, attn_gate_init)
            nn.init.constant_(self.txt_gate.bias, attn_gate_init)

    def forward(self, img: Tensor, txt: Tensor, pe: Tensor, mod_img, mod_txt) -> tuple[Tensor, Tensor]:
        lt = txt.shape[1]
        img_mod = (1 + mod_img[1]) * self.pre_norm(img) + mod_img[0]
        txt_mod = (1 + mod_txt[1]) * self.pre_norm(txt) + mod_txt[0]
        combined = torch.cat((txt_mod, img_mod), dim=1)
        q, k, v = self._qkv(combined)
        q, k = apply_rope(q, k, pe)
        attn = F.scaled_dot_product_attention(q, k, v)  # [B, H, L, D]
        if self.use_xsa:
            attn = exclusive_self_attention(attn, v)
        if self.use_attn_gate:
            g = torch.cat((self.txt_gate(txt_mod), self.img_gate(img_mod)), dim=1)  # [B, L, H]
            attn = attn * torch.sigmoid(g).transpose(1, 2).unsqueeze(-1).to(attn.dtype)
        attn = rearrange(attn, "B H L D -> B L (H D)")
        out = self.attn_out(attn) + self._mlp(combined)
        txt_out, img_out = out[:, :lt], out[:, lt:]
        return img + mod_img[2] * img_out, txt + mod_txt[2] * txt_out


# ---------------------------------------------------------------------------
# Positions and patching
# ---------------------------------------------------------------------------


def video_ids(t: int, h: int, w: int, batch: int, device, time_stride: int = 1) -> Tensor:
    """(t, h, w, l=0) ids per video token, FLUX's prc_vid layout."""
    ids = torch.cartesian_prod(
        torch.arange(t, device=device) * time_stride, torch.arange(h, device=device),
        torch.arange(w, device=device), torch.zeros(1, dtype=torch.long, device=device),
    )
    return ids[None].expand(batch, -1, -1)


def sincos_3d(dim: int, grid: tuple[int, int, int], device) -> Tensor:
    """[T*H*W, dim] fixed sincos embedding of (t, h, w), dim split as evenly as possible (even sizes)."""
    d_t = (dim // 3) // 2 * 2
    d_h = ((dim - d_t) // 2) // 2 * 2
    d_w = dim - d_t - d_h
    coords = torch.cartesian_prod(*(torch.arange(n, device=device, dtype=torch.float32) for n in grid))
    parts = []
    for i, d in enumerate((d_t, d_h, d_w)):
        omega = 1.0 / (10000 ** (torch.arange(d // 2, device=device, dtype=torch.float32) / max(d // 2, 1)))
        a = coords[:, i : i + 1] * omega[None]
        parts += [a.sin(), a.cos()]
    return torch.cat(parts, dim=1)


def text_ids(length: int, batch: int, device) -> Tensor:
    """(0, 0, 0, l) ids per text token, FLUX's prc_txt layout."""
    ids = torch.zeros(length, 4, dtype=torch.long, device=device)
    ids[:, 3] = torch.arange(length, device=device)
    return ids[None].expand(batch, -1, -1)


# ---------------------------------------------------------------------------
# Looped model
# ---------------------------------------------------------------------------


class LoopedFluxT2V(nn.Module):
    """Predicts clean video latents x0 [B, C, T, H, W] from noisy latents, the flow time t
    (0 = noise, 1 = data) and T5 text states."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        pre, core, post = cfg.loop_split
        self.pre, self.core, self.post = pre, core, post
        self.pe_embedder = EmbedND(cfg.theta, cfg.axes_dim)
        self.emb_in = nn.Linear(cfg.in_channels, h, bias=False)
        self.txt_in = nn.Linear(cfg.context_in_dim, h, bias=False)
        self.time_in = MLPEmbedder(256, h)
        # Replaces T5 states at padded positions, and everywhere for the CFG null prompt.
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.context_in_dim) * 0.02)
        self.early_mod = nn.ModuleDict({"video": Modulation(h), "txt": Modulation(h)})
        self.single_mod = nn.ModuleDict({"video": Modulation(h), "txt": Modulation(h)})
        self.video_mode_blocks = nn.ModuleList(
            ModeBlock(h, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias) for _ in range(cfg.depth))
        self.txt_mode_blocks = nn.ModuleList(
            ModeBlock(h, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias) for _ in range(cfg.depth))
        # Single blocks laid out as [pre-loop single | core (x N if untied) | post].
        n_pre_single = pre - cfg.depth
        looped = core if cfg.share_loop_weights else core * cfg.num_loops
        n_single = n_pre_single + looped + post
        self.single_blocks = nn.ModuleList(
            SingleStreamBlock(
                h, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias,
                use_attn_gate=cfg.use_attn_gate and n_pre_single <= i < n_pre_single + looped,
                use_xsa=cfg.use_xsa and n_pre_single <= i < n_pre_single + looped,
                attn_gate_init=cfg.attn_gate_init,
            )
            for i in range(n_single)
        )
        self.n_pre_single, self.looped = n_pre_single, looped
        self.final_layer = LastLayer(h, cfg.in_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for block in self.single_blocks:
            if block.use_attn_gate:
                nn.init.constant_(block.img_gate.bias, self.cfg.attn_gate_init)
                nn.init.constant_(block.txt_gate.bias, self.cfg.attn_gate_init)
        for mod in (*self.early_mod.values(), *self.single_mod.values()):
            nn.init.zeros_(mod.lin.weight)  # adaLN-zero: blocks start as identity
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_layer.linear.weight)  # zero x0 prediction at init
        nn.init.normal_(self.time_in.in_layer.weight, std=0.02)
        nn.init.normal_(self.time_in.out_layer.weight, std=0.02)

    # -- patching ----------------------------------------------------------
    def patchify(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        pt, ph, pw = self.cfg.patch_size
        b, c, t, hh, ww = x.shape
        grid = (t // pt, hh // ph, ww // pw)
        tokens = rearrange(x, "b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)", pt=pt, ph=ph, pw=pw)
        return tokens, grid

    def unpatchify(self, tokens: Tensor, grid: tuple[int, int, int]) -> Tensor:
        pt, ph, pw = self.cfg.patch_size
        t, h, w = grid
        return rearrange(tokens, "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
                         t=t, h=h, w=w, pt=pt, ph=ph, pw=pw)

    # -- stages ------------------------------------------------------------
    def loop_blocks(self, r: int) -> nn.ModuleList:
        start = self.n_pre_single if self.cfg.share_loop_weights else self.n_pre_single + (r - 1) * self.core
        return self.single_blocks[start : start + self.core]

    def post_blocks(self) -> nn.ModuleList:
        return self.single_blocks[len(self.single_blocks) - self.post :]

    def _run(self, block, *args):
        if self.cfg.grad_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def _single(self, blocks, img, txt, ctx):
        for block in blocks:
            img, txt = self._run(block, img, txt, ctx["pe"], ctx["mod_img"], ctx["mod_txt"])
        return img, txt

    def decode(self, img: Tensor, txt: Tensor, ctx: dict) -> Tensor:
        """Post-loop blocks and output head: a loop state -> x0 prediction."""
        img, _ = self._single(self.post_blocks(), img, txt, ctx)
        return self.unpatchify(self.final_layer(img, ctx["vec"]), ctx["grid"]).float()

    def encode(self, x: Tensor, t: Tensor, text: Tensor, text_mask: Tensor):
        """Embed inputs and run the pre-loop stage A. Returns (img, txt, ctx)."""
        b = x.shape[0]
        tokens, grid = self.patchify(x)
        text = torch.where(text_mask.bool()[:, :, None], text, self.mask_token.to(text.dtype))
        vec = self.time_in(timestep_embedding(t, 256).to(tokens.dtype))
        vec_txt = self.time_in(timestep_embedding(torch.zeros_like(t), 256).to(tokens.dtype)) if self.cfg.text_timestep_zero else vec
        img = self.emb_in(tokens)
        if self.cfg.abs_pos_embed:
            img = img + sincos_3d(img.shape[-1], grid, x.device).to(img.dtype)[None]
        txt = self.txt_in(text)
        pe_img = self.pe_embedder(video_ids(*grid, b, x.device, self.cfg.time_id_stride))
        pe_txt = self.pe_embedder(text_ids(text.shape[1], b, x.device))
        early_img, early_txt = self.early_mod["video"](vec), self.early_mod["txt"](vec_txt)
        for vb, tb in zip(self.video_mode_blocks, self.txt_mode_blocks):
            img = self._run(vb, img, pe_img, early_img)
            txt = self._run(tb, txt, pe_txt, early_txt)
        ctx = {
            "grid": grid, "vec": vec,
            "pe": torch.cat((pe_txt, pe_img), dim=2),
            "mod_img": self.single_mod["video"](vec), "mod_txt": self.single_mod["txt"](vec_txt),
        }
        img, txt = self._single(self.single_blocks[: self.n_pre_single], img, txt, ctx)
        return img, txt, ctx

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        text: Tensor,
        text_mask: Tensor,
        num_loops: int | None = None,
        exit_loops: tuple[int, ...] = (),
        return_states: bool = False,
    ):
        """x: noisy latents [B, C, T, H, W]; t: [B] flow time; text: T5 states [B, L, D];
        text_mask: [B, L], 1 for prompt tokens (all zero = unconditional).

        num_loops overrides the loop depth at inference; exit_loops lists intermediate depths
        r < num_loops to decode too, returning (final, {r: x0_hat(r)}). return_states returns
        (final, [video hidden states h_0..h_N]) for the ridge probe.
        """
        n = self.cfg.num_loops if num_loops is None else int(num_loops)
        if n < 1 or (not self.cfg.share_loop_weights and n > self.cfg.num_loops):
            raise ValueError(f"num_loops={n} not available (trained with {self.cfg.num_loops})")
        exits = sorted({int(r) for r in exit_loops})
        if any(not 1 <= r < n for r in exits):
            raise ValueError(f"exit_loops must lie in [1, {n}), got {exits}")

        img, txt, ctx = self.encode(x, t, text, text_mask)
        states, saved = [img.detach()] if return_states else None, {}
        for r in range(1, n + 1):
            img, txt = self._single(self.loop_blocks(r), img, txt, ctx)
            if r in exits:
                saved[r] = (img, txt)
            if return_states:
                states.append(img.detach())
        out = self.decode(img, txt, ctx)
        if return_states:
            return out, states
        if not exits:
            return out
        return out, {r: self.decode(*saved[r], ctx) for r in exits}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
