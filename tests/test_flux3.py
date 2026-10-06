"""Equivalence of LoopedFluxT2V (loops = 1, no gate) with the reference FLUX 3 Action transformer, and
the LoRA / loading helpers. The weight test needs the base checkpoint (skipped without it)."""
import os

import pytest
import torch

from ldv.flux3 import LoRALinear, add_lora, flux3_model_config, flux_training_loss, load_flux3_weights, \
    map_flux3_state_dict, set_trainable, time_id_stride
from ldv.model import LoopedFluxT2V

WEIGHTS = os.environ.get("FLUX3_WEIGHTS", "/dev/shm/ldv/flux3/base/flux-3-action-base.safetensors")


def test_key_mapping_and_stride():
    sd = {"single_blocks.3.q_proj.weight": torch.zeros(1), "content_mode_blocks.video.0.mlp_in.weight": torch.zeros(1),
          "content_mode_blocks.action_prediction.0.mlp_in.weight": torch.zeros(1), "vector_in.in_layer.weight": torch.zeros(1),
          "final_layer.video.linear.weight": torch.zeros(1), "emb_in.video_cond.weight": torch.zeros(1)}
    out, dropped = map_flux3_state_dict(sd)
    assert set(out) == {"single_blocks.3.q_proj.weight", "video_mode_blocks.0.mlp_in.weight", "final_layer.linear.weight"}
    assert len(dropped) == 3
    assert time_id_stride(8.0) == 50 and time_id_stride(15.0) == 27


def test_lora_starts_as_identity_and_trains_only_adapters():
    torch.manual_seed(0)
    cfg = flux3_model_config(hidden_size=128, num_heads=2, axes_dim=[16] * 4, depth=1, depth_single_blocks=6,
                             loop_split=(3, 2, 2), context_in_dim=32, num_loops=2)
    m = LoopedFluxT2V(cfg)
    x = torch.randn(1, 96, 2, 3, 4); t = torch.rand(1); text = torch.randn(1, 5, 32); mask = torch.ones(1, 5, dtype=torch.long)
    with torch.no_grad():
        ref = m(x, t, text, mask)
    n = add_lora(m, rank=4)
    assert n == 6 * (2 + 6)  # 6 targets per block, 2 mode blocks (video + txt) + 6 single blocks
    counts = set_trainable(m)
    with torch.no_grad():
        out = m(x, t, text, mask)
    assert torch.allclose(out, ref, atol=1e-6)  # B = 0 at init
    assert 0 < counts["trainable"] < 0.2 * counts["total"]
    lin = next(mod for mod in m.modules() if isinstance(mod, LoRALinear))
    assert not lin.base.weight.requires_grad and lin.lora_A.requires_grad
    loss, _ = flux_training_loss(m, x, text, torch.zeros(1, 5, 32), [1.0, 1.0])
    loss.backward()
    assert lin.lora_B.grad is not None and lin.base.weight.grad is None


@pytest.mark.skipif(not os.path.exists(WEIGHTS) or not torch.cuda.is_available(), reason="needs the FLUX 3 base weights and a GPU")
def test_matches_reference_transformer():
    from safetensors.torch import load_file
    from flux_action.models.positional import batched_prc_txt, batched_prc_vid
    from flux_action.models.transformer import JointSingleSeq, JointSingleSeqParams

    dev = torch.device("cuda")
    cfg = flux3_model_config(use_attn_gate=False, time_id_stride=50)
    with torch.device(dev):  # fp32: bf16 differences over 33 blocks are pure rounding noise (~3%)
        ours = LoopedFluxT2V(cfg)
    info = load_flux3_weights(ours, WEIGHTS)
    assert info["loaded"] == len(ours.state_dict()) - 1 and all(k.startswith("mask_token") for k in info["new_params"])
    ref = JointSingleSeq(JointSingleSeqParams()).to(dev)
    res = ref.load_state_dict(load_file(WEIGHTS, device="cpu"), strict=False)
    assert not res.unexpected_keys or all("action" in k for k in res.unexpected_keys)
    ref.eval(); ours.eval()

    g = torch.Generator(device=dev).manual_seed(0)
    T, H, W, L = 3, 4, 5, 80
    lat = torch.randn(1, 96, T, H, W, device=dev, generator=g)
    ctx = torch.randn(1, L, 20480, device=dev, generator=g) * 0.5
    t = torch.tensor([0.7], device=dev)
    video, video_ids = batched_prc_vid(lat, torch.arange(T, device=dev)[None] * 50)
    _, ctx_ids = batched_prc_txt(ctx)
    with torch.no_grad():
        out_ref = ref(ctx=ctx, ctx_ids=ctx_ids, vector=torch.zeros(1, 768, device=dev),
                      timesteps_ctx=torch.zeros(1, L, device=dev), x_video=video, x_video_ids=video_ids,
                      x_video_timesteps=torch.full((1, video.shape[1]), 0.7, device=dev))["x_video"]
        out_ours = ours(lat, t, ctx, torch.ones(1, L, dtype=torch.long, device=dev), num_loops=1)
    out_ref = out_ref.float().view(1, T, H, W, 96).permute(0, 4, 1, 2, 3)
    err = (out_ref - out_ours.float()).abs().max().item()
    scale = out_ref.abs().max().item()
    assert err < 1e-4 * scale, (err, scale)
