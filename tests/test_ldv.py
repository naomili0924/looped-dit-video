import pytest
import torch

from ldv.config import TrainConfig
from ldv.diffusion import deep_supervision_weights, euler_sample, training_loss
from ldv.model import LoopedFluxT2V, ModelConfig
from ldv.probe import RidgeProbe, grid_coords, run_probe

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tiny(**kw) -> ModelConfig:
    base = dict(hidden_size=128, num_heads=2, axes_dim=[16, 16, 16, 16], context_in_dim=32, depth=1,
                depth_single_blocks=6, loop_split=(3, 2, 2), num_loops=4)
    base.update(kw)
    return ModelConfig(**base)


def inputs(b=2, t=3, h=8, w=8, l=7):
    torch.manual_seed(0)
    x = torch.randn(b, 16, t, h, w, device=DEV)
    tt = torch.rand(b, device=DEV)
    text = torch.randn(b, l, 32, device=DEV)
    mask = torch.ones(b, l, dtype=torch.long, device=DEV)
    mask[0, 4:] = 0
    return x, tt, text, mask


def test_shapes_and_exits():
    m = LoopedFluxT2V(tiny()).to(DEV)
    x, t, text, mask = inputs()
    out, exits = m(x, t, text, mask, exit_loops=(1, 2, 3))
    assert out.shape == x.shape and set(exits) == {1, 2, 3}
    assert all(e.shape == x.shape for e in exits.values())
    out8 = m(x, t, text, mask, num_loops=8)  # deeper than trained, shared weights
    assert out8.shape == x.shape


def test_gate_only_in_looped_stage():
    m = LoopedFluxT2V(tiny(use_attn_gate=True))
    gated = [i for i, b in enumerate(m.single_blocks) if b.use_attn_gate]
    assert gated == [2, 3]  # pre = 1 mode + 2 single, core = 2
    assert all(hasattr(m.single_blocks[i], "img_gate") for i in gated)
    assert not any(b.use_xsa for b in m.single_blocks)
    mx = LoopedFluxT2V(tiny(use_attn_gate=False, use_xsa=True))
    assert [i for i, b in enumerate(mx.single_blocks) if b.use_xsa] == [2, 3]


def test_untied_baseline_has_more_blocks():
    m = LoopedFluxT2V(tiny(share_loop_weights=False))
    assert len(m.single_blocks) == 2 + 2 * 4 + 2
    assert m.loop_blocks(1)[0] is not m.loop_blocks(2)[0]
    with pytest.raises(ValueError):
        m(*inputs(), num_loops=5)


def test_loop_count_changes_output_and_n1_is_plain_stack():
    torch.manual_seed(0)
    m = LoopedFluxT2V(tiny()).to(DEV)
    for p in m.parameters():  # break the zero-init so the blocks do something
        p.data.add_(torch.randn_like(p) * 0.02)
    x, t, text, mask = inputs()
    o1, o2 = m(x, t, text, mask, num_loops=1), m(x, t, text, mask, num_loops=2)
    assert not torch.allclose(o1, o2)
    # N = 1 equals running every block once in order.
    img, txt, ctx = m.encode(x, t, text, mask)
    img, txt = m._single(m.loop_blocks(1), img, txt, ctx)
    assert torch.allclose(m.decode(img, txt, ctx), o1, atol=1e-5)


def test_grad_reaches_shared_blocks_and_loss_finite():
    torch.manual_seed(0)
    m = LoopedFluxT2V(tiny(grad_checkpointing=True)).to(DEV).train()
    x, t, text, mask = inputs()
    for step in range(3):  # after two SGD steps the zero-init layers have moved
        loss, metrics = training_loss(m, x, text, mask, deep_supervision_weights(4, "final_plus_mean"))
        assert torch.isfinite(loss) and {"loss_final", "loss_exit1", "loss_exit3"} <= set(metrics)
        loss.backward()
        if step == 2:
            break
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-2)
                    p.grad = None
    core = m.loop_blocks(1)[0]
    assert core.q_proj.weight.grad is not None and core.q_proj.weight.grad.abs().sum() > 0
    assert core.img_gate.weight.grad.abs().sum() > 0


def test_deep_supervision_weights():
    w = deep_supervision_weights(4, "final_plus_mean")
    assert pytest.approx(sum(w)) == 2.0 and pytest.approx(w[-1] / w[0]) == 3.0


def test_sampler_runs():
    m = LoopedFluxT2V(tiny()).to(DEV)
    _, _, text, mask = inputs()
    out = euler_sample(m, text, mask, (16, 2, 4, 4), steps=3, cfg_scale=2.0, num_loops=2)
    assert out.shape == (2, 16, 2, 4, 4) and torch.isfinite(out).all()


def test_ridge_probe_recovers_linear_coords_and_rejects_noise():
    torch.manual_seed(0)
    grid = (4, 6, 6)
    c = grid_coords(grid, "cpu")
    n_clips = 12
    coords = c.repeat(n_clips, 1)
    groups = torch.arange(n_clips).repeat_interleave(c.shape[0])
    a = torch.randn(3, 64)
    linear = coords @ a + 0.01 * torch.randn(coords.shape[0], 64)
    noise = torch.randn(coords.shape[0], 64)
    test = groups >= 9
    p = RidgeProbe()
    good = p.fit_eval(linear[~test], coords[~test], groups[~test], linear[test], coords[test])
    bad = p.fit_eval(noise[~test], coords[~test], groups[~test], noise[test], coords[test])
    assert min(good["r2_t"], good["r2_h"], good["r2_w"]) > 0.99
    assert bad["r2_joint"] < 0.05


def test_run_probe_end_to_end():
    m = LoopedFluxT2V(tiny()).to(DEV)
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    x, _, text, mask = inputs(b=8, t=2, h=6, w=6)
    rows = run_probe(m, x, text, mask, num_loops=3, t_values=(0.5,), tokens_per_clip=40)
    assert [r["loop"] for r in rows] == [0, 1, 2, 3]
    assert all(k in rows[0] for k in ("r2_t", "r2_h", "r2_w", "r2_space", "r2_joint"))


def test_configs_load():
    for name in ("t2v_7b_webvid50k", "t2v_7b_webvid1m", "tiny"):
        cfg = TrainConfig.from_yaml(f"configs/{name}.yml")
        assert cfg.model.loop_split[0] >= cfg.model.depth
    cfg = TrainConfig.from_yaml("configs/tiny.yml", {"model.num_loops": 2})
    assert cfg.model.num_loops == 2


def test_7b_layout_param_count():
    cfg = TrainConfig.from_yaml("configs/t2v_7b_webvid50k.yml").model
    with torch.device("meta"):
        m = LoopedFluxT2V(cfg)
    n = m.num_params()
    # FLUX 3 Action 7B (6.99B) minus action / action_cond / video_cond streams, smaller txt_in.
    assert 4.7e9 < n < 5.1e9, n


def test_abs_pos_embed_makes_position_decodable_at_init():
    from ldv.model import sincos_3d
    e = sincos_3d(128, (3, 4, 5), "cpu")
    assert e.shape == (60, 128)
    m = LoopedFluxT2V(tiny(abs_pos_embed=True)).to(DEV)
    x, _, text, mask = inputs(b=8, t=3, h=8, w=8)
    rows = run_probe(m, x, text, mask, num_loops=1, t_values=(0.5,), tokens_per_clip=48)
    assert rows[0]["r2_joint"] > 0.9  # h_0 carries the absolute position


def test_loader_reads_from_every_worker(tmp_path):
    """Regression: with several dataloader workers, each must yield samples (one shard each)."""
    import io
    import tarfile

    from ldv.data import make_loader

    for shard in range(2):
        with tarfile.open(tmp_path / f"s{shard}.tar", "w") as tar:
            for i in range(6):
                buf = io.BytesIO()
                torch.save(torch.full((16, 2, 4, 4), float(shard)), buf)
                for ext, data in (("latent.pth", buf.getvalue()), ("txt", f"clip {shard}-{i}".encode())):
                    info = tarfile.TarInfo(f"{shard}{i:03d}.{ext}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
    loader = iter(make_loader(str(tmp_path), batch_size=4, num_workers=2, shuffle_buffer=0))
    seen = set()
    for _ in range(4):  # round-robin over the two workers
        batch = next(loader)
        assert batch["latents"].shape == (4, 16, 2, 4, 4)
        seen |= {c.split()[1].split("-")[0] for c in batch["caption"]}
    assert seen == {"0", "1"}
