"""FSDP2 + AC (+ compile) with a block that runs several times per forward (the looped core): gradients must match
a plain single-process run. 2 ranks; CPU (gloo) or GPU (nccl).

    PYTHONPATH=/workspace/loop-dit flux-action/.venv/bin/torchrun --nproc_per_node 2 --master_port 29800 scripts/test_fsdp_loop.py [--cuda] [--compile]
"""
import argparse, os, torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from flux_action.models.transformer import JointSingleSeqParams
from flux_action.models.looped import LoopParams, LoopedJointSingleSeq, deep_supervision_weights
from flux_action.training.distributed import apply_activation_checkpointing, apply_compile, dit_blocks, clean_name

p = argparse.ArgumentParser(); p.add_argument("--cuda", action="store_true"); p.add_argument("--compile", action="store_true")
args = p.parse_args()
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dev = torch.device(f"cuda:{rank}" if args.cuda else "cpu")
if args.cuda: torch.cuda.set_device(dev)
dist.init_process_group("nccl" if args.cuda else "gloo", device_id=dev if args.cuda else None)
from torch.distributed.device_mesh import init_device_mesh
mesh = init_device_mesh("cuda" if args.cuda else "cpu", (world,))

torch.manual_seed(0)
params = JointSingleSeqParams(hidden_size=1024, num_heads=8, depth=2, depth_single_blocks=6)
loop = LoopParams(loop_split=(4, 2, 2), num_loops=4, attn_gate=True, attn_gate_init=4.0)
ref = LoopedJointSingleSeq(params, loop).float().to(dev)
sd = {k: v.clone() for k, v in ref.state_dict().items()}
d = torch.load("/dev/shm/runs/dump11/dit-inputs-rank0-call0001.pt", weights_only=False)
packed = {k: (v.float() if v.is_floating_point() else v).to(dev) for k, v in d["packed"].items()}
sl = d["seqlens"]; w = deep_supervision_weights(4, "final_plus_mean")
torch.manual_seed(1); target = torch.randn(1, 6912, 96, device=dev)

def loss_of(m):
    out = m(**packed, seqlens=sl, exit_loops=(1, 2, 3))
    return w[-1] * (out["x_video"].float() - target).pow(2).mean() + sum(
        w[r - 1] * (out["exits"][r]["x_video"].float() - target).pow(2).mean() for r in (1, 2, 3))

# reference: plain fp32 single process (same on every rank)
l_ref = loss_of(ref); l_ref.backward()
g_ref = {clean_name(n): p.grad.clone() for n, p in ref.named_parameters() if p.grad is not None}
ref.zero_grad(set_to_none=True)

m = LoopedJointSingleSeq(params, loop).float().to(dev); m.load_state_dict(sd)
apply_activation_checkpointing(m)
if args.compile: apply_compile(m)
mp = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=True)
for blocks, i in dit_blocks(m):
    fully_shard(blocks[i], mesh=mesh, mp_policy=mp, reshard_after_forward=False)
fully_shard(m, mesh=mesh, mp_policy=mp, reshard_after_forward=False)
l = loss_of(m); l.backward()
worst = 0.0; worst_name = ""
for n, p in m.named_parameters():
    if clean_name(n) not in g_ref:
        assert p.grad is None or p.grad.abs().sum().item() == 0.0, n
        continue
    g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
    diff = (g - g_ref[clean_name(n)]).abs().max().item()
    if diff > worst: worst, worst_name = diff, clean_name(n)
scale = max(v.abs().max().item() for v in g_ref.values())
if rank == 0:
    print(f"loss ref {l_ref.item():.6f} fsdp {l.item():.6f} | max grad diff {worst:.3e} at {worst_name} (grad scale {scale:.3e}) | compile={args.compile} dev={dev.type}")
dist.destroy_process_group()
