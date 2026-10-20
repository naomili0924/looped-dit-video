"""Random loop depth in training (PolicyConfig.loop_sampler): weight math, the shared draw, truncated backprop. CPU.

    PYTHONPATH=$PWD flux-action/.venv/bin/python scripts/test_random_depth.py
"""
import itertools
from types import SimpleNamespace

import torch
from torch import nn

from flux_action.config import PolicyConfig
from flux_action.models.looped import (LoopParams, LoopedJointSingleSeq, deep_supervision_weights,
                                       random_depth_exit_prob, random_depth_exits, random_depth_weights)
from flux_action.policy import FluxActionPolicy

close = lambda a, b: abs(a - b) < 1e-9

# 1. fixed N, no truncation: the per-step weights are the deep-supervision weights themselves (old behaviour)
for mode in ("final_plus_mean", "uniform", "sigmoid"):
    w = deep_supervision_weights(4, mode)
    got = random_depth_weights(4, 4, 4, w)
    assert list(got) == [1, 2, 3, 4] and all(close(got[r], w[r - 1]) for r in got), (mode, got)

# 2. expected weight of every exit over N ~ uniform(min..max) equals the target profile, with and without truncation
for (n_min, n_max), k, mode in itertools.product([(1, 8), (2, 8), (1, 2)], [None, 1, 2, 4], ["uniform", "sigmoid"]):
    if k is not None and k < n_min:  # rejected by the policy (exits below n_min - k + 1 would never get a loss)
        continue
    target = deep_supervision_weights(n_max, mode)
    expect = {r: 0.0 for r in range(1, n_max + 1)}
    for n in range(n_min, n_max + 1):
        for r, w in random_depth_weights(n, n_min, n_max, target, k).items():
            expect[r] += w / (n_max - n_min + 1)
    for r in expect:
        assert close(expect[r], target[r - 1]), ((n_min, n_max), k, mode, r, expect[r], target[r - 1])
    assert close(sum(expect.values()), 2.0)
assert random_depth_exits(8, 4) == [5, 6, 7, 8] and random_depth_exits(3, 4) == [1, 2, 3]
assert close(random_depth_exit_prob(1, 1, 8, None), 1.0) and close(random_depth_exit_prob(8, 1, 8, None), 1 / 8)

# 3. the draw: deterministic in (seed, update, micro-batch), uniform on min..max, independent of anything rank-local
cfg = PolicyConfig(loop={"num_loops": 4}, deep_supervision="uniform", loop_sampler={"min": 1, "max": 8, "seed": 0})
loop = LoopParams(num_loops=4)
def draw(step, micro, seed=0):
    fake = SimpleNamespace(config=cfg, loop_draw=(step, micro))
    cfg.loop_sampler["seed"] = seed
    return FluxActionPolicy._random_depth(fake, cfg.loop_sampler, loop)
a, b = draw(17, 0), draw(17, 0)  # two "ranks"
assert a == b
ns = [draw(s, m)[0] for s in range(1, 2001) for m in range(2)]
counts = [ns.count(n) for n in range(1, 9)]
assert min(counts) > 0.8 * len(ns) / 8 and max(counts) < 1.2 * len(ns) / 8, counts
assert ns != [draw(s, m, seed=1)[0] for s in range(1, 2001) for m in range(2)]
n, exits, w = draw(5, 1)
assert exits == list(range(1, n + 1)) and close(sum(w.values()), sum(2 / 8 / random_depth_exit_prob(r, 1, 8, None) for r in exits))

# 4. truncated backprop in the looped forward: toy stages (pre = one Linear, core = one Linear, post = identity)
class Toy(LoopedJointSingleSeq):
    def __init__(self):
        nn.Module.__init__(self)
        self.loop, self.n_pre_single, self.core = LoopParams(num_loops=4), 1, 1
        self.pre_lin, self.core_lin = nn.Linear(8, 8), nn.Linear(8, 8)
        self.single_blocks = [self.pre_lin, self.core_lin]
    @property
    def core_blocks(self):
        return [self.core_lin]
    def _prepare(self, ctx, *a):
        return ctx, {}
    def _run_blocks(self, blocks, state, st, *a, offset=0):
        return torch.tanh(blocks[0](state))
    def _decode(self, state, st, *a):
        return {"x_video": state}

toy = Toy()
x = torch.randn(4, 8)
def grads(n, k):
    toy.zero_grad(set_to_none=True)
    out = toy(x, None, None, None, num_loops=n, exit_loops=[r for r in random_depth_exits(n, k) if r < n], grad_loops=k)
    loss = out["x_video"].sum() + sum(o["x_video"].sum() for o in out.get("exits", {}).values())
    loss.backward()
    return toy.pre_lin.weight.grad, toy.core_lin.weight.grad, sorted(out.get("exits", {}))
pg, cg, ex = grads(6, None)
assert pg is not None and cg is not None and ex == [1, 2, 3, 4, 5]
pg, cg, ex = grads(6, 2)  # loops 1-4 without autograd: the pre stage gets no gradient, the core does
assert pg is None and cg is not None and ex == [5]
pg, cg, ex = grads(3, 4)  # k >= N: nothing truncated
assert pg is not None and ex == [1, 2]
# the truncated forward computes the same values
with torch.no_grad():
    full = toy(x, None, None, None, num_loops=6)["x_video"]
assert torch.allclose(full, toy(x, None, None, None, num_loops=6, grad_loops=2)["x_video"])

print("random-depth tests passed; uniform 1..8 counts over 4000 draws:", counts)
