"""Ridge-regression probe of spatio-temporal position across loop depth (diagnostic only).

Looped-DiT (Sec. 3, Fig. 3b) fits a ridge probe at each loop depth to predict each image
token's 2D patch-grid coordinates from its hidden state; the falling R^2 shows that repeated
attention erodes local spatial information. For video the target is the 3D grid coordinate
(t, h, w) of each video token. R^2 is reported per axis, for space (h, w) and jointly (t, h, w).

The probe is never part of the training loss.
"""

from __future__ import annotations

import torch

from .diffusion import sample_t  # noqa: F401  (re-exported for scripts)

DEFAULT_LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
AXES = ("t", "h", "w")


def grid_coords(grid: tuple[int, int, int], device) -> torch.Tensor:
    """[T*H*W, 3] coordinates in token order (t-major, as in the model), each scaled to [0, 1]."""
    t, h, w = grid
    c = torch.cartesian_prod(torch.arange(t), torch.arange(h), torch.arange(w)).float()
    c = c / torch.tensor([max(t - 1, 1), max(h - 1, 1), max(w - 1, 1)]).float()
    return c.to(device)


def _r2(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-column coefficient of determination."""
    ss_res = (y - pred).pow(2).sum(0)
    ss_tot = (y - y.mean(0)).pow(2).sum(0).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


class RidgeProbe:
    """Closed-form multi-output ridge, solved for many lambdas from one eigendecomposition.
    lambda is chosen per output on a validation split of the fit data, then refit on all of it."""

    def __init__(self, lambdas=DEFAULT_LAMBDAS, val_frac: float = 0.2, seed: int = 0):
        self.lambdas, self.val_frac, self.seed = tuple(lambdas), val_frac, seed

    @staticmethod
    def _solve_all(x: torch.Tensor, y: torch.Tensor, lambdas) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        mx, my = x.mean(0), y.mean(0)
        xc, yc = x - mx, y - my
        # Scale lambda by the mean feature variance so the grid is meaningful for any width.
        evals, evecs = torch.linalg.eigh(xc.T @ xc)
        xty = evecs.T @ (xc.T @ yc)
        scale = evals.clamp_min(0).mean()
        ws = [evecs @ (xty / (evals + lam * scale)[:, None]) for lam in lambdas]
        return mx, my, scale, ws

    def fit_eval(self, x_fit, y_fit, groups_fit, x_test, y_test) -> dict[str, float]:
        x_fit, y_fit = x_fit.double(), y_fit.double()
        x_test, y_test = x_test.double(), y_test.double()
        # Validation split by group (clip), so lambda is not chosen on tokens of seen clips.
        g = torch.unique(groups_fit)
        perm = g[torch.randperm(len(g), generator=torch.Generator().manual_seed(self.seed))]
        n_val = max(1, int(len(g) * self.val_frac)) if len(g) > 1 else 0
        val_groups = perm[:n_val].to(groups_fit.device)
        is_val = torch.isin(groups_fit, val_groups) if n_val else torch.zeros_like(groups_fit, dtype=torch.bool)
        if is_val.any() and (~is_val).any():
            mx, my, _, ws = self._solve_all(x_fit[~is_val], y_fit[~is_val], self.lambdas)
            r2s = torch.stack([_r2((x_fit[is_val] - mx) @ w + my, y_fit[is_val]) for w in ws])  # [L, out]
            best = r2s.argmax(0)
        else:
            best = torch.full((y_fit.shape[1],), len(self.lambdas) // 2)
        mx, my, _, ws = self._solve_all(x_fit, y_fit, self.lambdas)
        w = torch.stack([ws[int(best[j])][:, j] for j in range(y_fit.shape[1])], dim=1)
        pred = (x_test - mx) @ w + my
        per = _r2(pred, y_test)
        out = {f"r2_{a}": float(per[i]) for i, a in enumerate(AXES)}
        out["r2_space"] = float(per[1:].mean())
        out["r2_joint"] = float(per.mean())
        out.update({f"lambda_{a}": float(self.lambdas[int(best[i])]) for i, a in enumerate(AXES)})
        return out


@torch.no_grad()
def collect_states(model, latents, text, text_mask, t: float, num_loops: int, tokens_per_clip: int,
                   noise_scale: float = 1.0, seed: int = 0, convention: str = "x0"):
    """Noise clips to flow time t and return per-depth features.

    Returns (states, coords, groups): states[r] is [N, D] video-token features after r loops
    (r = 0 is the pre-loop output h_0), coords [N, 3], groups [N] clip index."""
    g = torch.Generator(device=latents.device).manual_seed(seed)
    b = latents.shape[0]
    tt = torch.full((b,), t, device=latents.device)
    noise = torch.randn(latents.shape, device=latents.device, generator=g) * noise_scale
    # x0 convention: t = 1 is data; flux convention: t = 1 is noise (same mix at t = 0.5)
    x_t = latents * t + noise * (1 - t) if convention == "x0" else noise * t + latents * (1 - t)
    dtype = next(model.parameters()).dtype
    amp = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    with torch.autocast("cuda", dtype=amp, enabled=latents.is_cuda):
        _, states = model(x_t, tt, text, text_mask, num_loops=num_loops, return_states=True)
    _, grid = model.patchify(latents[:1])
    coords_one = grid_coords(grid, latents.device)
    n_tok = coords_one.shape[0]
    k = min(tokens_per_clip, n_tok)
    idx = torch.stack([torch.randperm(n_tok, generator=g, device=latents.device)[:k] for _ in range(b)])  # [B, k]
    feats = [s.float().gather(1, idx[..., None].expand(-1, -1, s.shape[-1])).reshape(b * k, -1) for s in states]
    coords = coords_one[idx].reshape(b * k, 3)
    groups = torch.arange(b, device=latents.device).repeat_interleave(k)
    return feats, coords, groups


@torch.no_grad()
def run_probe(model, latents, text, text_mask, *, num_loops: int, t_values=(0.25, 0.5, 0.75),
              test_frac: float = 0.25, tokens_per_clip: int = 256, noise_scale: float = 1.0,
              micro_batch: int = 4, lambdas=DEFAULT_LAMBDAS, convention: str = "x0") -> list[dict]:
    """Fit the probe on some clips and evaluate on held-out clips, for each loop depth and t.
    Returns rows {"t", "loop", r2_t, r2_h, r2_w, r2_space, r2_joint, lambda_*}."""
    was_training = model.training
    model.eval()
    b = latents.shape[0]
    n_test = max(1, int(round(b * test_frac)))
    rows = []
    for t in t_values:
        per_depth, coords, groups = None, [], []
        for s in range(0, b, micro_batch):
            sl = slice(s, s + micro_batch)
            f, c, g = collect_states(model, latents[sl], text[sl], text_mask[sl], t, num_loops,
                                     tokens_per_clip, noise_scale, seed=s, convention=convention)
            per_depth = [[x] for x in f] if per_depth is None else [p + [x] for p, x in zip(per_depth, f)]
            coords.append(c)
            groups.append(g + s)
        coords, groups = torch.cat(coords), torch.cat(groups)
        test = groups >= b - n_test
        probe = RidgeProbe(lambdas)
        for r, feats in enumerate(per_depth):
            x = torch.cat(feats)
            res = probe.fit_eval(x[~test], coords[~test], groups[~test], x[test], coords[test])
            rows.append({"t": t, "loop": r, **res})
    model.train(was_training)
    return rows
