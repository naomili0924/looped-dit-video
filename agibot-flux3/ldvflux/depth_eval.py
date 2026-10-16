"""Where does extra loop depth go: background or motion?  (Looped-DiT retrofit, Phase 5 analysis)

For a set of evaluation cases (k context frames + a task prompt) the looped model is sampled at several loop depths
N from the *same* noise, optionally also with alternative prompts for the same context frames. The outputs are then
split into a moving region and a static background and compared:

  per depth N (real prompt, ground truth available)
    psnr_bg / lpips          background pixels: does more depth clean up the static scene?
    psnr_mov                 moving pixels (where the real clip moves)
    dir_cos, mag_ratio, region_cos   RAFT-flow agreement with the real motion (ldvflux.motion_eval)
  between consecutive depths (N1 -> N2, N2 -> N4, N4 -> N8), same noise
    delta_in / delta_out     mean |x_N' - x_N| inside / outside the moving region; concentration = in / out
    mass_in                  share of the total change that falls inside the moving region (compare with area_in)
  prompt sensitivity (alternative prompts, same context frames and noise, no ground truth)
    motion                   mean compensated flow magnitude of the output (does it move at all?)
    lpips_vs_real_prompt     how different the output is from the real-prompt output
    dir_cos_vs_real_prompt   cosine of the region-mean motion vectors of the two outputs (-1 = opposite motion)

Usage (generation needs a GPU; analysis runs on the saved clips, GPU or CPU):
    python -m ldvflux.depth_eval --config configs/looped_ft_384_r1.json --ema <EMA> --cases configs/eval_cases_gallery8.json \
        --depths 1 2 4 8 --k 1 9 --out results/depth_eval/loop2500 [--text-encoder <Qwen3-VL dir>] [--mirror <eval-splits dir>]
    python -m ldvflux.depth_eval --analyze results/depth_eval/loop2500 [--device cuda:0]
    python -m ldvflux.depth_eval --summarize results/depth_eval --md results/depth_eval/DEPTH.md
Plain (non-looped) checkpoints can be evaluated with --depths 1 only (reference rows).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .motion_eval import compare, compensate_camera, flow, load_raft
from .stream import EpisodeFetcher, decode_frames, load_splits, window_frame_indices

MOVE_THRESH = 25.0
FUTURE, STRIDE, HW = 33, 2, (384, 512)


# ---------------------------------------------------------------------------- cases
def resolve_caption(case: dict, splits: dict | None, mirror: Path | None) -> str:
    if case.get("caption"):
        return case["caption"]
    if splits is not None:
        for ep in splits["splits"][case["split"]]:
            if ep["episode"] == str(case["episode"]):
                for w in ep["windows"]:
                    if w["anchor"] == case["anchor"]:
                        return w["caption"]
    if mirror is not None:
        p = mirror / "windows" / f"{case['split']}.jsonl"
        if p.exists():
            for line in p.open():
                r = json.loads(line)
                if str(r["episode"]) == str(case["episode"]) and r["anchor"] == case["anchor"]:
                    return r["caption"]
        for f in (mirror / "fixed").glob(f"{case['split']}_seed*.json"):
            for r in json.loads(f.read_text())["picks"]:
                if str(r["episode"]) == str(case["episode"]) and r["anchor"] == case["anchor"]:
                    return r["caption"]
    raise KeyError(f"caption of {case['split']}/{case['episode']}@{case['anchor']} not found; give it in the cases file")


def load_case(case: dict, kmax: int, splits: dict | None, mirror: Path | None, fetcher: EpisodeFetcher | None) -> dict:
    """-> window dict as ldvflux.evaluate._load_windows builds it (frames at kmax context, uint8 (kmax+33, 3, H, W))."""
    idx = window_frame_indices(case["anchor"], kmax, STRIDE, FUTURE)
    if mirror is not None:
        data = (mirror / "videos" / case["split"] / f"{case['episode']}.mp4").read_bytes()
    else:
        ep = next(e for e in splits["splits"][case["split"]] if e["episode"] == str(case["episode"]))
        data = fetcher.fetch(ep)
    frames = decode_frames(data, set(idx), HW)
    missing = [i for i in idx if i not in frames]
    if missing:
        raise ValueError(f"{case}: frames {missing[:3]} missing")
    return {"frames": np.stack([frames[i] for i in idx]), "kmax": kmax, "caption": resolve_caption(case, splits, mirror),
            "episode": str(case["episode"]), "anchor": case["anchor"]}


# ---------------------------------------------------------------------------- generation
def write_clip(path: Path, frames: np.ndarray, fps: float = 15.0) -> None:
    t, h, w, _ = frames.shape
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                    "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv444p", str(path)],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)


def read_clip(path: Path) -> np.ndarray:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split(","))
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "image2pipe", "-vcodec", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def generate(args: argparse.Namespace) -> None:
    from .eval_phase4 import generate_windows, load_policy, to_uint8_video

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(args.cases).read_text())["cases"]
    mirror = Path(args.mirror) if args.mirror else None
    dev = torch.device(args.device)
    cfg = json.load(open(args.config))
    # alternative prompts are not in the pre-encoded caption table: --text-encoder selects the real Qwen3-VL encoder
    policy, cfg = load_policy(args.config, args.ema, dev, args.depths[0] if cfg["policy"].get("loop") else None,
                              args.text_encoder)
    looped = cfg["policy"].get("loop") is not None
    depths = args.depths if looped else [1]
    splits = None if mirror else load_splits(cfg["dataset_kwargs"]["splits"])
    fetcher = None if mirror else EpisodeFetcher()
    kmax = max(args.k)
    manifest = {"config": args.config, "ema": args.ema, "depths": depths, "k": args.k, "seed": args.seed, "steps": args.steps,
                "cfg": args.cfg, "shift": args.shift, "cases": []}
    for ci, case in enumerate(cases):
        w = load_case(case, kmax, splits, mirror, fetcher)
        prompts = [w["caption"]] + list(case.get("alt_prompts", []))
        rec = {**case, "caption": w["caption"], "prompts": prompts, "id": case.get("id", f"case{ci:02d}")}
        manifest["cases"].append(rec)
        for k in args.k:
            cdir = out / "clips" / rec["id"] / f"k{k}"
            cdir.mkdir(parents=True, exist_ok=True)
            write_clip(cdir / "gt.mp4", w["frames"][kmax : kmax + FUTURE].transpose(0, 2, 3, 1))
            write_clip(cdir / "context.mp4", w["frames"][kmax - k : kmax].transpose(0, 2, 3, 1)) if k else None
            for pi, prompt in enumerate(prompts):
                for d in depths:
                    path = cdir / f"p{pi}_N{d}.mp4"
                    if path.exists():
                        continue
                    policy.loops = d if looped else None
                    gens = generate_windows(policy, [{**w, "caption": prompt}], k, FUTURE, 1, args.seed + ci, args.steps,
                                            args.cfg, args.shift)
                    write_clip(path, to_uint8_video(gens[0][0]))
                    print(f"[{rec['id']} k{k} p{pi} N{d}] {prompt[:60]!r}", flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("generation done:", out)


# ---------------------------------------------------------------------------- analysis
def psnr(a, b, mask=None):
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        d = d[mask]
    return float(10 * np.log10(255.0 ** 2 / max(float(d.mean()), 1e-6)))


class Lpips:
    def __init__(self, device):
        import lpips

        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        x = torch.from_numpy(a).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)
        y = torch.from_numpy(b).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)
        return float(np.mean([float(self.net(x[s : s + 16], y[s : s + 16]).mean()) for s in range(0, len(x), 16)]))


def moving_mask(gt: np.ndarray) -> np.ndarray:
    """(T, H, W) bool: pixels that change vs the first future frame in the real clip."""
    return np.abs(gt.astype(np.float32) - gt[0:1].astype(np.float32)).mean(-1) > MOVE_THRESH


def motion_mask_from_flow(f: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """(T, H, W) bool from a flow sequence (T-1, 2, H, W): moving pixels (camera motion removed), last frame repeated."""
    m = np.linalg.norm(compensate_camera(f), axis=1) > tau
    return np.concatenate([m, m[-1:]], 0)


def analyze(args: argparse.Namespace) -> None:
    out = Path(args.analyze)
    man = json.loads((out / "manifest.json").read_text())
    dev = args.device
    lp = Lpips(dev)
    raft = load_raft(dev, large=not args.small)
    depths = man["depths"]
    rows = []
    for rec in man["cases"]:
        for k in man["k"]:
            cdir = out / "clips" / rec["id"] / f"k{k}"
            gt = read_clip(cdir / "gt.mp4")[1:]  # frame 0 is one step from the context: excluded everywhere
            m_gt = moving_mask(np.concatenate([read_clip(cdir / "gt.mp4")[:1], gt]))[1:]
            f_gt = flow(raft, read_clip(cdir / "gt.mp4"))[1:]
            outs = {(pi, d): read_clip(cdir / f"p{pi}_N{d}.mp4")[1:] for pi in range(len(rec["prompts"])) for d in depths}
            flows = {key: flow(raft, read_clip(cdir / f"p{key[0]}_N{key[1]}.mp4"))[1:] for key in outs}
            for pi, prompt in enumerate(rec["prompts"]):
                # region: real motion for the real prompt; the deepest output's own motion for alternative prompts
                region = m_gt if pi == 0 else motion_mask_from_flow(flows[(pi, depths[-1])])[: len(gt)]
                r3 = np.broadcast_to(region[..., None], gt.shape)
                area_in = float(region.mean())
                for d in depths:
                    x = outs[(pi, d)]
                    fc = compensate_camera(flows[(pi, d)])
                    row = {"case": rec["id"], "class": rec.get("class"), "k": k, "prompt_idx": pi, "prompt": prompt, "depth": d,
                           "area_in": area_in, "motion": float(np.linalg.norm(fc, axis=1)[region[:-1] if len(region) > len(fc) else region].mean()) if region.any() else 0.0}
                    if pi == 0:
                        row.update({"psnr": psnr(x, gt), "psnr_bg": psnr(x, gt, ~r3), "psnr_mov": psnr(x, gt, r3), "lpips": lp(x, gt),
                                    "static_psnr_bg": psnr(np.repeat(gt[0:1], len(gt), 0), gt, ~r3),
                                    "static_psnr_mov": psnr(np.repeat(gt[0:1], len(gt), 0), gt, r3),
                                    **{f"flow_{a}": b for a, b in compare(flows[(pi, d)], f_gt).items() if a != "moving_frac"}})
                    else:
                        x0 = outs[(0, d)]
                        f0 = compensate_camera(flows[(0, d)])
                        reg = region[: len(fc)]
                        va = np.stack([fc[t][:, reg[t]].mean(1) if reg[t].sum() > 20 else np.zeros(2) for t in range(len(fc))]).mean(0)
                        vb = np.stack([f0[t][:, reg[t]].mean(1) if reg[t].sum() > 20 else np.zeros(2) for t in range(len(fc))]).mean(0)
                        row.update({"lpips_vs_real_prompt": lp(x, x0),
                                    "dir_cos_vs_real_prompt": float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6)),
                                    "psnr_bg_vs_gt": psnr(x, gt, ~r3)})
                    rows.append(row)
                for d1, d2 in zip(depths[:-1], depths[1:]):
                    delta = np.abs(outs[(pi, d2)].astype(np.float32) - outs[(pi, d1)].astype(np.float32)).mean(-1)
                    din, dout = float(delta[region].mean()) if region.any() else float("nan"), float(delta[~region].mean())
                    rows.append({"case": rec["id"], "class": rec.get("class"), "k": k, "prompt_idx": pi, "prompt": prompt,
                                 "transition": f"N{d1}->N{d2}", "area_in": area_in, "delta_in": din, "delta_out": dout,
                                 "concentration": din / (dout + 1e-6), "mass_in": float(delta[region].sum() / (delta.sum() + 1e-6))})
            print(f"[{rec['id']} k{k}] done", flush=True)
    (out / "analysis.json").write_text(json.dumps(rows, indent=1))
    make_strips(out, man)
    print("wrote", out / "analysis.json")


def make_strips(out: Path, man: dict) -> None:
    """Per case / k / prompt: GT | N1 | N2 | ... (alternative prompts: context-less, no GT column), 256x192 cells."""
    sdir = out / "strips"
    sdir.mkdir(exist_ok=True)
    for rec in man["cases"]:
        for k in man["k"]:
            cdir = out / "clips" / rec["id"] / f"k{k}"
            for pi in range(len(rec["prompts"])):
                srcs = ([cdir / "gt.mp4"] if pi == 0 else []) + [cdir / f"p{pi}_N{d}.mp4" for d in man["depths"]]
                args, filt = [], ""
                for j, s in enumerate(srcs):
                    args += ["-i", str(s)]
                    filt += f"[{j}:v]scale=256:192[c{j}];"
                filt += "".join(f"[c{j}]" for j in range(len(srcs))) + f"hstack=inputs={len(srcs)}[v]"
                subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", filt, "-map", "[v]", "-r", "15",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
                                str(sdir / f"{rec['id']}_k{k}_p{pi}.mp4")], check=True)


def _group(rows: list[dict], keys: tuple[str, ...]) -> list[tuple[tuple, list[dict]]]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0]))


def _mean(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return f"{np.mean(vals):.3f}" if vals else "–"


def summarize(folder: Path, md: Path | None) -> None:
    rows: list[dict] = []
    for f in sorted(folder.glob("*/analysis.json")):
        for r in json.loads(f.read_text()):
            rows.append({**r, "model": f.parent.name})
    if not rows:
        raise SystemExit(f"no */analysis.json under {folder}")
    (folder / "summary.json").write_text(json.dumps(rows))
    if not md:
        return
    real = [r for r in rows if r["prompt_idx"] == 0 and "depth" in r]
    trans = [r for r in rows if "transition" in r and r["prompt_idx"] == 0]
    alt = [r for r in rows if r["prompt_idx"] > 0 and "depth" in r]
    L = ["# Loop depth: background vs motion", "",
         "Same noise at every depth. Moving region = pixels that move in the real clip (real prompt) or in the deepest output "
         "(alternative prompts). PSNR bg / moving: pixels outside / inside that region. Flow metrics as in MOTION.md. "
         "`motion` = mean compensated flow magnitude of the output inside the region (px/frame).", "",
         "## Per depth, real prompt (mean over cases)", "",
         "| model | k | N | n | PSNR bg | PSNR moving | LPIPS | dir cos | mag ratio | region cos | motion px/fr |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    cols = ["psnr_bg", "psnr_mov", "lpips", "flow_dir_cos", "flow_mag_ratio", "flow_region_cos", "motion"]
    for (m, k, d), g in _group(real, ("model", "k", "depth")):
        L.append(f"| {m} | {k} | {d} | {len(g)} | " + " | ".join(_mean(g, c) for c in cols) + " |")
    L += ["", f"static baseline (copy the first kept frame): PSNR bg {_mean(real, 'static_psnr_bg')}, PSNR moving {_mean(real, 'static_psnr_mov')}", "",
          "## Change between consecutive depths, real prompt", "",
          "| model | k | transition | n | area in | delta in | delta out | concentration | mass in |", "|---|--:|---|--:|--:|--:|--:|--:|--:|"]
    for (m, k, t), g in _group(trans, ("model", "k", "transition")):
        L.append(f"| {m} | {k} | {t} | {len(g)} | {_mean(g, 'area_in')} | {_mean(g, 'delta_in')} | {_mean(g, 'delta_out')} | "
                 f"{_mean(g, 'concentration')} | {_mean(g, 'mass_in')} |")
    L += ["", "concentration > 1: the extra loops change the moving region more (per pixel) than the background; "
          "mass in > area in: the moving region receives a larger share of the change than its area share.", ""]
    if alt:
        L += ["## Alternative prompts, same context frames and noise", "",
              "| model | k | N | n | motion (real prompt) | motion (alt prompt) | LPIPS alt vs real | dir cos alt vs real |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for (m, k, d), g in _group(alt, ("model", "k", "depth")):
            r = [x for x in real if x["model"] == m and x["k"] == k and x["depth"] == d]
            L.append(f"| {m} | {k} | {d} | {len(g)} | {_mean(r, 'motion')} | {_mean(g, 'motion')} | {_mean(g, 'lpips_vs_real_prompt')} | "
                     f"{_mean(g, 'dir_cos_vs_real_prompt')} |")
        L += ["", "dir cos alt vs real near +1: the output ignores the prompt (same motion as with the real prompt); "
              "near 0 or negative: the prompt changed the motion.", ""]
    if any(r.get("class") for r in real):
        L += ["## By case class, real prompt", "", "| model | k | class | N | n | dir cos | mag ratio | PSNR moving |", "|---|--:|---|--:|--:|--:|--:|--:|"]
        for (m, k, c, d), g in _group(real, ("model", "k", "class", "depth")):
            L.append(f"| {m} | {k} | {c} | {d} | {len(g)} | {_mean(g, 'flow_dir_cos')} | {_mean(g, 'flow_mag_ratio')} | {_mean(g, 'psnr_mov')} |")
    md.write_text("\n".join(L))
    print("wrote", md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--ema")
    p.add_argument("--cases", help="json with {cases: [{id, split, episode, anchor, caption?, alt_prompts?, class?}]}")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--k", type=int, nargs="+", default=[1, 9])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--text-encoder", default=None, help="Qwen3-VL directory or Hub id; required when alt_prompts are used")
    p.add_argument("--mirror", default=None, help="local snapshot of Jinyan0924/agibot-flux3-eval-splits (videos/, fixed/)")
    p.add_argument("--out")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--analyze", help="output dir of a generation run")
    p.add_argument("--small", action="store_true", help="raft_small (CPU analysis)")
    p.add_argument("--summarize", help="folder containing <model>/analysis.json")
    p.add_argument("--md", default=None)
    a = p.parse_args()
    if a.summarize:
        summarize(Path(a.summarize), Path(a.md) if a.md else None)
    elif a.analyze:
        analyze(a)
    else:
        if not (a.config and a.cases and a.out):
            raise SystemExit("generation needs --config, --cases, --out (and --ema for a fine-tuned model)")
        generate(a)


if __name__ == "__main__":
    main()
