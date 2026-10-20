#!/usr/bin/env python3
"""Landing page of the GitHub Pages site: the evaluated models, headline numbers and how to run the two evaluations.
    scripts/build_index_page.py --out docs/index.html      (reads configs/models.json and results/)
"""
from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPO = "https://github.com/naomili0924/looped-dit-video"
DEPTHS = (1, 2, 4, 8)


def load(name):
    p = RES / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def has_rollout(tag):
    return len(glob.glob(str(RES / "rollout" / tag / "clips" / "*" / "*.mp4"))) >= 120


def esc(s):
    return html.escape(str(s))


def build() -> str:
    reg = json.loads((ROOT / "configs" / "models.json").read_text())
    models = reg["models"]
    ft = load("ft_step6000")
    base = load("base")

    def row_model(m):
        normal = all(load(f"{m['tag']}_loops{n}") for n in DEPTHS)
        ev = []
        ev.append('<a href="loop-depth/">normal eval</a>' if normal else '<span class="miss">normal eval: not run</span>')
        ev.append('<a href="long-rollout/">long rollout</a>' if has_rollout(m["tag"]) else '<span class="miss">long rollout: not run</span>')
        ema = m["ema"].replace("hf://", "")
        return (f"<tr><td><b>{esc(m['label'])}</b><div class=sub>{esc(m['note'])}</div></td><td><code>{esc(m['tag'])}</code></td>"
                f"<td>{m['step']}</td><td>{', '.join(map(str, m['trained_depths']))}</td>"
                f"<td><code>{esc(ema)}</code><div class=sub><code>{esc(m['config'])}</code></div></td><td>{' · '.join(ev)}</td></tr>")

    def num(s, key, nd):
        return "–" if s is None or s.get(key) is None else f"{s[key]:.{nd}f}"

    def row_res(label, s, cls=""):
        k1, k9 = (s or {}).get("test_k1", {}), (s or {}).get("test_k9", {})
        j = k1.get("judge", {}).get("adherence")
        return (f"<tr{cls}><td>{label}</td><td>{num(k1, 'psnr', 2)}</td><td>{num(k1, 'ssim', 3)}</td><td>{num(k1, 'lpips', 3)}</td>"
                f"<td>{num(k1, 'fvd', 1)}</td><td>{'–' if j is None else f'{j:.2f}'}</td><td>{num(k9, 'psnr', 2)}</td><td>{num(k9, 'lpips', 3)}</td></tr>")

    res_rows = [row_res("base FLUX 3 Action, zero-shot (N = 1)", base), row_res("fine-tuned 6000 steps, Phase 3 (N = 1)", ft, " class=ref")]
    for m in models:
        for n in DEPTHS:
            s = load(f"{m['tag']}_loops{n}")
            if s:
                res_rows.append(row_res(f"{esc(m['label'])}, N = {n}", s, " class=train" if n in m["trained_depths"] else ""))

    return f"""<title>Looped FLUX 3 Action on AgiBot</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; same tokens as the two evaluation pages (graphite neutrals, verdigris accent). */
:root {{
  --bg: #f4f5f2; --surface: #ffffff; --surface-2: #e9ebe6; --ink: #1c1f1e; --ink-2: #4d5553; --ink-3: #828b88;
  --line: #d6d9d3; --accent: #1f6f69; --accent-soft: #d9ebe8; --ref: #efe9e0;
  --display: "Newsreader", Georgia, "Times New Roman", serif; --body: "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark;
  --bg: #161918; --surface: #1e2220; --surface-2: #272c2a; --ink: #e8eae6; --ink-2: #b4bab6; --ink-3: #7f8884;
  --line: #353b38; --accent: #5fbfb4; --accent-soft: #1f3b38; --ref: #2b2823; }} }}
:root[data-theme="dark"] {{ color-scheme: dark;
  --bg: #161918; --surface: #1e2220; --surface-2: #272c2a; --ink: #e8eae6; --ink-2: #b4bab6; --ink-3: #7f8884;
  --line: #353b38; --accent: #5fbfb4; --accent-soft: #1f3b38; --ref: #2b2823; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 16px; line-height: 1.55;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding-block: 32px 72px; }}
h1 {{ font-family: var(--display); font-weight: 500; font-size: clamp(30px, 4.5vw, 44px); margin: 0 0 8px; line-height: 1.1; text-wrap: balance; }}
h2 {{ font-family: var(--display); font-weight: 500; font-size: 26px; margin: 44px 0 10px; text-wrap: balance; }}
p {{ margin: 0 0 12px; max-width: 72ch; }}
.eyebrow {{ font-family: var(--mono); font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 10px; }}
.lede {{ color: var(--ink-2); font-size: 18px; max-width: 76ch; }}
a {{ color: var(--accent); }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin: 20px 0 8px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 16px 18px; }}
.card h3 {{ font-family: var(--display); font-weight: 500; font-size: 22px; margin: 0 0 6px; }}
.card h3 a {{ text-decoration: none; }} .card h3 a:hover {{ text-decoration: underline; }}
.card .say {{ font-family: var(--mono); font-size: 12px; color: var(--ink-3); letter-spacing: .05em; text-transform: uppercase; margin-bottom: 6px; }}
.card p {{ font-size: 15px; margin: 0; }}
.tbl {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14.5px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ color: var(--ink-2); font-weight: 500; font-size: 12px; font-family: var(--mono); letter-spacing: .04em; white-space: nowrap; }}
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {{ text-align: left; }}
.models td {{ text-align: left; }} .models td:nth-child(3), .models td:nth-child(4) {{ text-align: right; }}
.sub {{ font-size: 13px; color: var(--ink-3); max-width: 46ch; }}
.miss {{ color: var(--ink-3); }}
tr.ref td {{ background: var(--ref); }} tr.train td {{ background: var(--accent-soft); }}
code {{ font-family: var(--mono); font-size: 13px; background: var(--surface-2); padding: 1px 5px; border-radius: 3px; }}
pre {{ font-family: var(--mono); font-size: 13px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; overflow-x: auto; line-height: 1.5; }}
.note {{ font-size: 14px; color: var(--ink-3); max-width: 84ch; }}
.foot {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 13.5px; color: var(--ink-3); max-width: 90ch; }}
</style>
<div class="wrap">
  <div class="eyebrow">AgiBot World Beta · FLUX 3 Action 7B · Looped-DiT retrofit · evaluation pages</div>
  <h1>Looped FLUX 3 Action on AgiBot</h1>
  <p class="lede">A FLUX 3 Action video model fine-tuned on AgiBot World Beta head-camera clips (k past frames + task caption → 33 future frames, 15 fps, 384×512), then retrofitted as a Looped-DiT whose 9 core blocks run N times per denoising step. Two evaluations, each a page with videos, cover every looped checkpoint we trained; a third page sweeps noise seeds over the random-depth long run. Code and notes: <a href="{REPO}">naomili0924/looped-dit-video</a>, folder <code>agibot-flux3/</code>.</p>

  <div class="cards">
    <div class="card"><div class="say">say “normal eval”</div><h3><a href="loop-depth/">Trained Loop Depth</a></h3>
      <p>Test split, 100 windows, k = 1 and 9, the model sampled at loop depth N = 1, 2, 4, 8: PSNR / SSIM / LPIPS, FVD (300 windows), a VLM judge, validation curves, the training recipe, and 8 fixed test windows as strips <i>GT | base | fine-tuned | N = 1 | 2 | 4 | 8</i>.</p></div>
    <div class="card"><div class="say">say “long-rollout eval”</div><h3><a href="long-rollout/">Long Rollout</a></h3>
      <p>Zero-shot: one real frame and one prompt, four 33-frame windows chained autoregressively to 8.8 s. 8 test cases × 3 prompts (the real caption and two alternatives) × loop depths 1, 2, 4, 8, 16 next to the ground truth, the base model and the Phase-3 fine-tune.</p></div>
    <div class="card"><div class="say">say “seed sweep”</div><h3><a href="seed-sweep/">Random-Depth Seed Sweep</a></h3>
      <p>The long rollout of the random-depth (N 1–8) long run, repeated over 5 noise seeds at every checkpoint (steps 3000–9500): 5 × 5 seed-by-depth grid videos per case and prompt, and per depth the distance to the ground truth, its spread over seeds, motion and the LPIPS diversity between seeds. What it learns and fails to learn: <a href="{REPO}/blob/main/agibot-flux3/results/seedsweep/README.md">findings</a>.</p></div>
  </div>

  <h2>Models</h2>
  <div class="tbl"><table class="models"><thead><tr><th>model</th><th>tag</th><th>step</th><th>trained N</th><th>weights (Hugging Face) · inference config</th><th>evaluations</th></tr></thead>
  <tbody>{''.join(row_model(m) for m in models)}</tbody></table></div>
  <p class="note">All looped models start from the Phase-3 fine-tune (<code>Jinyan0924/flux3-agibot-videoft</code>, <code>ema_step-6000.safetensors</code>), blocks split 12 / 9 / 12 with head-wise attention gates in the 9 core blocks. The EMA weights are what is evaluated; the loop depth is a sampling-time argument.</p>

  <h2>Headline numbers, test split</h2>
  <div class="tbl"><table><thead><tr><th>model</th><th>k1 PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>FVD ↓</th><th>judge ↑</th><th>k9 PSNR ↑</th><th>k9 LPIPS ↓</th></tr></thead>
  <tbody>{''.join(res_rows)}</tbody></table></div>
  <p class="note">k = 1: PSNR / SSIM / LPIPS on 100 windows, FVD on 300, judge adherence on 100 (yes-rate with the true caption minus yes-rate with a swapped caption). Shaded rows: the depth the model was trained at. Full tables per split in <a href="{REPO}/blob/main/agibot-flux3/results/RESULTS.md">results/RESULTS.md</a>; the <a href="loop-depth/">Trained Loop Depth</a> page has the per-depth and validation tables.</p>

  <h2>How to run the evaluations</h2>
  <p>Both run on the box that holds the repo (two GPUs, the Phase-3 EMA and the Qwen3-VL text encoder under <code>flux-action/outputs/weights</code>); the full guide with prerequisites is <a href="{REPO}/blob/main/agibot-flux3/EVAL.md">EVAL.md</a>. A checkpoint is named by a tag, its training step, its inference config and its EMA file (local or <code>hf://user/repo/path</code>).</p>
<pre># normal eval  (≈ 2.5 h on 2 H100s) → docs/loop-depth
HF_TOKEN=... scripts/normal_eval.sh &lt;tag&gt; &lt;step&gt; &lt;config&gt; &lt;ema&gt; ["label"] ["trained depths, e.g. 4"]

# long-rollout eval  (≈ 35 min on 2 H100s) → docs/long-rollout
HF_TOKEN=... scripts/run_long_rollout.sh &lt;tag&gt; &lt;config&gt; &lt;ema&gt;        # env: STEP, LABEL, TRAINED

# both end by rebuilding this site (scripts/build_pages.sh); publish it:
GH_TOKEN=... scripts/push_all.sh "agibot-flux3: eval of &lt;tag&gt;"</pre>
  <p class="note">Models are registered in <code>configs/models.json</code> (tag, step, label, trained depths, config, weights); the wrappers add new tags automatically. Example for the random-depth model: <code>scripts/normal_eval.sh rand2500 2500 configs/looped_ft_384_random_n1-8.json hf://Jinyan0924/flux3-agibot-looped/random_depth_n1-8/ema_step-2500.safetensors "random depth N 1–8, step 2500" 1,2,3,4,5,6,7,8</code>.</p>

  <div class="foot">Frames are from <a href="https://agibot-world.com/">AgiBot World Beta</a> (CC BY-NC-SA 4.0: attribution, non-commercial, share-alike). Weights: <code>Jinyan0924/flux3-agibot-looped</code>, <code>Jinyan0924/flux3-agibot-videoft</code> (Hugging Face). Sampling: Euler 30 steps, CFG 4, shift 5, seed 0.</div>
</div>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(build())
    print(f"wrote {a.out} ({Path(a.out).stat().st_size} bytes)")
