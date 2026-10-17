#!/usr/bin/env python3
"""Build the "Trained Loop Depth" evaluation page (private artifact: AgiBot frames) for one or more trained
looped models, from the eval outputs in results/.

    scripts/build_loop_gallery.py --tags loop2500 [loop5000] --out <page.html>

Per tag it needs results/<tag>_loops{1,2,4,8}/summary.json and the strip videos results/loop_videos_<tag>/
(scripts/make_looped_strips.sh), published as videos_<tag>/k{1,9}_win{0..7}.mp4 next to the page. The validation
curve is read from the run's metrics.jsonl when the run directory still exists and cached as
results/<tag>_val_curve.json. Reference rows: results/base, results/ft_step6000, results/ft_step6000_loops{2,4,8}.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RUNS = {"loop2500": "/dev/shm/runs/loop384", "loop4500": "/dev/shm/runs/loop384_r2", "loop5000": "/dev/shm/runs/loop384_r2", "ft_step6000": "/dev/shm/runs/ft384"}
LABEL = {"loop2500": "looped, 2500 steps", "loop4500": "looped, 4500 steps", "loop5000": "looped, 5000 steps"}
try:  # labels of every registered model (configs/models.json)
    for _m in json.loads((ROOT / "configs" / "models.json").read_text())["models"]:
        LABEL.setdefault(_m["tag"], _m["label"])
except (OSError, KeyError, ValueError):
    pass
DEPTHS = (1, 2, 4, 8)
BLOCK_PASSES = {n: 12 + 9 * n + 12 for n in (1, 2, 4, 8, 10)}
WINDOWS = [  # the first 8 windows of the Phase-4 test evaluation set (seed 0), captions as recorded in AgiBot World Beta
    ("Wipe the toilet with the held cloth.", "task 541 · episode 779846"),
    ("Pick up the baseball cap on the sofa.", "task 527 · episode 769126"),
    ("Place the large bucket of potato chips held in the left arm into the gray felt bag on the left side of the cash register.", "task 492 · episode 730816"),
    ("Use pink vacuum cleaner to vacuum the garbage on the floor with right arm.", "task 554 · episode 920026"),
    ("Turn the doorknob to open the door with left arm.", "task 774 · episode 902908"),
    ("Pick up the uncapped whiteboard marker on the table with left arm.", "task 613 · episode 839320"),
    ("Slide the dark shorts on the desktop upward with the left arm.", "task 735 · episode 908086"),
    ("Pick up the purple taro chips from the restocking box with the right arm.", "task 566 · episode 791357"),
]


def load(name: str) -> dict | None:
    p = RES / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def val_curve(tag: str) -> list[dict]:
    cache = RES / f"{tag}_val_curve.json"
    run = Path(RUNS.get(tag, ""))
    if (run / "metrics.jsonl").exists():
        rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if '"eval"' in l]
        rows = [r for r in rows if r.get("event") == "eval"]
        if rows:
            cache.write_text(json.dumps(rows, indent=1))
    return json.loads(cache.read_text()) if cache.exists() else []


def f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def cell(m: dict | None, key: str, nd: int, best: float | None = None, lower=True):
    if m is None or m.get(key) is None:
        return "<td>–</td>"
    v = m[key]
    cls = ' class="best"' if best is not None and abs(v - best) < 1e-12 else ""
    return f"<td{cls}>{v:.{nd}f}</td>"


def judge(m):
    j = (m or {}).get("judge")
    return None if not j else j["adherence"]


def per_window(m):
    return None if not m or not m.get("gen_seconds") else m["gen_seconds"] / m["n_gen"]


def test_table(tags: list[str], base, ft, loops: dict[str, dict[int, dict | None]]) -> str:
    rows = []  # (label, model-class, k1 metrics, k9 metrics, N)
    rows.append(("base FLUX 3 Action, zero-shot", "", base, 1))
    rows.append(("fine-tuned 6000 steps (Phase 3)", "ref", ft, 1))
    for tag in tags:
        for n in DEPTHS:
            s = loops[tag].get(n)
            rows.append((f"{LABEL[tag]}, N = {n}", "train" if n == 4 else "", s, n))
    out = []
    for k in (1, 9):
        key = f"test_k{k}"
        ms = [(lab, cls, (s or {}).get(key), n) for lab, cls, s, n in rows]
        best = {}
        for metric, lower in (("psnr", False), ("ssim", False), ("lpips", True), ("fvd", True)):
            vals = [m[metric] for _, _, m, _ in ms if m and m.get(metric) is not None]
            if vals:
                best[metric] = min(vals) if lower else max(vals)
        jvals = [judge(m) for _, _, m, _ in ms if judge(m) is not None]
        jbest = max(jvals) if jvals else None
        head = "<th>model</th><th>N</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th>"
        head += "<th>FVD ↓</th><th>judge ↑</th>" if k == 1 else ""
        head += "<th>s / window</th>"
        body = []
        for lab, cls, m, n in ms:
            tr = f'<tr class="{cls}">' if cls else "<tr>"
            tr += f"<td>{html.escape(lab)}</td><td>{n}</td>"
            tr += cell(m, "psnr", 2, best.get("psnr")) + cell(m, "ssim", 3, best.get("ssim")) + cell(m, "lpips", 3, best.get("lpips"))
            if k == 1:
                tr += cell(m, "fvd", 1, best.get("fvd"))
                jv = judge(m)
                tr += "<td>–</td>" if jv is None else f'<td{" class=best" if jbest is not None and abs(jv - jbest) < 1e-9 else ""}>{jv:.2f}</td>'
            pw = per_window(m)
            tr += f"<td>{f(pw, 1)}</td></tr>"
            body.append(tr)
        n_fvd = (ms[1][2] or {}).get("n_fvd")
        note = (f"k = {k}: PSNR / SSIM / LPIPS on 100 windows" + (f"; FVD on {n_fvd} windows; judge on 100 windows (adherence = yes-rate with the true caption minus yes-rate with a swapped caption, Qwen3-VL judge)." if k == 1 else "."))
        out.append(f'<h3>k = {k} context frame{"s" if k > 1 else ""}</h3><div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div><p class="note">{note}</p>')
    return "".join(out)


def zeroshot_table(tags, ft, probe: dict[int, dict | None], loops) -> str:
    head = "<tr><th>N</th><th>block passes</th>" + "".join(
        f"<th>k = {k}, {lab}</th>" for k in (1, 9) for lab in ["untrained loop"] + [LABEL[t] for t in tags]) + "</tr>"
    body = []
    for n in (1, 2, 4, 8):
        tr = f"<tr{' class=ref' if n == 1 else ''}><td>{n}</td><td>{BLOCK_PASSES[n]}</td>"
        for k in (1, 9):
            key = f"test_k{k}"
            src = ft if n == 1 else probe.get(n)
            tr += cell((src or {}).get(key), "lpips", 3)
            for t in tags:
                tr += cell((loops[t].get(n) or {}).get(key), "lpips", 3)
        body.append(tr + "</tr>")
    return (f'<div class="tbl"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'
            '<p class="note">LPIPS (lower is better). "Untrained loop" is the Phase-3 model wrapped as a Looped-DiT without gates and sampled at depth N with no looped training (first 50 of the same windows); N = 1 is the Phase-3 model itself. The trained columns are the Phase-5 models on 100 windows.</p>')


def val_table(tags) -> str:
    cols = [("val_loss_k0", "k0"), ("val_loss_k1", "k1"), ("val_loss_k5", "k5"), ("val_loss_k9", "k9"), ("val_loss_k17", "k17"),
            ("val_loss_k9_loops1", "k9, N=1"), ("val_loss_k9_loops2", "k9, N=2"), ("val_loss_k9_loops8", "k9, N=8")]
    head = "<tr><th>run</th><th>step</th>" + "".join(f"<th>{c}</th>" for _, c in cols) + "</tr>"
    body = []
    ft = [r for r in val_curve("ft_step6000") if r["step"] == 6000]
    for r in ft:
        body.append("<tr class=ref><td>Phase-3 fine-tune (N = 1)</td><td>6000</td>" + "".join(cell(r, k, 4) if k in r else "<td>–</td>" for k, _ in cols) + "</tr>")
    for tag in tags:
        rows = val_curve(tag)
        seen = set()
        for r in rows:
            if r["step"] in seen or (tag in ("loop4500", "loop5000") and r["step"] <= 2000):
                continue
            seen.add(r["step"])
            body.append(f"<tr{' class=train' if r['step'] in (2500, 4500, 5000) else ''}><td>{LABEL[tag]}</td><td>{r['step']}</td>" + "".join(cell(r, k, 4) for k, _ in cols) + "</tr>")
    return (f'<div class="tbl"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'
            '<p class="note">Velocity-prediction MSE of the EMA weights on 32 fixed validation windows at 8 timesteps, per context length k; the looped runs are evaluated at the training depth N = 4 unless the column says otherwise. Lower is better. The 4500-step run shares steps 0–2000 with the 2500-step run and continues from that checkpoint (the step-4500 point itself is missing: the trainer was killed before that eval).</p>')


def strips_js(tags, urls=None) -> str:
    caps = json.dumps([[c, m] for c, m in WINDOWS])
    return f"""
(function () {{
  var caps = {caps};
  var URLS = {json.dumps(urls or {})};
  var tags = {json.dumps(tags)}, labels = {json.dumps([LABEL[t] for t in tags])};
  var tag = tags[tags.length - 1], k = 1, w = 0, FPS = 15, NF = 34, rate = 0.25;
  try {{ var saved = +localStorage.getItem('strip_rate'); if (saved === 0.25 || saved === 0.5 || saved === 1) {{ rate = saved; }} }} catch (e) {{}}
  var v = document.getElementById('v'), cap = document.getElementById('cap'), meta = document.getElementById('meta');
  var scrub = document.getElementById('scrub'), tl = document.getElementById('tlabel'), play = document.getElementById('play');
  scrub.max = NF - 1;
  function mk(id, vals, cur, fmt, on) {{
    var box = document.getElementById(id); if (!box) return; box.innerHTML = '';
    vals.forEach(function (x) {{
      var b = document.createElement('button'); b.className = 'opt'; b.type = 'button'; b.id = id + '_' + String(x).replace(/\\W/g, ''); b.textContent = fmt(x);
      b.setAttribute('aria-pressed', x === cur ? 'true' : 'false');
      b.addEventListener('click', function () {{ on(x); }});
      box.appendChild(b);
    }});
  }}
  function render() {{
    mk('mbtns', tags, tag, function (x) {{ return labels[tags.indexOf(x)]; }}, function (x) {{ tag = x; load(); }});
    mk('kbtns', [1, 9], k, function (x) {{ return 'k = ' + x; }}, function (x) {{ k = x; load(); }});
    mk('wbtns', [0, 1, 2, 3, 4, 5, 6, 7], w, function (x) {{ return String(x + 1); }}, function (x) {{ w = x; load(); }});
    mk('sbtns', [0.25, 0.5, 1], rate, function (x) {{ return x === 1 ? 'real time (15 fps)' : (x === 0.5 ? '½ speed' : '¼ speed'); }}, function (x) {{ rate = x; v.playbackRate = rate; try {{ localStorage.setItem('strip_rate', String(rate)); }} catch (e) {{}} render(); }});
    cap.textContent = caps[w][0]; meta.textContent = caps[w][1] + ' · k = ' + k;
  }}
  function load() {{
    render();
    var name = tag + '_k' + k + '_win' + w + '.mp4';
    v.src = URLS[name] || ('videos_' + tag + '/k' + k + '_win' + w + '.mp4');
    v.load();
    v.playbackRate = rate;
    v.play().catch(function () {{}});
    play.textContent = 'pause'; play.setAttribute('aria-pressed', 'true');
  }}
  v.addEventListener('loadedmetadata', function () {{ v.playbackRate = rate; }});
  v.addEventListener('timeupdate', function () {{
    var fr = Math.min(NF - 1, Math.round(v.currentTime * FPS));
    scrub.value = fr; tl.textContent = 'frame ' + fr + ' / ' + (NF - 1);
  }});
  scrub.addEventListener('input', function () {{
    v.pause(); play.textContent = 'play'; play.setAttribute('aria-pressed', 'false');
    v.currentTime = (+scrub.value + 0.5) / FPS; tl.textContent = 'frame ' + scrub.value + ' / ' + (NF - 1);
  }});
  play.addEventListener('click', function () {{
    if (v.paused) {{ v.play(); play.textContent = 'pause'; play.setAttribute('aria-pressed', 'true'); }}
    else {{ v.pause(); play.textContent = 'play'; play.setAttribute('aria-pressed', 'false'); }}
  }});
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{ v.autoplay = false; }}
  load();
}})();
"""


CSS = """
/* Layout: one reading column (max 1180px); the comparison strip is the hero, selectors above it, measurements and
   method below. Palette: cool graphite neutrals with a slight green cast, one verdigris accent for the selected state
   and the training-depth rows; Phase-3 reference rows in a flat warm grey. Type: Newsreader display, Source Sans 3 body,
   JetBrains Mono for labels and numbers. */
:root {
  --bg: #f4f5f2; --surface: #ffffff; --surface-2: #e9ebe6; --ink: #1c1f1e; --ink-2: #4d5553; --ink-3: #828b88;
  --line: #d6d9d3; --accent: #1f6f69; --accent-soft: #d9ebe8; --accent-ink: #ffffff; --ref: #efe9e0; --best: #0f5a55;
  --display: "Newsreader", Georgia, "Times New Roman", serif; --body: "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --bg: #161918; --surface: #1e2220; --surface-2: #272c2a; --ink: #e8eae6; --ink-2: #b4bab6; --ink-3: #7f8884;
  --line: #353b38; --accent: #5fbfb4; --accent-soft: #1f3b38; --accent-ink: #10201e; --ref: #2b2823; --best: #8adbd1; } }
:root[data-theme="dark"] { color-scheme: dark;
  --bg: #161918; --surface: #1e2220; --surface-2: #272c2a; --ink: #e8eae6; --ink-2: #b4bab6; --ink-3: #7f8884;
  --line: #353b38; --accent: #5fbfb4; --accent-soft: #1f3b38; --accent-ink: #10201e; --ref: #2b2823; --best: #8adbd1; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 16px; line-height: 1.55; }
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 32px 72px; padding-inline: 20px; }
h1 { font-family: var(--display); font-weight: 500; font-size: clamp(30px, 4.5vw, 44px); margin: 0 0 8px; line-height: 1.1; text-wrap: balance; letter-spacing: -.01em; }
h2 { font-family: var(--display); font-weight: 500; font-size: 26px; margin: 48px 0 10px; text-wrap: balance; }
h3 { font-family: var(--mono); font-weight: 500; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-2); margin: 22px 0 8px; }
p { margin: 0 0 12px; max-width: 72ch; }
.eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 10px; }
.lede { color: var(--ink-2); font-size: 18px; max-width: 76ch; }
.sel { display: flex; gap: 22px; flex-wrap: wrap; align-items: flex-start; margin: 24px 0 12px; }
.group { display: grid; gap: 6px; }
.group .lbl { font-family: var(--mono); font-size: 12px; color: var(--ink-3); letter-spacing: .06em; text-transform: uppercase; }
.btns { display: flex; gap: 6px; flex-wrap: wrap; }
button.opt { font: inherit; font-family: var(--mono); font-size: 13px; padding: 6px 11px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 4px; cursor: pointer; }
button.opt[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
button.opt:focus-visible, .ctl input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.caption { font-size: 16px; margin: 4px 0 10px; }
.caption .mono { font-family: var(--mono); font-size: 12px; color: var(--ink-3); margin-left: 8px; }
.strip { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px; }
.cols { display: grid; grid-template-columns: repeat(7, 1fr); gap: 0; margin-bottom: 6px; }
.cols div { font-family: var(--mono); font-size: 12px; text-align: center; color: var(--ink-2); padding: 2px 0; border-bottom: 2px solid var(--line); }
.cols div.ref { border-bottom-color: var(--ink-3); color: var(--ink); }
.cols div.train { border-bottom-color: var(--accent); color: var(--ink); }
video { width: 100%; display: block; border-radius: 3px; background: #000; }
.ctl { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 8px; font-size: 14px; color: var(--ink-2); }
.ctl input[type=range] { width: min(100%, 360px); accent-color: var(--accent); }
.ctl .t { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 9ch; font-size: 13px; }
.note { font-size: 14px; color: var(--ink-3); max-width: 84ch; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 500; font-size: 12px; font-family: var(--mono); letter-spacing: .04em; }
td:first-child, th:first-child { text-align: left; }
td:nth-child(2), th:nth-child(2) { text-align: right; }
tr.ref td { background: var(--ref); }
tr.train td { background: var(--accent-soft); }
td.best { color: var(--best); font-weight: 600; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; align-items: start; }
@media (max-width: 820px) { .two { grid-template-columns: 1fr; } .cols div { font-size: 10px; } .lede { font-size: 16px; } }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; font-size: 15px; margin: 0; }
.kv dt { font-family: var(--mono); font-size: 12.5px; color: var(--ink-3); padding-top: 3px; } .kv dd { margin: 0; max-width: 62ch; }
.stages { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 14px 0 10px; }
.stage { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; font-size: 14px; }
.stage.core { border-color: var(--accent); background: var(--accent-soft); }
.stage .n { font-family: var(--mono); font-size: 12px; color: var(--ink-3); letter-spacing: .05em; text-transform: uppercase; }
.stage .big { font-family: var(--display); font-size: 24px; line-height: 1.1; margin: 4px 0 2px; }
@media (max-width: 820px) { .stages { grid-template-columns: 1fr 1fr; } }
ol.steps { padding-left: 1.3em; margin: 0 0 12px; max-width: 72ch; } ol.steps li { margin-bottom: 8px; }
code { font-family: var(--mono); font-size: 13px; background: var(--surface-2); padding: 1px 5px; border-radius: 3px; }
.foot { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 13.5px; color: var(--ink-3); max-width: 90ch; }
"""


def build(tags: list[str], allow_missing: bool = False, urls: dict[str, str] | None = None) -> str:
    base, ft = load("base"), load("ft_step6000")
    probe = {n: load(f"ft_step6000_loops{n}") for n in (2, 4, 8, 10)}
    loops = {t: {n: load(f"{t}_loops{n}") for n in DEPTHS} for t in tags}
    for t in tags:
        missing = [n for n in DEPTHS if loops[t][n] is None]
        if missing and not allow_missing:
            raise SystemExit(f"{t}: missing results for N = {missing} (pass --allow-missing for a draft)")
        for n in missing:
            loops[t][n] = {"test_k1": {}, "test_k9": {}}
    latest = tags[-1]
    s4 = loops[latest][4]["test_k1"]
    f4 = ft["test_k1"]
    model_sel = ('<div class="group"><span class="lbl">model</span><div class="btns" id="mbtns"></div></div>' if len(tags) > 1 else "")
    steps_word = " and ".join(t.replace("loop", "") for t in tags)

    page = f"""<title>Trained Loop Depth</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style>

<div class="wrap">
  <div class="eyebrow">AgiBot World Beta · test split · Looped-DiT retrofit of the fine-tuned FLUX 3 Action · Phase 5</div>
  <h1>Trained Loop Depth</h1>
  <p class="lede">The Phase-3 fine-tune, retrofitted as a Looped-DiT (12 pre-loop blocks, 9 looped core blocks, 12 post-loop blocks, head-wise attention gates) and trained for {steps_word} steps at loop depth N = 4 with deep supervision. The same eight test windows, captions and noise seed as the <a href="https://claude.ai/artifact/YEtv4D6nkgQ2FSx4BK4rwh">zero-shot loop-depth gallery</a>: only the model and the number of core passes change across the columns. 33 future frames at 15 fps, 384×512, Euler 30 steps, CFG 4.</p>

  <div class="sel">
    {model_sel}
    <div class="group"><span class="lbl">context frames k</span><div class="btns" id="kbtns"></div></div>
    <div class="group"><span class="lbl">test window</span><div class="btns" id="wbtns"></div></div>
    <div class="group"><span class="lbl">playback speed</span><div class="btns" id="sbtns"></div></div>
  </div>
  <div class="caption"><span id="cap"></span><span class="mono" id="meta"></span></div>

  <div class="strip">
    <div class="cols">
      <div>ground truth</div><div>base, N = 1</div><div class="ref">fine-tuned, N = 1</div><div>looped, N = 1</div><div>N = 2</div><div class="train">N = 4</div><div>N = 8</div>
    </div>
    <video id="v" muted loop playsinline preload="auto"></video>
    <div class="ctl">
      <button class="opt" id="play" aria-pressed="true">pause</button>
      <input type="range" id="scrub" min="0" max="33" value="0" step="1" aria-label="frame">
      <span class="t" id="tlabel">frame 0 / 33</span>
      <span class="note">drag to step through frames; the loop restarts automatically</span>
    </div>
  </div>
  <p class="note" style="margin-top:8px">Columns from left: ground-truth future, base FLUX 3 Action zero-shot, the Phase-3 fine-tune (grey underline, the previous reference), then the looped model sampled with the core run 1, 2, 4 and 8 times. N = 4 (green underline) is the depth it was trained at. Cells are downscaled to 256×192 for the page.</p>

  <h2>Test split, the same 100 windows as Phase 4</h2>
  <p>At the training depth the looped model reaches LPIPS {s4['lpips']:.3f} and FVD {s4['fvd']:.0f} at k = 1, against {f4['lpips']:.3f} and {f4['fvd']:.0f} for the Phase-3 fine-tune it started from, after {'half an epoch' if latest == 'loop2500' else '0.9 epoch'} of looped training. Green rows are the training depth; the best value in each column is marked. The last column is the measured sampling time per 33-frame window (batch 4; one A100 for the 2500 model, one H100 for the 4500 model; 30 Euler steps, two forward passes per step for guidance).</p>
  {test_table(tags, base, ft, loops)}

  <h2>Before and after training the loop</h2>
  <p>The untrained loop (Phase 4's zero-shot probe) degraded quickly with depth because the core blocks had only ever run once. After Phase 5 every depth is a usable exit.</p>
  {zeroshot_table(tags, ft, probe, loops)}

  <h2>Validation during training</h2>
  {val_table(tags)}

  <h2>How the model was fine-tuned</h2>
  <div class="two">
    <div>
      <h3>Phase 3 · full fine-tune of the backbone</h3>
      <dl class="kv">
        <dt>model</dt><dd>FLUX 3 Action 7B transformer, video-only configuration: the video, video-condition and text streams (5.57 B parameters) are trained in full, no LoRA. The robot action and state streams are dropped. The video VAE and the Qwen3-VL text encoder stay frozen.</dd>
        <dt>data</dt><dd>AgiBot World Beta head-camera clips, 15 fps, resized to 384×512. Train: 7960 episodes over 212 tasks, 159,983 windows; a window is k past frames plus a caption (the segment's action text) and the 33 frames that follow.</dd>
        <dt>input</dt><dd>The past frames are VAE-encoded and enter the video-condition stream as clean tokens at timestep 0; the future frames are encoded, noised and enter the video stream (9 latent frames × 12 × 16 patches = 1728 tokens). Context length k cycles through 1, 5 and 9 per micro-batch, with k = 0 (caption only) 10% of the time. Captions are dropped to the empty string 10% of the time so guidance works at inference.</dd>
        <dt>objective</dt><dd>Rectified-flow velocity MSE on the future tokens, timesteps drawn as sigmoid(N(1.08, 1.0)) as in the base model's recipe.</dd>
        <dt>optimizer</dt><dd>AdamW, learning rate 1e-5, betas (0.9, 0.999), no weight decay, gradient clip 1.0, fp32 master weights with bf16 compute, EMA 0.999 updated every 10 steps (the EMA is what is evaluated). 32 windows per step: 2 GPUs × 4 windows × 4 accumulation steps.</dd>
        <dt>schedule</dt><dd>6000 steps (1.2 epochs): 500-step warmup, constant, linear cooldown to zero over steps 5000–6000.</dd>
        <dt>infrastructure</dt><dd>2 × A100 80 GB, FSDP2 sharding with activation checkpointing and compiled blocks, 16 s per step. Steps 0–3000 ran with a reduce-order bug that mixed gradients across ranks; it was fixed and steps 3000–6000 are clean.</dd>
      </dl>
    </div>
    <div>
      <h3>Phase 5 · Looped-DiT retrofit</h3>
      <p>The 33 transformer blocks of the fine-tuned model (5 dual-stream mode blocks, 28 joint single blocks) are split into three stages. The retrofit adds no blocks: the core is 9 of the existing single blocks, run N times with shared weights.</p>
      <div class="stages">
        <div class="stage"><div class="n">A · pre-loop, once</div><div class="big">12</div>5 mode blocks + 7 single blocks</div>
        <div class="stage core"><div class="n">B · looped core, × N</div><div class="big">9</div>single blocks, shared weights, gated attention</div>
        <div class="stage"><div class="n">C · post-loop, once per exit</div><div class="big">12</div>single blocks + output head</div>
        <div class="stage"><div class="n">exits</div><div class="big">N</div>every loop state decodes through C</div>
      </div>
      <dl class="kv">
        <dt>initialisation</dt><dd>All trunk weights are the Phase-3 EMA at step 6000. The only new parameters are head-wise sigmoid attention gates in the 9 core blocks (one linear map for text tokens, one for video tokens, zero weights, bias +4), so at initialisation each gate passes 98% of the attention output and the model computes almost exactly what the fine-tune did at N = 1.</dd>
        <dt>loss</dt><dd>Deep supervision: after each pass through the core the state is decoded through C and scored with the same velocity loss. Weights 1/3, 1/3, 1/3, 1 over exits 1–4, rescaled to sum 2, so the final exit counts most but every depth is trained.</dd>
        <dt>schedule</dt><dd>Same data stream, seed, batch size, learning rate and optimizer as Phase 3; 200-step warmup. The 2500-step run cools down over steps 2000–2500 (half an epoch). The 4500-step run restarts from the step-2000 checkpoint and keeps the learning rate constant to step 4500 (0.9 epoch), where it was stopped without a cooldown; its EMA (decay 0.999) is the evaluated model.</dd>
        <dt>cost</dt><dd>12 + 4 × 9 + 4 × 12 = 96 block passes per training forward against 33 for the plain model; 31 s per step on the same two GPUs, so the 2500-step run roughly matches Phase 3's compute.</dd>
      </dl>
    </div>
  </div>

  <h2>How a video is generated</h2>
  <div class="two">
    <div>
      <ol class="steps">
        <li>The k past frames are encoded by the video VAE (4× in time, 32× in space, 96 channels) into clean conditioning tokens at timestep 0. With k = 0 there is no conditioning video.</li>
        <li>The caption is encoded by the frozen Qwen3-VL text encoder (80 text tokens); the empty caption is encoded too, as the unconditional branch.</li>
        <li>Gaussian noise is drawn for the 9 future latent frames from a fixed seed (seed 0 for every evaluation window), so the columns of a strip share their noise.</li>
        <li>30 Euler steps run on the time-shifted schedule (shift 5, t from 1 to 0). At each step the transformer predicts the velocity twice, with and without the caption, and the two are combined with classifier-free guidance scale 4.</li>
        <li>The loop depth is a call-time argument: the same weights are sampled with the core run 1, 2, 4 or 8 times; the pre- and post-loop stages and the head are unchanged.</li>
        <li>The final latents are decoded by the VAE to 33 frames at 384×512 and compared with the real continuation: PSNR, SSIM and LPIPS per window, FVD over 300 windows, and a VLM judge that asks whether the clip matches its caption, scored against a caption from another task.</li>
      </ol>
    </div>
    <div>
      <h3>Cost per depth</h3>
      <div class="tbl"><table>
        <thead><tr><th>N</th><th>block passes per forward</th><th>relative</th>{"".join(f"<th>s / window, {LABEL[t]}</th>" for t in tags)}</tr></thead>
        <tbody>{"".join(f"<tr{' class=train' if n == 4 else ''}><td>{n}</td><td>{BLOCK_PASSES[n]}</td><td>{BLOCK_PASSES[n] / BLOCK_PASSES[1]:.2f}×</td>" + "".join(f"<td>{f(per_window(loops[t][n]['test_k1']), 1)}</td>" for t in tags) + "</tr>" for n in DEPTHS)}</tbody>
      </table></div>
      <p class="note">Seconds per 33-frame window at k = 1, including the guidance pass (two forwards per step) and the VAE decode; batch 4, one A100 for the 2500 model and one H100 for the 4500 model, so the two columns are not comparable in absolute terms. The plain Phase-3 model takes {f(per_window(ft['test_k1']), 1)} s under the same protocol.</p>
    </div>
  </div>

  <div class="foot">
    Evaluation protocol as in Phase 4 (<code>ldvflux/eval_phase4.py</code>, seed 0, Euler 30, CFG 4, shift 5). Looped weights: EMA of the Phase-5 run, loaded strictly; depth set with <code>--loops N</code>. Frames are from <a href="https://agibot-world.com/">AgiBot World Beta</a> (CC BY-NC-SA 4.0: attribution, non-commercial, share-alike). Weights: <code>Jinyan0924/flux3-agibot-looped</code> and <code>Jinyan0924/flux3-agibot-videoft</code>; code: <code>naomili0924/looped-dit-video</code>, <code>agibot-flux3/</code>.
  </div>
</div>

<script>{strips_js(tags, urls)}</script>
"""
    return page


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["loop2500"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true", help="draft: render missing depths as dashes")
    ap.add_argument("--urls", default=None, help="json {<tag>_k<k>_win<w>.mp4: asset url} after uploading the strips as artifact assets")
    a = ap.parse_args()
    urls = json.loads(Path(a.urls).read_text()) if a.urls else {}
    Path(a.out).write_text(build(a.tags, a.allow_missing, urls))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size} bytes)")
