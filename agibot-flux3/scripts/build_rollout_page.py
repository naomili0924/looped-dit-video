#!/usr/bin/env python3
"""Compose the "Long Rollout" strips and page for one or more looped checkpoints (after scripts/run_long_rollout.sh).

    scripts/build_rollout_page.py --models loop4500=4500 [loop2500=2500 ...] [--out results/rollout/page/long_rollout.html]
                                  [--urls results/rollout/page/urls.json] [--box "2xH100 SXM"]

Per model <tag>=<step> it needs results/rollout/<tag>/clips/<case>/p<i>_N<d>.mp4 (depths 1 2 4 8 16) plus the shared
results/rollout/{gt,base,ft6000}. It writes one strip per model / case / prompt to results/rollout/page/videos/
<tag>_<case>_p<i>.mp4 (8 columns of 256x192: GT | base | ft6000 | N1 | N2 | N4 | N8 | N16) and the page HTML, with a
checkpoint selector that shows the training step of every looped model. Publishing: publish the page with the `assets`
capability, upload the strips as assets, write {filename: url} to --urls and build again so the page points at them.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLL = ROOT / "results" / "rollout"
DEPTHS = (1, 2, 4, 8, 16)
FPS, FUTURE = 15, 33
# gallery order of the Long Rollout page: worst -> best motion score of the Phase-3 model (EVAL_MOTION.md)
ORDER = ["test04_vacuum", "test01_toilet", "test05_doorknob", "test06_marker", "test02_cap", "test07_shorts", "test03_chips_bag", "test08_taro_chips"]
SHORT = {"test04_vacuum": "vacuum", "test01_toilet": "toilet", "test05_doorknob": "doorknob", "test06_marker": "marker", "test02_cap": "cap",
         "test07_shorts": "shorts", "test03_chips_bag": "chips bag", "test08_taro_chips": "taro chips"}


def compose(tag: str, case: str, pi: int, vdir: Path) -> Path:
    dst = vdir / f"{tag}_{case}_p{pi}.mp4"
    if dst.exists():
        return dst
    srcs = [ROLL / "gt" / f"{case}.mp4", ROLL / "base" / "clips" / case / f"p{pi}_N1.mp4", ROLL / "ft6000" / "clips" / case / f"p{pi}_N1.mp4"]
    srcs += [ROLL / tag / "clips" / case / f"p{pi}_N{d}.mp4" for d in DEPTHS]
    missing = [str(s) for s in srcs if not s.exists()]
    if missing:
        raise FileNotFoundError(missing)
    args, filt = [], ""
    for j, s in enumerate(srcs):
        args += ["-i", str(s)]
        filt += f"[{j}:v]scale=256:192[c{j}];"
    filt += "".join(f"[c{j}]" for j in range(len(srcs))) + f"hstack=inputs={len(srcs)}[v]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", filt, "-map", "[v]", "-r", str(FPS), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(dst)], check=True)
    return dst


def mean_time(folder: Path, depth: int | None) -> float | None:
    p = folder / "times.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text())
    vals = [v for k, v in t.items() if depth is None or k.endswith(f"_N{depth}")]
    return sum(vals) / len(vals) if vals else None


def build(models: list[tuple[str, int, str, list[int]]], out: Path, urls: dict[str, str], box: str, windows: int, allow_missing: bool = False) -> None:
    vdir = out.parent / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    gt_meta = json.loads((ROLL / "gt" / "meta.json").read_text())
    prompts = json.loads((ROLL / "base" / "prompts.json").read_text())
    cases = [c for c in ORDER if c in prompts] + [c for c in prompts if c not in ORDER]
    n = len(cases)
    case_rows = []
    for i, c in enumerate(cases):
        try:
            for tag, *_ in models:
                for pi in range(len(prompts[c])):
                    compose(tag, c, pi, vdir)
        except FileNotFoundError as e:
            if not allow_missing:
                raise
            print(f"[{c}] skipped (clips not finished): {e.args[0][0]}")
            continue
        case_rows.append({"id": c, "btn": f"#{n - i} {SHORT.get(c, c)}", "gt_s": gt_meta[c]["gt_s"], "gt_full": gt_meta[c]["gt_full"], "prompts": prompts[c]})
    times = {}
    for tag, *_ in models:
        times[tag] = {f"N{d}": mean_time(ROLL / tag, d) for d in DEPTHS}
    t_base, t_ft = mean_time(ROLL / "base", 1), mean_time(ROLL / "ft6000", 1)
    nf = 1 + windows * FUTURE

    def secs(x):
        return "–" if x is None else f"{x:.0f} s"

    time_note = (f"Generation time for {windows} windows on one GPU of the {html.escape(box)} box, while other jobs shared it: "
                 f"base {secs(t_base)}, fine-tuned {secs(t_ft)}; " + "; ".join(
                     f"{html.escape(label)}: " + ", ".join(f"N = {d} {secs(times[tag][f'N{d}'])}" for d in DEPTHS) for tag, step, label, _ in models) +
                 ". Depths marked (trained) in the column headers are the ones the selected checkpoint was trained at; the others are extrapolation.")
    model_sel = ('<div class="group"><span class="lbl">looped checkpoint (training step)</span><div class="btns" id="mbtns"></div></div>')
    models_js = [{"tag": t, "step": s, "label": l, "trained": tr} for t, s, l, tr in models]
    page = f"""<title>Long Rollout</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Viewing bench: selectors, a full-width 8-column strip, a timeline under it marking where each 2.2 s window starts. */
:root {{
  --bg: #f3f4f6; --surface: #ffffff; --ink: #16191d; --ink-2: #4a515a; --ink-3: #7c848e; --line: #d5d9de;
  --accent: #2f5d8a; --accent-ink: #ffffff; --train: #1f6f69;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark;
  --bg: #121417; --surface: #1b1e22; --ink: #e6e8eb; --ink-2: #b0b7bf; --ink-3: #7f8892; --line: #2f343a;
  --accent: #7fb0e0; --accent-ink: #0f1820; --train: #5fbfb4; }} }}
:root[data-theme="dark"] {{ color-scheme: dark;
  --bg: #121417; --surface: #1b1e22; --ink: #e6e8eb; --ink-2: #b0b7bf; --ink-3: #7f8892; --line: #2f343a;
  --accent: #7fb0e0; --accent-ink: #0f1820; --train: #5fbfb4; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }}
.wrap {{ max-width: 1320px; margin: 0 auto; padding-block: 26px 60px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 18px; }}
h1 {{ font-size: 24px; font-weight: 600; margin: 0; text-wrap: balance; }}
p {{ margin: 0; max-width: 88ch; }}
.eyebrow {{ font-family: var(--mono); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }}
.head {{ display: grid; gap: 6px; }}
.note {{ font-size: 13.5px; color: var(--ink-3); }}
.sel {{ display: flex; flex-wrap: wrap; gap: 14px 26px; align-items: flex-start; }}
.group {{ display: grid; gap: 6px; }}
.group .lbl {{ font-family: var(--mono); font-size: 11.5px; color: var(--ink-3); letter-spacing: .07em; text-transform: uppercase; }}
.btns {{ display: flex; flex-wrap: wrap; gap: 6px; }}
button.opt {{ font: 500 12.5px var(--mono); padding: 5px 10px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 4px; cursor: pointer; }}
button.opt[aria-pressed="true"] {{ background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }}
button.opt:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.prompt .text {{ font-size: 18px; max-width: 90ch; }}
.prompt .sub {{ font-family: var(--mono); font-size: 12px; color: var(--ink-3); }}
.strip {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px; overflow-x: auto; }}
.inner {{ min-width: 1080px; }}
.cols {{ display: grid; grid-template-columns: repeat(8, 1fr); margin-bottom: 4px; }}
.cols div {{ font-family: var(--mono); font-size: 11.5px; text-align: center; color: var(--ink-2); border-bottom: 2px solid var(--line); padding: 2px; }}
.cols div:first-child {{ color: var(--ink); border-bottom-color: var(--ink-3); }}
.cols div.train {{ color: var(--ink); border-bottom-color: var(--train); }}
.cols div b {{ font-weight: 500; color: var(--ink); }}
video {{ width: 100%; display: block; background: #000; border-radius: 2px; }}
.tl {{ position: relative; height: 34px; margin-top: 8px; cursor: pointer; }}
.tl .track {{ position: absolute; left: 0; right: 0; top: 6px; height: 6px; background: var(--line); border-radius: 3px; }}
.tl .fill {{ position: absolute; left: 0; top: 6px; height: 6px; background: var(--accent); border-radius: 3px; width: 0; }}
.tl .mk {{ position: absolute; top: 2px; width: 0; height: 14px; border-left: 1px solid var(--ink-3); }}
.tl .mk span {{ position: absolute; top: 15px; transform: translateX(-50%); font: 11px var(--mono); color: var(--ink-3); white-space: nowrap; }}
.tl .mk.big {{ border-left: 2px solid var(--ink); }}
.tl .mk.big span {{ color: var(--ink); font-weight: 500; }}
.readout {{ font-family: var(--mono); font-size: 13px; color: var(--ink-2); }}
.ctlrow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 8px; }}
button.play {{ font: 600 14px var(--body); padding: 7px 16px; min-width: 96px; border: 1px solid var(--accent); background: var(--accent); color: var(--accent-ink); border-radius: 4px; cursor: pointer; }}
button.play:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.readout b {{ color: var(--ink); font-weight: 500; }}
</style>

<div class="wrap">
  <div class="head">
    <div class="eyebrow">{html.escape(box)} · gallery set, 8 test windows · k = 1 · Euler 30, CFG 4, shift 5</div>
    <h1>Autoregressive rollout to {nf / FPS:.1f} s</h1>
    <p class="note">Each model makes {windows} windows of {FUTURE} frames in a row. Window 1 starts from the real last frame and is the gallery clip. Each later window starts from the last frame the model itself generated, with the same prompt and fresh noise. {nf} frames at {FPS} fps. The looped columns belong to the checkpoint selected below; its training step is in the column header.</p>
  </div>

  <div class="sel">
    {model_sel}
    <div class="group"><span class="lbl">case, worst → best motion (gallery order)</span><div class="btns" id="cbtns"></div></div>
    <div class="group"><span class="lbl">prompt</span><div class="btns" id="pbtns"></div></div>
    <div class="group"><span class="lbl">playback</span><div class="btns" id="sbtns"></div></div>
  </div>

  <div class="prompt"><div class="sub" id="psub"></div><div class="text" id="ptext"></div></div>

  <div class="strip"><div class="inner">
    <div class="cols"><div>ground truth</div><div>base (zero-shot)</div><div>fine-tuned 6000</div><div id="h1"></div><div id="h2"></div><div id="h4"></div><div id="h8"></div><div id="h16"></div></div>
    <video id="v" muted loop playsinline autoplay preload="auto"></video>
    <div class="tl" id="tl"><div class="track"></div><div class="fill" id="fill"></div></div>
    <div class="ctlrow"><button type="button" class="play" id="play">Pause</button><button type="button" class="opt" id="restart">Restart</button><div class="readout" id="ro"></div></div>
  </div></div>

  <p class="note" id="gtnote"></p>
  <p class="note">{time_note} Frames from AgiBot World Beta (CC BY-NC-SA 4.0); this page is private for that reason.</p>
</div>

<script>
var CASES = {json.dumps(case_rows)};
var MODELS = {json.dumps(models_js)};
var URLS = {json.dumps(urls)};
(function () {{
  var FPS = {FPS}, NF = {nf}, DUR = NF / FPS, starts = {json.dumps([1 + FUTURE * i for i in range(windows)])};
  var st = {{ m: MODELS.length - 1, c: 0, p: 0, r: 0.5 }};
  try {{ var s = +localStorage.getItem('roll_rate'); if (s === 0.25 || s === 0.5 || s === 1) st.r = s; }} catch (e) {{}}
  var v = document.getElementById('v'), tl = document.getElementById('tl'), fill = document.getElementById('fill'), ro = document.getElementById('ro');
  function mark(t, label, big) {{ var m = document.createElement('div'); m.className = 'mk' + (big ? ' big' : ''); m.style.left = (100 * t / DUR) + '%';
    var s = document.createElement('span'); s.textContent = label; m.appendChild(s); tl.appendChild(m); }}
  starts.slice(1).forEach(function (f, i) {{ mark(f / FPS, 'w' + (i + 2), false); }});
  mark(6, '6 s', true); mark(8, '8 s', true);
  function mk(id, vals, cur, fmt, on) {{
    var box = document.getElementById(id); if (!box) return; box.innerHTML = '';
    vals.forEach(function (x) {{ var b = document.createElement('button'); b.type = 'button'; b.className = 'opt'; b.textContent = fmt(x);
      b.setAttribute('aria-pressed', x === cur ? 'true' : 'false'); b.addEventListener('click', function () {{ on(x); }}); box.appendChild(b); }});
  }}
  function render(reload) {{
    var c = CASES[st.c], m = MODELS[st.m];
    mk('mbtns', MODELS.map(function (_, i) {{ return i; }}), st.m, function (i) {{ return MODELS[i].label; }}, function (i) {{ st.m = i; render(true); }});
    mk('cbtns', CASES.map(function (_, i) {{ return i; }}), st.c, function (i) {{ return CASES[i].btn; }}, function (i) {{ st.c = i; render(true); }});
    mk('sbtns', [0.25, 0.5, 1], st.r, function (x) {{ return x === 1 ? 'real time' : (x === 0.5 ? '½' : '¼'); }},
       function (x) {{ st.r = x; v.playbackRate = x; try {{ localStorage.setItem('roll_rate', String(x)); }} catch (e) {{}} render(false); }});
    mk('pbtns', [0, 1, 2], st.p, function (p) {{ return ['real', 'alt 1', 'alt 2'][p]; }}, function (p) {{ st.p = p; render(true); }});
    [1, 2, 4, 8, 16].forEach(function (d) {{
      var h = document.getElementById('h' + d), tr = m.trained.indexOf(d) >= 0;
      h.innerHTML = (d === 1 ? '<b>' + m.label.replace(/<[^>]*>/g, '') + '</b> · ' : '') + 'N = ' + d + (tr ? ' (trained)' : '');
      h.className = tr ? 'train' : '';
    }});
    document.getElementById('psub').textContent = st.p === 0 ? 'real prompt (AgiBot annotation), used for all ' + starts.length + ' windows'
      : 'alt ' + st.p + ' (cases file), used for all ' + starts.length + ' windows; no ground truth of its own, the GT column is the real continuation';
    document.getElementById('ptext').textContent = c.prompts[st.p];
    document.getElementById('gtnote').textContent = c.gt_full ? 'The ground truth covers the full ' + DUR.toFixed(1) + ' s.'
      : 'The recorded episode ends at ' + c.gt_s + ' s; after that the ground-truth column is black. The models keep going.';
    if (reload) {{ var name = m.tag + '_' + c.id + '_p' + st.p + '.mp4'; v.src = URLS[name] || ('videos/' + name); v.load(); v.playbackRate = st.r; v.play().catch(function () {{}}); }}
  }}
  function tick() {{
    var t = v.currentTime || 0, f = Math.min(NF - 1, Math.floor(t * FPS + 1e-6)), w = 0;
    for (var i = 0; i < starts.length; i++) if (f >= starts[i]) w = i + 1;
    fill.style.width = (100 * Math.min(t, DUR) / DUR) + '%';
    ro.innerHTML = '';
    var a = document.createElement('b'); a.textContent = 't = ' + t.toFixed(1) + ' s'; ro.appendChild(a);
    ro.appendChild(document.createTextNode(w ? '  ·  window ' + w + ' of ' + starts.length + (w === 1 ? ' (from the real frame)' : ' (from its own last frame)') : '  ·  context frame'));
    requestAnimationFrame(tick);
  }}
  tl.addEventListener('click', function (e) {{ var r = tl.getBoundingClientRect(); v.currentTime = DUR * Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); }});
  var pb = document.getElementById('play');
  function toggle() {{ if (v.paused) v.play().catch(function () {{}}); else v.pause(); }}
  function sync() {{ pb.textContent = v.paused ? 'Play' : 'Pause'; }}
  pb.addEventListener('click', toggle);
  document.getElementById('restart').addEventListener('click', function () {{ v.currentTime = 0; v.play().catch(function () {{}}); }});
  v.addEventListener('click', toggle);
  v.addEventListener('play', sync); v.addEventListener('pause', sync);
  document.addEventListener('keydown', function (e) {{ if (e.key === ' ' && e.target === document.body) {{ e.preventDefault(); toggle(); }} }});
  v.addEventListener('loadedmetadata', function () {{ v.playbackRate = st.r; }});
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) v.autoplay = false;
  render(true); sync(); requestAnimationFrame(tick);
}})();
</script>
"""
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size} bytes); strips in {vdir} ({len(list(vdir.glob('*.mp4')))} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="<tag>=<step>, e.g. loop4500=4500 loop2500=2500 (results/rollout/<tag>)")
    ap.add_argument("--out", default=str(ROLL / "page" / "long_rollout.html"))
    ap.add_argument("--urls", default=None, help="json {strip filename: asset url} after uploading the strips as artifact assets")
    ap.add_argument("--box", default="2×H100 SXM")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--allow-missing", action="store_true", help="preview: skip cases whose clips are not finished yet")
    a = ap.parse_args()
    models = []
    for m in a.models:  # tag=step[=label[=trained depths]]
        parts = m.split("=")
        tag, step = parts[0], int(parts[1])
        label = parts[2] if len(parts) > 2 and parts[2] else f"looped, step {step}"
        trained = [int(x) for x in parts[3].split(",")] if len(parts) > 3 and parts[3] else [4]
        models.append((tag, step, label, trained))
    urls = json.loads(Path(a.urls).read_text()) if a.urls and Path(a.urls).exists() else {}
    build(models, Path(a.out), urls, a.box, a.windows, a.allow_missing)
