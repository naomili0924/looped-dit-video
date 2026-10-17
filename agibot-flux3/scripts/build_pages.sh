#!/bin/bash
# Rebuild the GitHub Pages site (public repo, docs/) from results/ and configs/models.json:
#   docs/index.html            models, headline numbers, how to run the two evaluations
#   docs/loop-depth/           "Trained Loop Depth" (normal eval) page + strips videos_<tag>/
#   docs/long-rollout/         "Long Rollout" page + strips videos/
#   scripts/build_pages.sh [DOCS=/workspace/loop_dit_video/docs]     then GH_TOKEN=... scripts/push_all.sh "..."
set -eu
cd "$(dirname "$0")/.."
DOCS=${1:-/workspace/loop_dit_video/docs}
PY=flux-action/.venv/bin/python
mkdir -p "$DOCS/loop-depth" "$DOCS/long-rollout/videos"; touch "$DOCS/.nojekyll"
# models with a complete normal eval (results/<tag>_loops{1,2,4,8}/summary.json), in registry order
TAGS=$($PY -c 'import json,os
for m in json.load(open("configs/models.json"))["models"]:
    t=m["tag"]
    if all(os.path.exists(f"results/{t}_loops{n}/summary.json") for n in (1,2,4,8)): print(t)')
for t in $TAGS; do
  [ -d results/loop_videos_$t ] || bash scripts/make_looped_strips.sh $t
  rm -rf "$DOCS/loop-depth/videos_$t"; cp -r results/loop_videos_$t "$DOCS/loop-depth/videos_$t"
done
$PY scripts/build_loop_gallery.py --tags $TAGS --out "$DOCS/loop-depth/index.html"
# models with a complete long rollout (120 clips)
MODELS=$($PY - <<'PY'
import json, glob
for m in json.load(open("configs/models.json"))["models"]:
    t = m["tag"]
    if len(glob.glob(f"results/rollout/{t}/clips/*/*.mp4")) >= 120:
        print(f"{t}={m['step']}={m['label']}={','.join(map(str, m['trained_depths']))}")
PY
)
cp -n results/rollout/page/videos/*.mp4 "$DOCS/long-rollout/videos/" 2>/dev/null || true   # reuse strips composed for the artifacts
IFS=$'\n'; set -- $MODELS; unset IFS
$PY scripts/build_rollout_page.py --models "$@" --out "$DOCS/long-rollout/index.html"
$PY scripts/build_index_page.py --out "$DOCS/index.html"
for f in "$DOCS/index.html" "$DOCS/loop-depth/index.html" "$DOCS/long-rollout/index.html"; do $PY scripts/wrap_html.py "$f"; done   # fragments -> full documents
echo "pages built in $DOCS: normal eval [$TAGS], long rollout [$(printf '%s; ' "$@")]"
du -sh "$DOCS" | cut -f1
