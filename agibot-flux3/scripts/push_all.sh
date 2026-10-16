#!/bin/bash
# Push the private repo and mirror NOTES.md / scripts / ldvflux / configs / results text into the public repo, then push it.
#   GH_TOKEN=ghp_... bash scripts/push_all.sh ["commit message for the mirror"]
# (tokens are never stored on disk; pass GH_TOKEN in the environment of this one command)
set -eu
: "${GH_TOKEN:?GH_TOKEN}"
MSG=${1:-"agibot-flux3: sync notes, scripts and results"}
cd /workspace/loop-dit
git push -q "https://naomili0924:${GH_TOKEN}@github.com/naomili0924/loop-dit.git" HEAD:main && echo "private repo pushed ($(git rev-parse --short HEAD))"
M=/workspace/loop_dit_video/agibot-flux3
mkdir -p "$M/scripts" "$M/ldvflux" "$M/configs" "$M/results"
cp NOTES.md EVAL_MOTION.md "$M/" 2>/dev/null || true
cp scripts/*.sh scripts/*.py "$M/scripts/"
cp ldvflux/*.py "$M/ldvflux/"
cp configs/*.json "$M/configs/"
find results -maxdepth 2 \( -name '*.md' -o -name '*.json' -o -name '*.jsonl' \) -size -5M | while read -r f; do mkdir -p "$M/$(dirname "$f")"; cp "$f" "$M/$f"; done
cd /workspace/loop_dit_video
git add -A agibot-flux3
git commit -qm "$MSG" 2>/dev/null || echo "mirror: nothing new to commit"
git push -q "https://naomili0924:${GH_TOKEN}@github.com/naomili0924/looped-dit-video.git" HEAD:main && echo "public mirror pushed ($(git rev-parse --short HEAD))"
echo PUSH_ALL_DONE
