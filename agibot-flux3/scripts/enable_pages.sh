#!/bin/bash
# Enable GitHub Pages for the public repo from main:/docs (or switch an existing Pages config to it).
#   GH_TOKEN=ghp_... bash scripts/enable_pages.sh
set -eu
: "${GH_TOKEN:?GH_TOKEN}"
API=https://api.github.com/repos/naomili0924/looped-dit-video/pages
H=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")
code=$(curl -s -o /tmp/pages_resp.json -w '%{http_code}' -X POST "${H[@]}" "$API" -d '{"source":{"branch":"main","path":"/docs"}}')
if [ "$code" = 409 ]; then  # already enabled: point it at main:/docs
  code=$(curl -s -o /tmp/pages_resp.json -w '%{http_code}' -X PUT "${H[@]}" "$API" -d '{"source":{"branch":"main","path":"/docs"}}')
  [ "$code" = 204 ] && echo "Pages source updated to main:/docs"
fi
echo "http $code"; head -c 400 /tmp/pages_resp.json; echo
curl -s "${H[@]}" "$API" | grep -o '"html_url": *"[^"]*"\|"status": *"[^"]*"' | head -3
