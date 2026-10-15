#!/bin/bash
# Per-window comparison strips for a trained looped model, from the eval grid videos
# (grid = 8 rows of windows x [GT | generated], 512x384 cells, 15 fps).
# Columns: GT | base N=1 | Phase-3 fine-tune N=1 | looped N=1 | N=2 | N=4 | N=8   (cells scaled to 256x192).
#   scripts/make_looped_strips.sh [TAG=loop2500] [OUT=results/loop_videos_<TAG>] [RESULTS=results]
set -e
TAG=${1:-loop2500}; OUT=${2:-/workspace/loop-dit/results/loop_videos_$TAG}; R=${3:-/workspace/loop-dit/results}
mkdir -p "$OUT"
CW=512; CH=384; SW=256; SH=192
for k in 1 9; do
  srcs=("$R/ft_step6000/test_k${k}_grid.mp4:0" "$R/base/test_k${k}_grid.mp4:1" "$R/ft_step6000/test_k${k}_grid.mp4:1")
  for n in 1 2 4 8; do f="$R/${TAG}_loops$n/test_k${k}_grid.mp4"; [ -f "$f" ] || { echo "missing $f"; exit 1; }; srcs+=("$f:1"); done
  for i in 0 1 2 3 4 5 6 7; do
    args=(); filt=""; j=0
    for s in "${srcs[@]}"; do
      f=${s%:*}; side=${s##*:}
      args+=(-i "$f")
      filt+="[$j:v]crop=$CW:$CH:$((side*CW)):$((i*CH)),scale=$SW:$SH[c$j];"
      j=$((j+1))
    done
    ins=""; for ((m=0; m<j; m++)); do ins+="[c$m]"; done
    filt+="${ins}hstack=inputs=$j[v]"
    ffmpeg -v error -y "${args[@]}" -filter_complex "$filt" -map "[v]" -r 15 -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "$OUT/k${k}_win$i.mp4"
  done
  echo "k=$k: ${#srcs[@]} columns -> $OUT/k${k}_win{0..7}.mp4"
done
ls -la "$OUT" | tail -n +2 | awk '{s+=$5} END {printf "total %.1f MB\n", s/1e6}'
