#!/bin/bash
# Per-window comparison strips from the Phase-4 grid videos (grid = 8 rows of windows x [GT | generated], 512x384 cells).
# For each k and window i: GT | base | N=1 | N=2 | N=4 | N=8 | N=10  (cells scaled to 256x192), h264, 15 fps.
#   scripts/make_loop_strips.sh [RESULTS=results] [OUT=results/loop_videos] [STEP=6000]
set -e
R=${1:-/workspace/loop-dit/results}; OUT=${2:-/workspace/loop-dit/results/loop_videos}; STEP=${3:-6000}
mkdir -p "$OUT"
CW=512; CH=384; SW=256; SH=192
for k in 1 9; do
  # column sources (grid file, side): side 0 = GT half, 1 = generated half
  srcs=("$R/ft_step$STEP/test_k${k}_grid.mp4:0" "$R/base/test_k${k}_grid.mp4:1" "$R/ft_step$STEP/test_k${k}_grid.mp4:1")
  for n in 2 4 8 10; do f="$R/ft_step${STEP}_loops$n/test_k${k}_grid.mp4"; [ -f "$f" ] && srcs+=("$f:1"); done
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
