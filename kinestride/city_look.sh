#!/usr/bin/env bash
# One city frame from render.mojo -DCITY=1 (under gpu-wait): the fly model as the placeholder, the camera given here.
# usage: kinestride/city_look.sh OUT W H MODE ex ey ez tx ty tz fovy seed ox oy far
#   MODE look  -> OUT/city.ppm      MODE depth -> OUT/city.depth, checked against city_hit on the CPU (city_check frame)
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1 w=$2 h=$3 mode=$4; shift 4
flags=(-DCITY=1); [ "$mode" = depth ] && flags+=(-DCITY_DEPTH=1)
log=$out/city-$mode.log; mkdir -p "$out"
.venv/bin/python -m kinestride.render_look "$out" 1 "$w" "$h" "${flags[@]}" > "$log" 2>&1 || { echo "FAIL city_look build: see $log"; exit 1; }
cam=("$1" "$2" "$3" "$4" "$5" "$6" "$7"); city=("$8" "$9" "${10}" "${11}")
if [ "$mode" = depth ]; then
  "$out/render1" "$out/" "$out/smooth.bin" "$out/city.rgba" 0 "${cam[@]}" 0 "${city[@]}" >> "$log" 2>&1 || { echo "FAIL city_look render: see $log"; exit 1; }
  mv "$out/city.rgba" "$out/city.depth"
  .venv/bin/mojo build -I kinestride kinestride/city_check.mojo -o "$out/city_check" >> "$log" 2>&1
  "$out/city_check" frame "$out/city.depth" "$w" "$h" "${cam[@]}" "${city[@]}"
else
  "$out/render1" "$out/" "$out/smooth.bin" "$out/city.ppm" 0 "${cam[@]}" 0 "${city[@]}" | tee -a "$log"
fi
