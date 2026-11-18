#!/usr/bin/env bash
# Test bodies for the check.sh gates, kept out of git (11 MB fly, 135 MB flight model; cp reflinks on btrfs, 0 B extra).
# Copies them into .work/ from a source checkout that already built them (default: fly-connectome).
# MuJoCo C sources for porting come from `mj-port source FN FILE` (3.9.0 tag via gh, cached in .work/mujoco-src).
# Flight is optional: without its sources it prints SKIP flight, not FAIL. Flybody assets: tools/fetch_flybody.sh.
# usage: [CMU=dir] [FLYBODY=dir] tools/assets.sh [SRC]   -> .work/assets/fly/body.xml, .work/assets/flight/flight.xml, .work/spider-body/, .work/assets/cmu/*.bvh
set -euo pipefail
cd "$(dirname "$0")/.."
S=${1:-$HOME/Projects/science/fly-connectome}; fail=0
copy() { [ -e "$1" ] || { echo "FAIL asset $2: $1 missing"; fail=$((fail+1)); return; }; mkdir -p "$(dirname "$2")"; rm -rf "$2"; cp -r "$1" "$2"; echo "ok $2 ($(du -sh "$2" | cut -f1))"; }
copy "$S/.work/agent/gpu-body-port/body-v2" .work/assets/fly
# flight-body leaves out 26 of flight.xml's 85 meshes; flybody's own assets fill them, checked by the name's suffix (the content sha1)
FL=$S/.work/flight-body; [ -e .work/flight-body/flight.xml ] && FL=.work/flight-body; F=${FLYBODY:-.work/flybody/flybody/fruitfly/assets}
if [ ! -e "$FL/flight.xml" ]; then
  echo "SKIP flight: $FL/flight.xml missing (run tools/fetch_flybody.sh, then tools/flight_export.py .work/flight-body with flybody installed)"
elif [ ! -d "$F" ]; then
  echo "SKIP flight: flybody assets missing at $F (run tools/fetch_flybody.sh, or set FLYBODY=dir)"
else
  copy "$FL" .work/assets/flight
  for m in $(grep -oE 'file="[^"]+"' .work/assets/flight/flight.xml | cut -d'"' -f2 | sort -u); do
    [ -e ".work/assets/flight/$m" ] && continue
    [ "$(sha1sum < "$F/${m%-*}.obj" 2>/dev/null | cut -c1-40).obj" = "${m##*-}" ] && cp "$F/${m%-*}.obj" ".work/assets/flight/$m" || { echo "FAIL asset flight mesh $m: not in $F"; fail=$((fail+1)); }
  done
fi
C=${CMU:-$HOME/Projects/gpu-media/Blender/.work/mocap}  # CMU mocap (mocap.cs.cmu.edu: copy, modify, redistribute freely)
if [ -d "$C" ]; then
  for f in 02_05 10_01 55_22 60_01; do copy "$C/$f.bvh" .work/assets/cmu/$f.bvh; done; copy "$C/index.txt" .work/assets/cmu/index.txt
else
  echo "SKIP cmu: $C missing (motion stages only; set CMU=dir with 02_05 10_01 55_22 60_01 .bvh from mocap.cs.cmu.edu)"
fi
.venv/bin/python -m kinestride.models.spider .work/spider-body > .work/spider-body.log 2>&1 && echo "ok .work/spider-body (generated)" || { echo "FAIL spider body: .work/spider-body.log"; fail=$((fail+1)); }
[ $fail -eq 0 ] || { echo "FAIL $fail assets"; exit 1; }
