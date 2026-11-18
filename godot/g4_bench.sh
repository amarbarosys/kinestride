#!/usr/bin/env bash
# G4 (private note GODOT_SPIKE): crowd-mode frame timings in windowed Godot at each F (default 1024 16384 65536).
# Each F gets its own export and library (FLIES is compile-time; exports in parallel), swapped in as .work/godot/libkinestride.so
# for its run, then the 1,024 build is restored. Prints the last "G4 ..." line per F.
# usage: godot/g4_bench.sh [F...]   (runs its own gpu-wait jobs; needs a Wayland/X session; spike_check.sh run first)
set -euo pipefail
cd "$(dirname "$0")/.."
W=.work/godot; R=$PWD; SIZES=("${@:-1024 16384 65536}"); SIZES=(${SIZES[@]})
[ -s $W/libkinestride.so ] && [ -s $W/g1/parts.txt ] || { echo "FAIL g4: run godot/spike_check.sh first"; exit 1; }
cp $W/libkinestride.so $W/libkinestride.1024.so
trap 'cp $W/libkinestride.1024.so $W/libkinestride.so' EXIT
for f in "${SIZES[@]}"; do [ $f = 1024 ] || F=$f EXPORT_ONLY=1 godot/spike_check.sh > $W/g4_export_$f.log 2>&1 & done
wait || { echo "FAIL g4: export ($W/g4_export_*.log)"; exit 1; }
E="env WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-} DISPLAY=${DISPLAY:-} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-}"
for f in "${SIZES[@]}"; do
  D=$W/g1; [ $f = 1024 ] || D=$W/g1_$f
  [ $f = 1024 ] && cp $W/libkinestride.1024.so $W/libkinestride.so || cp $D/libkinestride.so $W/libkinestride.so
  cp $D/start-qpos.bin $D/qpos.bin; cp $D/start-qvel.bin $D/qvel.bin
  gpu-wait run --timeout 900 -- $E bash -c "cd godot/spike && godot --path . -- crowd $R/$D/ $R/$W/g1/parts.txt $R/$W/g4_$f.png 300" > $W/g4_$f.log 2>&1 \
    || { echo "FAIL g4: F=$f run ($W/g4_$f.log)"; exit 1; }
  grep '^G4' $W/g4_$f.log | tail -1 || { echo "FAIL g4: F=$f printed no timings"; exit 1; }
done
