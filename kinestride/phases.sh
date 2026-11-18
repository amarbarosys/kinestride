#!/usr/bin/env bash
# Phase split of the STAGE 6 step: one binary per stop point (-DSTOP=k returns after phase k), timed on the
# bench states (bench.sh build MJCF 6 first). Phases: 1 kinematics, 2 comPos + crb, 3 factorM, 4 comVel +
# passive + rne, 5 collision + constraints, 6 Newton, 7 Euler. Phase cost = time(k) - time(k-1).
# usage: kinestride/phases.sh build MJCF; gpu-wait run --priority 90 --timeout 1800 -- kinestride/phases.sh run MJCF
set -euo pipefail
cd "$(dirname "$0")/.."
mode=$1; st=${STAGE:-6}; X=.work/bench/$(basename "$2" .xml)-$st${CONVEX:+-convex}; fail=0
for n in ${NS:-1024 65536}; do
  for k in 1 2 3 4 5 6 7; do
    if [ "$mode" = build ]; then
      tools/bin/mojo-cached build $(cat "$X/$n/sizes.txt") -DFLIES=$n -DSTAGE=$st -DWAVE=${WAVE:-0} -DSTOP=$k kinestride/smooth.mojo -o "$X/$n/stop$k" > "$X/$n/stop$k.log" 2>&1 || { echo "FAIL build $n stop $k: $X/$n/stop$k.log"; fail=$((fail+1)); }
    else
      echo -n "stop $k: "; "$X/$n/stop$k" "$X/$n/" || { echo "FAIL run $n stop $k"; fail=$((fail+1)); }
    fi
  done
done
rm -f "$X"/*/smooth.bin
[ $fail -eq 0 ] || { echo "FAIL $fail phase runs"; exit 1; }
