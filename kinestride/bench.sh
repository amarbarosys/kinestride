#!/usr/bin/env bash
# Engine timing at 1k / 16k / 65k animals. STAGE 5 (S1 gate): full contact-free step, random states.
# STAGE 6 (S2 probe): foot contacts + Newton + step, distinct states from actuated rollouts (model.standing).
# usage: kinestride/bench.sh build MJCF [STAGE]   (CPU: export, states, one binary per batch size)
#        gpu-wait run --priority 90 --timeout 1200 -- kinestride/bench.sh run MJCF [STAGE]
# Correctness is check.sh step's job; run it first (a speed number from an unchecked build is not reported).
set -euo pipefail
cd "$(dirname "$0")/.."
mode=$1; mjcf=$2; stage=${3:-5}; X=.work/bench/$(basename "$mjcf" .xml)-$stage${CONVEX:+-convex}; fail=0
for n in ${NS:-1024 16384 65536}; do
  d=$X/$n; mkdir -p "$d"
  if [ "$mode" = build ]; then
    .venv/bin/python -c "
import os, sys, numpy as np
from kinestride.model import export, standing, FEET, VSCALE
from mj_oracle import states, velocities
m = export('$mjcf', '$d', FEET if $stage >= 6 else None, $stage >= 7, bool(os.environ.get('CONVEX')))
if not os.path.exists('$d/ctrl.bin'):  # states are seeded: reuse them, the export is always fresh
    q, v = standing(m, $n) if $stage >= 6 else (states(m, $n), velocities(m, $n, scale=VSCALE))
    q.astype(np.float32).T.copy().tofile('$d/qpos.bin')
    v.astype(np.float32).T.copy().tofile('$d/qvel.bin')
    u = np.zeros((m.nu, $n))  # actuated bench: servos hold the state's own joint angles, adhesion on
    if $stage >= 7:
        for i in range(m.nu):
            u[i] = q[:, m.jnt_qposadr[m.actuator_trnid[i, 0]]] if m.actuator_trntype[i] == 0 else 1.0
    u.astype(np.float32).tofile('$d/ctrl.bin')" >/dev/null || { echo "FAIL export $n"; fail=$((fail+1)); continue; }
    tools/bin/mojo-cached build $(cat "$d/sizes.txt") -DFLIES=$n -DSTAGE=$stage -DWAVE=${WAVE:-0} kinestride/smooth.mojo -o "$d/step" > "$d/build.log" 2>&1 || { echo "FAIL build $n: $d/build.log"; fail=$((fail+1)); }  # exit code, not output: 1.1 deprecation notes span lines
  else
    "$d/step" "$d/" || { echo "FAIL run $n"; fail=$((fail+1)); }
  fi
done
[ $fail -eq 0 ] || { echo "FAIL $fail/3 batch sizes"; exit 1; }
