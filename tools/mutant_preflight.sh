#!/usr/bin/env bash
# CPU preflight for a check.sh stage, before any GPU time: every run builds (BUILD_ONLY=1), and every mutant matches exactly
# one line of its file and still compiles. A mutant whose sed matches nothing, or whose edit breaks the build, is "caught"
# by mutant-check without testing anything.
# usage: tools/mutant_preflight.sh SPEC   (lines: run|ENV|MJCF|STAGE  or  mut|ENV|MJCF|STAGE|FILE|SED; ENV like "XFRC=4 CONVEX=1")
set -euo pipefail
cd "$(dirname "$0")/.."
spec=${1:?usage: tools/mutant_preflight.sh SPEC}; M=.venv/bin/python; X=.work/preflight; mkdir -p $X; fail=0; k=0
while IFS='|' read -r kind envs mjcf stage file sedx; do
  [ -z "$kind" ] || [ "${kind:0:1}" = "#" ] && continue
  k=$((k+1)); log=$X/$k.log
  if [ "$kind" = mut ]; then
    n=$(sed -n "${sedx}p" "$file" | wc -l)
    [ "$n" = 1 ] || { echo "FAIL $k: mutant matches $n lines of $file (want 1): $sedx"; fail=$((fail+1)); continue; }
    cp "$file" $X/orig; sed -i "$sedx" "$file"
  fi
  if env $envs BUILD_ONLY=1 $M -m kinestride.model check "$mjcf" $X/b$k "$stage" 64 > $log 2>&1; then echo "PASS $k $kind builds ($stage, $(basename "$mjcf"))"; else echo "FAIL $k $kind does not build: $log"; fail=$((fail+1)); fi
  [ "$kind" = mut ] && cp $X/orig "$file"
done < "$spec"
[ $fail -eq 0 ] && echo "PASS preflight $k/$k" || { echo "FAIL preflight $fail/$k"; exit 1; }
