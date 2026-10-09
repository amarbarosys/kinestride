#!/usr/bin/env bash
# Every check.sh stage in two phases, so the card is never held for compiling:
#   1. build: every stage's binaries on the CPU, in parallel, no GPU (BUILD_ONLY=1: model.py exports and builds, runs nothing)
#   2. test:  all stages in ONE gpu-wait job, every build a mojo-cached hit
# Builds are keyed by flags and sources, so parallel stages sharing an output dir cannot cache a wrong binary.
# QUICK=1 (default here) skips the planted-bug mutants; QUICK=0 (full merge gate) also builds every mutant on the CPU,
# one at a time: mutant-check edits sources in place and mojo-cached hashes every .mojo before compiling, so a parallel
# build could cache a mutant under a clean key. Measured on gravcomp: GPU hold 103.2 s -> 5.6 s.
# usage: tools/check_all.sh [STAGE ...]     logs: .work/check/build-<stage>.log, .work/check/<stage>.log
set -euo pipefail
cd "$(dirname "$0")/.."
L=.work/check; mkdir -p $L
stages=${*:-$(grep -oE '"\$stage" = [A-Za-z_0-9]+' kinestride/check.sh | awk '{print $NF}' | sort -u)}
export QUICK=${QUICK:-1}
t0=$(date +%s)
build() { printf '%s\n' $stages | xargs -P "$1" -I{} sh -c 'CHECK_LOCK_HELD=1 BUILD_ONLY=1 kinestride/check.sh {} > .work/check/build-{}.log 2>&1 || true'; }
QUICK=1 build "${JOBS:-8}"
[ "$QUICK" = 1 ] || build 1
t1=$(date +%s); echo "build phase: $(printf '%s\n' $stages | wc -l) stages in $((t1 - t0)) s (CPU)"
gpu-wait run --timeout 3600 -- bash -c '
  fail=0; bad=""
  for s in '"$(echo $stages)"'; do
    if kinestride/check.sh "$s" > .work/check/$s.log 2>&1 && grep -q "^PASS" .work/check/$s.log; then echo "PASS $s"; else echo "FAIL $s: .work/check/$s.log"; fail=$((fail+1)); bad="$bad $s"; fi
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail stages:$bad"; exit 1; }
  echo "PASS all stages"'
echo "test phase: $(( $(date +%s) - t1 )) s (one GPU job)"
