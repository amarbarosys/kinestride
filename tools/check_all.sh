#!/usr/bin/env bash
# Every check.sh stage, each in its own copy of the tree (.work/par/<stage>: tracked + untracked sources, venv and the
# shared read-only caches linked by lane_prep.sh), so stages never share outputs, the check.sh lock or a mutated source.
#   1. build: every stage's binaries on the CPU, JOBS at a time, no GPU (BUILD_ONLY=1: model.py exports and builds)
#   2. test:  every stage its own gpu-wait --shared job, GPU_JOBS at a time, every build a mojo-cached hit
# mojo-cached keys -I by the argument as given, so the copies share one cache.
# QUICK=1 (default here) skips the planted-bug mutants; QUICK=0 (full merge gate) runs them too, safe in parallel
# because each mutant edits only its own copy.
# usage: tools/check_all.sh [STAGE ...]     logs: .work/check/build-<stage>.log, .work/check/<stage>.log
set -euo pipefail
cd "$(dirname "$0")/.."
R=$PWD; L=.work/check; mkdir -p $L .work/par
stages=${*:-$(grep -oE '"\$stage" = [A-Za-z_0-9]+' kinestride/check.sh | awk '{print $NF}' | sort -u)}
export QUICK=${QUICK:-1} R
t0=$(date +%s)
for s in $stages; do
  d=.work/par/$s; rm -rf "$d"; mkdir -p "$d"
  git ls-files -co --exclude-standard -z | rsync -a0 --files-from=- ./ "$d/"
  tools/lane_prep.sh "$d" - "$R"
done
printf '%s\n' $stages | xargs -P "${JOBS:-$(nproc)}" -I{} sh -c 'cd .work/par/{} && PYTHONPATH=$PWD BUILD_ONLY=1 kinestride/check.sh {} > "$R/.work/check/build-{}.log" 2>&1 || true'
t1=$(date +%s); echo "build phase: $(printf '%s\n' $stages | wc -l) stages in $((t1 - t0)) s (CPU)"
printf '%s\n' $stages | xargs -P "${GPU_JOBS:-4}" -I{} sh -c '
  if gpu-wait run --shared --timeout 3600 -- sh -c "cd .work/par/{} && PYTHONPATH=\$PWD kinestride/check.sh {}" > "$R/.work/check/{}.log" 2>&1 \
    && grep -q "^PASS" "$R/.work/check/{}.log"; then echo "PASS {}"; else echo "FAIL {}: .work/check/{}.log"; fi' | tee $L/summary.txt
bad=$(grep -c '^FAIL' $L/summary.txt || true)
echo "test phase: $(( $(date +%s) - t1 )) s (${GPU_JOBS:-4} GPU jobs)"
[ "$bad" -eq 0 ] || { echo "FAIL $bad stages:$(grep '^FAIL' $L/summary.txt | cut -d' ' -f2 | tr -d : | tr '\n' ' ')"; exit 1; }
echo "PASS all stages"
