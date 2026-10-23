#!/usr/bin/env bash
# bisect_stage.sh STAGE: `git bisect run` helper for one check.sh stage on the current tree.
# Exit 0 PASS, 1 FAIL, 125 skip (stage absent or build broken). Builds on the CPU first when the tree's check.sh
# knows BUILD_ONLY; an older check.sh without it runs whole under gpu-wait, so the card is never used bare.
# usage (in a worktree with tools/lane_prep.sh links): git bisect run /abs/tools/bisect_stage.sh cone_hfield
set -uo pipefail
stage=${1:?STAGE}; L=.work/check; mkdir -p $L
grep -q "\"\$stage\" = $stage\b" kinestride/check.sh || { echo "SKIP $stage: not in this tree"; exit 125; }
if grep -q BUILD_ONLY kinestride/check.sh; then
  CHECK_LOCK_HELD=1 BUILD_ONLY=1 QUICK=1 timeout 900 kinestride/check.sh "$stage" > $L/build-$stage.log 2>&1 \
    || grep -q '^PASS\|^FAIL' $L/build-$stage.log || { echo "SKIP $stage: build broken ($L/build-$stage.log)"; exit 125; }
fi
gpu-wait run --shared --vram 2 --timeout 900 -- env QUICK=1 kinestride/check.sh "$stage" > $L/$stage.log 2>&1
rc=$?
r=$(git rev-parse --short HEAD)
if [ $rc -eq 0 ] && grep -q '^PASS' $L/$stage.log; then echo "PASS $stage $r"; exit 0; fi
grep -q '^FAIL' $L/$stage.log || { echo "SKIP $stage $r: no verdict, exit $rc ($L/$stage.log)"; exit 125; }
echo "FAIL $stage $r"; exit 1
