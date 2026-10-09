#!/usr/bin/env bash
# Merge gate for lane commits: run check.sh stages on a CLEAN worktree at COMMIT (never a lane's working tree, which
# may hold uncommitted edits), data links as in the main checkout, every stage under gpu-wait.
# usage: [MERGE=1] kinestride/verify_commit.sh COMMIT STAGE [STAGE ...]   -> .work/verify/<commit>-<stage>.log
# First checks that COMMIT merges into main without conflicts (no GPU time spent on a branch that must rebase first).
# MERGE=1: on PASS, merge --no-ff into main. Uncommitted edits in the main checkout that the merge touches are set
#   aside as a patch (.work/verify/aside-<commit>.patch), the merge runs, the patch is re-applied (3-way).
# FIRST_FAIL=1: stop at the first failing stage (iteration); default runs every stage (merge gate).
# This is the stage chain for lanes too: commit, then run it on the sha from the lane worktree.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); C=$1; shift
if ! git -C "$R" merge-tree --write-tree main "$C" >/dev/null; then
  echo "FAIL merge: $C conflicts with main $(git -C "$R" rev-parse --short main):"
  git -C "$R" merge-tree --write-tree --name-only main "$C" | sed -n '2,/^$/p' | sed '/^$/d;s/^/  /'; exit 1
fi
W=$R/.work/verify-$C; L=$R/.work/verify; mkdir -p "$L"
git -C "$R" worktree add -q --detach "$W" "$C"
trap 'git -C "$R" worktree remove --force "$W"' EXIT
ln -s "$R/.venv" "$W/.venv"
mkdir -p "$W/.work"
for d in assets spider-body mujoco-src; do [ -e "$R/.work/$d" ] && ln -s "$R/.work/$d" "$W/.work/$d"; done
fail=0; bad=""
for s in "$@"; do
  if gpu-wait run --priority 70 --shared --vram 1 --timeout 3600 -- "$W/kinestride/check.sh" "$s" > "$L/$C-$s.log" 2>&1 \
     && grep -q '^PASS' "$L/$C-$s.log"; then  # exit 0 with no PASS line (lost log stream, empty stage) is not a pass
    echo "PASS $s ($(grep -cE '^PASS' "$L/$C-$s.log") PASS lines)"
  else
    echo "FAIL $s: $L/$C-$s.log"; fail=$((fail+1)); bad="$bad $s"
    [ "${FIRST_FAIL:-0}" = 1 ] && break
  fi
done
[ $fail = 0 ] || { echo "FAIL $fail/$# stages:$bad"; exit 1; }
echo "PASS $C: $*"
if [ "${MERGE:-0}" = 1 ]; then
  b=$(git -C "$R" for-each-ref --contains "$C" --format='%(refname:short)' refs/heads/team | head -1)
  git -C "$R" merge-tree --write-tree main "$C" >/dev/null || { echo "FAIL merge: main moved to $(git -C "$R" rev-parse --short main) and $C now conflicts; rebase"; exit 1; }
  M=$(git -C "$R" rev-parse --show-toplevel); touch=$(git -C "$M" diff --name-only main..."$C" | grep -Fxf - <(git -C "$M" diff --name-only HEAD) || true)
  P=$L/aside-$C.patch
  if [ -n "$touch" ]; then git -C "$M" diff -- $touch > "$P"; git -C "$M" checkout -- $touch; echo "aside: $(echo $touch) -> $P"; fi
  git -C "$M" merge --no-ff -q -m "Merge ${b:-$C} ($C): verified $*" "$C" && echo "MERGED $C -> $(git -C "$M" rev-parse --short main)"
  if [ -n "$touch" ]; then git -C "$M" apply --3way "$P" && git -C "$M" reset -q && echo "aside re-applied" || { echo "FAIL aside: resolve $P by hand"; exit 1; }; fi
fi
