#!/usr/bin/env bash
# Every .md a tracked file names must exist: ~/x.md on disk, a/b.md from the repo root, bare X.md in docs/ or the root.
# usage: tools/doclinks.sh   (PASS n links, or FAIL k/n <names> and exit 1)
set -euo pipefail
cd "$(dirname "$0")/.."
refs=$(git grep -h -o -E '[A-Za-z0-9_.~/-]*[A-Za-z0-9_]\.md\b' -- . ':!tools/doclinks.sh' | sort -u)
bad=""
for r in $refs; do
  case $r in '~/'*) p=$HOME/${r#'~/'};; */*) p=$r;; *) p=docs/$r; [ -e "$p" ] || p=$r;; esac
  [ -e "$p" ] || bad="$bad $r"
done
n=$(echo "$refs" | wc -l); k=$(echo $bad | wc -w)
[ "$k" -eq 0 ] || { echo "FAIL doclinks $k/$n:$bad"; exit 1; }
echo "PASS doclinks $n"
