#!/usr/bin/env bash
# usage: kinestride/cmp_export.sh WORKTREE   (a lane branch must not change existing exports)
# fly + spider export of main vs a branch worktree must be byte-identical (mi.bin, mf.bin, sizes.txt)
set -euo pipefail
W=$1; R=$(cd "$(dirname "$0")/.." && pwd); O=$R/.work/cmpexp; rm -rf $O; mkdir -p $O; fail=0
for B in $R/.work/assets/fly/body.xml $R/.work/spider-body/spider.xml; do
  for side in main:$R lane:$W; do n=${side%%:*}; d=${side#*:}
    (cd $d && PYTHONPATH=$d $R/.venv/bin/python -c "from kinestride.model import export; export('$B', '$O/$n-$(basename $B .xml)', '.', True)" >/dev/null)
  done
  for f in mi.bin mf.bin sizes.txt; do cmp -s $O/main-$(basename $B .xml)/$f $O/lane-$(basename $B .xml)/$f && echo "SAME $(basename $B) $f" || { echo "DIFF $(basename $B) $f"; fail=1; }; done
done
[ $fail = 0 ] && echo "PASS export byte-identical" || { echo "FAIL export differs"; exit 1; }
