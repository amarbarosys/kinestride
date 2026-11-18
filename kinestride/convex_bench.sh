#!/usr/bin/env bash
# GPU cost of the convex narrowphase per pair class and precision (convex_gpu.mojo, one thread per case, 6000 random poses
# of convex_check.py), and its kernel resources (rocprofv3 --kernel-trace: VGPR, scratch, LDS per kernel).
# usage: kinestride/convex_bench.sh build            (CPU: case files and four binaries)
#        gpu-wait run --priority 60 --timeout 600 -- kinestride/convex_bench.sh run
set -euo pipefail
cd "$(dirname "$0")/.."
X=.work/convex; M=.venv/bin/python; fail=0
if [ "$1" = build ]; then
  for k in ellipsoid box; do
    $M -m kinestride.convex_check $k 6000 f32 1e-4 1 >/dev/null || true  # writes the case files (the gate's own verdict is check.sh convex)
    W=$X/$k-f32; NC=$(( $(stat -c %s $W/cases.f64) / 320 ))
    for p in f32 f64; do
      FL=""; [ $p = f64 ] && FL="-DCCD64=1"
      tools/bin/mojo-cached build $FL -DNCASES=$NC -DNHV=$(( $(stat -c %s $W/hverts.f32) / 12 )) -DNHF=$(( $(stat -c %s $W/hfirst.i32) / 4 )) \
        -DNHE=$(( $(stat -c %s $W/hedges.i32) / 4 )) -DNHM=$(( $(stat -c %s $W/hmeta.i32) / 12 )) kinestride/convex_gpu.mojo -o $W/cg-$p 2>&1 | grep -A4 error && { echo "FAIL build $k $p"; fail=$((fail+1)); } || true
    done
  done
else
  for k in ellipsoid box; do for p in f32 f64; do
    W=$X/$k-f32
    echo -n "$k $p: "; $W/cg-$p $PWD/$W/ || { echo "FAIL run $k $p"; fail=$((fail+1)); }
  done; done
  for k in ellipsoid box; do  # VGPR, scratch (spill) bytes, LDS per kernel (rocprofv3 kernel trace; --pmc crashes on our kernels)
    rm -rf $X/prof-$k
    /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d $X/prof-$k -o $k -- $PWD/$X/$k-f32/cg-f64 $PWD/$X/$k-f32/ > /dev/null 2>&1 || { echo "FAIL rocprofv3 $k"; fail=$((fail+1)); continue; }
    echo "$k f64 kernel:"; ~/iTools/bin/rocprof-kernels "$(find $X/prof-$k -name '*kernel_trace.csv' | head -1)" --match kernel
  done
fi
[ $fail -eq 0 ] || { echo "FAIL $fail steps"; exit 1; }
