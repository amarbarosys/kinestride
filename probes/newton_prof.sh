#!/usr/bin/env bash
# rocprofv3 of the STAGE 6 step stopped after phase 5 (collision + constraints) and after phase 6 (Newton): device
# time, VGPR, scratch, LDS, bytes fetched and written per launch. Newton = stop6 - stop5.
# usage: NS="1024 65536" kinestride/bench.sh build MJCF 6; NS="1024 65536" kinestride/phases.sh build MJCF
#        gpu-wait run --priority 90 --timeout 1800 -- probes/newton_prof.sh MJCF
set -euo pipefail
cd "$(dirname "$0")/.."
X=.work/bench/$(basename "$1" .xml)-6; P=.work/newton-prof; fail=0
mkdir -p "$P"
for n in ${NS:-1024 65536}; do
  for k in 5 6; do
    for c in FETCH_SIZE WRITE_SIZE; do  # one counter per pass: both at once exceed gfx1100's counter hardware
      o=$P/$n-stop$k-$c
      /opt/rocm/bin/rocprofv3 --kernel-trace --pmc $c --output-format csv -d "$o" -o run -- "$X/$n/stop$k" "$X/$n/" > "$o.log" 2>&1 \
        || { echo "FAIL rocprof $n stop $k $c: $o.log"; fail=$((fail+1)); continue; }
      echo "=== $n stop$k $c: $(grep -vE '^[WIE]2026|^\[' "$o.log" | tail -1)"
      [ $c = WRITE_SIZE ] || find "$o" -name '*kernel_trace.csv' | head -1 | xargs -r ~/iTools/gpu-telemetry/rocprof-kernels/rocprof-kernels.sh
      find "$o" -name '*counter_collection.csv' | head -1 | xargs -r awk -F, 'NR==1{for(i=1;i<=NF;i++){gsub(/"/,"",$i);h[$i]=i}; next}
        {gsub(/"/,""); s+=$h["Counter_Value"]; c++} END{printf "'"$c"' total %.6g KB over %d dispatches\n", s, c}'
    done
  done
done
rm -f "$X"/*/smooth.bin
[ $fail -eq 0 ] || { echo "FAIL $fail rocprof runs"; exit 1; }
