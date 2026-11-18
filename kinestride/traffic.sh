#!/usr/bin/env bash
# DRAM bytes per launch of the engine step from L2->memory request counters (bench.sh build first).
# Needs, on this box (private note rocprofiler-install): iGPU unbound from amdgpu and card1 at profile_standard,
# else rocprofv3 --pmc crashes or every GL2C counter reads 0. Restore `auto` after; ms under profile_standard
# are not comparable with ms under auto.
# Reads = sum of GL2C_EA_RDREQ_{32,64,96,128}B x size. Writes = GL2C_EA_WRREQ_64B x 64 + GL2C_MC_WRREQ x 32
# (both raw, reported separately: the L2 is write-back, so only traffic that leaves it is counted).
# usage: gpu-wait run --priority 90 --timeout 1200 -- kinestride/traffic.sh MJCF [STAGE]
set -euo pipefail
cd "$(dirname "$0")/.."
lvl=$(cat /sys/class/drm/card1/device/power_dpm_force_performance_level)
[ "$lvl" = profile_standard ] || { echo "FAIL perf level: $lvl, need profile_standard"; exit 1; }
[ ! -e /sys/bus/pci/devices/0000:18:00.0/driver ] || { echo "FAIL iGPU 18:00.0 still bound: rocprofv3 --pmc crashes"; exit 1; }
X=.work/bench/$(basename "$1" .xml)-${2:-7}; fail=0
for n in ${NS:-1024 65536}; do
  d=$X/$n; o=$d/prof; rm -rf "$o"; mkdir -p "$o"
  for c in GL2C_EA_RDREQ_32B GL2C_EA_RDREQ_64B GL2C_EA_RDREQ_96B GL2C_EA_RDREQ_128B GL2C_EA_WRREQ_64B GL2C_MC_WRREQ; do
    timeout -s KILL 300 /opt/rocm/bin/rocprofv3 --pmc $c --kernel-include-regex smooth --output-format csv -d "$o/$c" -o p -- "$d/step" "$d/" > "$o/$c.log" 2>&1 \
      || { echo "FAIL $n $c: $o/$c.log"; fail=$((fail+1)); }
  done
  .venv/bin/python - "$o" "$n" <<'PY' || { echo "FAIL sum $n"; fail=$((fail+1)); }
import csv, glob, sys
o, n = sys.argv[1], int(sys.argv[2])
size = {'GL2C_EA_RDREQ_32B': 32, 'GL2C_EA_RDREQ_64B': 64, 'GL2C_EA_RDREQ_96B': 96, 'GL2C_EA_RDREQ_128B': 128, 'GL2C_EA_WRREQ_64B': 64, 'GL2C_MC_WRREQ': 32}
med = {}
for c in size:
    f = glob.glob(f"{o}/{c}/**/*counter_collection.csv", recursive=True)
    if not f: sys.exit(f"no csv for {c}")
    v = sorted(float(r['Counter_Value']) for r in csv.DictReader(open(f[0])) if 'smooth' in r['Kernel_Name'])
    med[c] = v[len(v) // 2]
rd = sum(med[c] * size[c] for c in size if 'RDREQ' in c)
print(f"{n} animals: DRAM read {rd/2**20:.1f} MB/launch = {rd/n:.0f} B/animal; "
      f"EA writes {med['GL2C_EA_WRREQ_64B']*64/2**20:.1f} MB ({med['GL2C_EA_WRREQ_64B']*64/n:.0f} B/animal); "
      f"MC writes {med['GL2C_MC_WRREQ']*32/2**20:.1f} MB; " + " ".join(f"{c.split('_',2)[2]}={med[c]:.0f}" for c in size))
PY
done
[ $fail -eq 0 ] || { echo "FAIL $fail"; exit 1; }
