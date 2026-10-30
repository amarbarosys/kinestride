#!/usr/bin/env bash
# G0 gate (docs/plans/GODOT_SPIKE.md): libkinestride.so's ks_ping (a GPU launch) from a C host, from headless Godot,
# and from windowed Godot on Vulkan and on GL, so HIP shares the process with Godot's renderer.
# usage: gpu-wait run --timeout 600 -- godot/g0_check.sh   (needs a Wayland/X session for the windowed runs)
set -euo pipefail
cd "$(dirname "$0")/.."
W=.work/godot; mkdir -p $W
[ -s $W/gdextension_interface.h ] || (cd $W && godot --headless --dump-gdextension-interface >/dev/null 2>&1)
.venv/bin/mojo build --emit shared-lib kinestride/ks_lib.mojo -o $W/libkinestride.so 2>&1 | grep -v warning || true
[ -s $W/libkinestride.so ] || { echo "FAIL build: libkinestride.so"; exit 1; }
cc -O2 godot/ks_ping_test.c -o $W/ks_ping_test || { echo "FAIL build: ks_ping_test"; exit 1; }
cc -shared -fPIC -O2 -I $W godot/ks_gdext.c -L$W -lkinestride -Wl,-rpath,'$ORIGIN' -o $W/libks_gdext.so || { echo "FAIL build: libks_gdext.so"; exit 1; }
$W/ks_ping_test $W/libkinestride.so | grep -q '^PASS' || { echo "FAIL c host: ks_ping"; exit 1; }
(cd godot/spike && godot --headless --import >/dev/null 2>&1 || true)
run() { (cd godot/spike && godot --path . "$@" 2>&1) | grep -q '^PASS ks_ping in godot' || { echo "FAIL godot $*: ks_ping"; exit 1; }; }
run --headless --quit
run --rendering-method forward_plus --quit-after 60
run --rendering-method gl_compatibility --quit-after 60
echo "PASS g0: ks_ping from C, headless Godot, Godot Vulkan, Godot GL"
