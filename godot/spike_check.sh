#!/usr/bin/env bash
# Godot spike gates (docs/plans/GODOT_SPIKE.md). Export: character.blend + floor (plane-mesh pairs only), 1,024 animals at
# qpos0 + 2 cm, no actuators. G0: ks_ping (a GPU launch) from C, headless Godot, windowed Godot on Vulkan and GL.
# G1: 101 ks_step calls through the .so give the same pose bytes as the engine binary's smooth.bin (1 + 100 launches, K=10).
# N1: ks_npc (npc-mind base brains + world) for 200 ticks: per-tick NPC x, y, face byte-equal to `npc_live base grid 0 200`
# (oracle .work/npclive/npc_live0 and weights from kinestride/check.sh npclive; NPC_MIND as there).
# G3: ks_part (Godot MultiMesh [R | t] per part, geom quaternion applied) at K=1 after 300 steps equals MuJoCo geom_xmat and
# geom_xpos of the qpos before the last step (pose rows are the kinematics at the start of a step), 1e-5.
# usage: gpu-wait run --timeout 1200 -- env WAYLAND_DISPLAY=.. DISPLAY=.. XDG_RUNTIME_DIR=.. godot/spike_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
W=.work/godot; D=$W/g1; F=1024; K=10; mkdir -p $D
M=~/iTools/bin/mojo-cached
[ -s .work/blender/char/character.blend ] || blender -b --factory-startup -P tools/blender_rig.py -- \
  .work/blender/mjcf_export_plugin-3fb51ccfe559f66714601180f8268f8587e442af.py .work/blender/char character > $W/char.log 2>&1 \
  || { echo "FAIL blender character: $W/char.log"; exit 1; }
KINDS=plane-mesh .venv/bin/python - .work/blender/char/character.blend $D $F <<'PY' || { echo "FAIL export: character + floor"; exit 1; }
import sys, numpy as np
from pathlib import Path
from kinestride import blend
from kinestride.model import export, with_auto_pairs, FEET
x, d, F = blend.resolve(sys.argv[1]), sys.argv[2], int(sys.argv[3])
fx = x.replace('.xml', '_floor.xml')
Path(fx).write_text(Path(x).read_text().replace('  <worldbody>\n', '  <worldbody>\n    <geom name="floor" type="plane" size="50 50 0.1"/>\n', 1))
px = with_auto_pairs(fx, fx.replace('.xml', '_pairs.xml')); Path(f'{d}/model.txt').write_text(px)  # G3's MuJoCo oracle
m = export(px, d, FEET, True)
with open(f'{d}/parts.txt', 'w') as f:  # per part: geom-frame half sizes, geom quaternion in its body (main.gd, ks_part)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] > 0:
            f.write(' '.join(f'{v:.9g}' for v in [*m.geom_aabb[g][3:], *m.geom_quat[g]]) + '\n')
q = np.tile(m.qpos0, (F, 1)); q[:, 2] += 0.02
q.astype(np.float32).T.copy().tofile(f'{d}/start-qpos.bin'); np.zeros((m.nv, F), np.float32).tofile(f'{d}/start-qvel.bin')
np.zeros((m.nu, F), np.float32).tofile(f'{d}/ctrl.bin')
PY
start() { cp $D/start-qpos.bin $D/qpos.bin; cp $D/start-qvel.bin $D/qvel.bin; }
[ -s $W/gdextension_interface.h ] || (cd $W && godot --headless --dump-gdextension-interface >/dev/null 2>&1)
FL="-I kinestride $(cat $D/sizes.txt) -DFLIES=$F -DSTAGE=7 -DK=$K"
$M build $FL kinestride/smooth.mojo -o $D/smooth > $D/build.log 2>&1 || { echo "FAIL build: smooth ($D/build.log)"; exit 1; }
$M build --emit shared-lib $FL kinestride/ks_lib.mojo -o $W/libkinestride.so >> $D/build.log 2>&1 || { echo "FAIL build: ks_lib ($D/build.log)"; exit 1; }
cc -O2 godot/ks_ping_test.c -ldl -o $W/ks_ping_test && cc -O2 godot/ks_world_test.c -ldl -o $W/ks_world_test \
  && cc -shared -fPIC -O2 -I $W godot/ks_gdext.c -L$W -lkinestride -Wl,-rpath,'$ORIGIN' -o $W/libks_gdext.so || { echo "FAIL build: C"; exit 1; }
$W/ks_ping_test $W/libkinestride.so | grep -q '^PASS' || { echo "FAIL g0 c host: ks_ping"; exit 1; }
(cd godot/spike && godot --headless --import >/dev/null 2>&1 || true)
run() { (cd godot/spike && godot --path . "$@" 2>&1) | grep -q '^PASS ks_ping in godot' || { echo "FAIL g0 godot $*: ks_ping"; exit 1; }; }
run --headless --quit
run --rendering-method forward_plus --quit-after 60
run --rendering-method gl_compatibility --quit-after 60
start; $W/ks_world_test $W/libkinestride.so $D/ 101 $D/lib_pose.bin 2> $D/g1_time.txt | grep -q '^PASS' || { echo "FAIL g1: ks_world_test"; exit 1; }
start; $D/smooth $D/ > $D/smooth.txt
cmp -s -n $(( $(.venv/bin/python -c "print(7 * (int('$(grep -o 'NBODY=[0-9]*' $D/sizes.txt | cut -d= -f2)') - 1) * $F * 4)") )) $D/lib_pose.bin $D/smooth.bin \
  || { echo "FAIL g1: .so poses differ from the engine binary after 101 launches"; exit 1; }
$M build --emit shared-lib -I kinestride $(cat $D/sizes.txt) -DFLIES=$F -DSTAGE=7 -DK=1 kinestride/ks_lib.mojo -o $D/libk1.so >> $D/build.log 2>&1 \
  || { echo "FAIL build: ks_lib K=1 ($D/build.log)"; exit 1; }
NQ=$(grep -o 'NQ=[0-9]*' $D/sizes.txt | cut -d= -f2)
start; PARTS=$D/parts.txt NQ=$NQ $W/ks_world_test $D/libk1.so $D/ 300 $D/k1_pose.bin 2>/dev/null | grep -q '^PASS' || { echo "FAIL g3: ks_world_test PARTS"; exit 1; }
.venv/bin/python -I - $D $F <<'PY' || exit 1
import sys, numpy as np, mujoco
d, F = sys.argv[1], int(sys.argv[2])
m = mujoco.MjModel.from_xml_path(open(f'{d}/model.txt').read()); dd = mujoco.MjData(m)
geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0]
got = np.fromfile(f'{d}/k1_pose.bin.parts', np.float32).reshape(len(geoms), F, 3, 4)
q = np.fromfile(f'{d}/k1_pose.bin.qpos', np.float32)[:m.nq * F].reshape(m.nq, F)
err = 0.0
for a in range(F):
    dd.qpos[:] = q[:, a]; mujoco.mj_kinematics(m, dd)
    err = max(err, float(np.abs(got[:, a] - np.concatenate([dd.geom_xmat[geoms].reshape(-1, 3, 3), dd.geom_xpos[geoms][:, :, None]], 2)).max()))
print(f'g3 max abs err {err:.2e}', file=open(f'{d}/g3.txt', 'w'))
sys.exit(f'FAIL g3: part transforms vs MuJoCo geom frames, max abs err {err:.2e}' if err >= 1e-5 else 0)
PY
NPC=${NPC_MIND:-$HOME/Projects/NN-lanes/kinestride-pin/npc-mind}; NL=.work/npclive; T=$W/n1; mkdir -p $T/inc
[ -x $NL/npc_live0 ] && [ -s $NL/wout_0.bin ] || { echo "FAIL n1: run kinestride/check.sh npclive first ($NL/npc_live0, wout_0.bin)"; exit 1; }
ln -sfn "$NPC/world.mojo" $T/inc/npc_world.mojo
$M build --emit shared-lib -I "$NPC" -I $T/inc kinestride/ks_npc.mojo -o $W/libks_npc.so > $T/build.log 2>&1 || { echo "FAIL build: ks_npc ($T/build.log)"; exit 1; }
cc -O2 godot/ks_npc_test.c -ldl -o $W/ks_npc_test || { echo "FAIL build: ks_npc_test"; exit 1; }
$W/ks_npc_test $W/libks_npc.so $NL/ 0 200 $T/lib_trace.bin | grep -q '^PASS' || { echo "FAIL n1: ks_npc_test"; exit 1; }
[ -s $T/live.bin ] || $NL/npc_live0 $NL/ base grid 0 200 $T/live.bin > /dev/null
.venv/bin/python -I - $T <<'PY' || exit 1
import sys, numpy as np
N, F, T, d = 100, 8, 200, sys.argv[1]
rec = 3 * N * 4 + 4 * N * 8 + 3 * F * 4 + 3 * F * 4  # npc_live tick record: x,y,face; needs; body pose; grid move
live = np.fromfile(f'{d}/live.bin', np.uint8)
want = np.stack([np.frombuffer(live[t * rec:t * rec + 3 * N * 4].tobytes(), np.int32) for t in range(T)])
got = np.fromfile(f'{d}/lib_trace.bin', np.int32).reshape(T, 3 * N)
bad = np.flatnonzero((got != want).any(1))
sys.exit(f'FAIL n1: tick {bad[0]} first of {bad.size} differing from npc_live' if bad.size else 0)
PY
echo "PASS g0 (C, headless, Vulkan, GL) + g1 (101 launches byte-equal; $(grep -o "[0-9.]* us per launch" $D/smooth.txt) at F=$F K=$K) + n1 (200 ticks byte-equal to npc_live) + g3 ($(cat $D/g3.txt))"
