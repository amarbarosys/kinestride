#!/usr/bin/env bash
# Every ported stage against MuJoCo on two bodies (fly: free + hinge; 8-leg spider test model:
# free + ball + hinge + slide), plus a mutation that must fail (hinge angle sign flipped).
# usage: gpu-wait run --timeout 1200 -- [env BODY=path.xml] kinestride/check.sh [STAGE]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${CHECK_LOCK_HELD:-}" ]; then  # a stage that calls another (solver -> loopk) passes the lock down
  mkdir -p .work; exec 9>>.work/check.lock  # one stage per worktree: mutant-check edits sources in place (SHARED rule 13)
  flock -n 9 || { echo "FAIL check.sh: another stage is running in $(pwd) ($(tail -n 1 .work/check.lock)); run stages one at a time"; exit 1; }
  echo "pid $$ stage ${1:-kinematics} since $(date +%T)" > .work/check.lock
  export CHECK_LOCK_HELD=1
fi
stage=${1:-kinematics}; M=.venv/bin/python; X=.work; fail=0
[ "${QUICK:-0}" = 1 ] && { echo "QUICK=1: no mutants (all states kept, same bars): an iteration run, NOT a gate (verify_commit runs the full stage)"; export QUICK; trap 'echo "QUICK=1 run finished: rerun without QUICK before READY"' EXIT; }
if [ "$stage" = convex ]; then  # GJK + EPA narrowphase (convex.mojo, CPU build) vs MuJoCo mj_collision, per pair kind: float64 exact, float32 bars
  KINDS=${KINDS:-ellipsoid box}
  for kind in $KINDS; do
    $M -m kinestride.convex_check $kind 6000 f64 1e-9 1 || { echo "FAIL convex $kind f64"; fail=$((fail+1)); }
    $M -m kinestride.convex_check $kind 6000 f32 1e-4 1 || { echo "FAIL convex $kind f32"; fail=$((fail+1)); }
  done
  # the gate must fail with the EPA contact normal flipped
  ~/iTools/bin/mutant-check kinestride/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    $M -m kinestride.convex_check ellipsoid 600 f64 1e-9 3 || fail=$((fail+1))
  # box-mesh only: the gate must fail with the perturbation search's negative tilt dropped (contact count)
  case " $KINDS " in *" box "*) ~/iTools/bin/mutant-check kinestride/convex.mojo 's/var sgn = S(-1) if angle_id == 0 else S(1)/var sgn = S(1)/' -- \
    $M -m kinestride.convex_check box 600 f64 1e-9 3 || fail=$((fail+1));; esac
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all convex checks"; exit 0
fi
if [ "$stage" = city ]; then  # the endless city (city.mojo, CPU build): point march, rebase at 10^6 m, streets, golden snapshot
  C="mojo build -I kinestride kinestride/city_check.mojo -o $X/city/check && $X/city/check kinestride/city_golden.bin"
  mkdir -p $X/city; PATH=$PWD/.venv/bin:$PATH bash -c "$C" || { echo "FAIL city check"; fail=$((fail+1)); }
  for m in 's/seed + 7)/seed + 9)/' 's/R \* (1 + a \* cos/R * (1 - a * cos/' 's/if mx < my:/if mx > my:/' 's/for i in range(1, 33):/for i in range(1, 9):/'; do
    ~/iTools/bin/mutant-check kinestride/city.mojo "$m" -- env PATH=$PWD/.venv/bin:$PATH bash -c "$C" || fail=$((fail+1))  # lot hash, footprint sign, DDA order, polar steps
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all city checks"; exit 0
fi
if [ "$stage" = hfield ]; then  # exact heightfield contact (mjc_ConvexHField: GJK + EPA per cell prism, convex.mojo, CPU build) on a bumpy seeded terrain vs MuJoCo's own hfield geom
  $M -m kinestride.hfield_check 6000 f64 1 || { echo "FAIL hfield f64"; fail=$((fail+1)); }
  $M -m kinestride.hfield_check 6000 f32 1 || { echo "FAIL hfield f32"; fail=$((fail+1)); }
  # the gate must fail with the contact normal flipped, and with the prism height test inverted (cells under the mesh skipped)
  ~/iTools/bin/mutant-check kinestride/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    $M -m kinestride.hfield_check 600 f64 3 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/convex.mojo 's/if o1.prism\[3\]\[2\] < zmin and/if o1.prism[3][2] > zmin and/' -- \
    $M -m kinestride.hfield_check 600 f64 3 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all hfield checks"; exit 0
fi
if [ "$stage" = arena ]; then  # the original arena (box walls, stripes, beacons; ellipsoid stones): every pair incl. box-mesh and ellipsoid-mesh (GJK + EPA), states moved into walls and stones
  B=.work/assets/fly/body.xml
  # bar REL=3e-3 (default 1e-5), measured: qacc 7.1e-4, qacc_euler 2.0e-3 (3 x the float32 input floor is 3.5e-4 / 1.5e-3). `row_diag.py` on the same
  # states: convex contacts' aref within 6e-5 of the state's largest row, plane contacts 1.3e-4, ncon equal in all 64 states: the rest is the
  # engine's float32 solver on states with a 0.03 mm deep wall or stone contact, not the convex pairs. The gate must fail with the convex normal flipped.
  for st in contact actuated; do
    CONVEX=1 REL=3e-3 $M -m kinestride.model check $B $X/arena-$st $st 64 || { echo "FAIL arena $st"; fail=$((fail+1)); }
  done
  # the gate must fail with the convex contact normal flipped
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/var normal = cs.normal\[count\].cast\[DType.float32\]()/var normal = -cs.normal[count].cast[DType.float32]()/' -- \
    env CONVEX=1 REL=3e-3 $M -m kinestride.model check $B $X/mutant contact 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all arena checks"; exit 0
fi
if [ "$stage" = convexstep ]; then  # box and ellipsoid stones in a CHAINED engine run vs MuJoCo mj_step: fly (arena walls and stones) and spider (stones under its feet, model.stones)
  # per body: (1) 20 steps in one launch, qpos and qvel vs 20 chained mj_step, states whose own MuJoCo trajectory moves under a 1e-6 perturbation dropped (model.check,
  # at most half, listed); (2) the full one-step fields (ncon, qacc, qfrc_constraint, ...) on the states the engine reached after 3 steps (spider: 1), so forces are gated
  # on engine-evolved contacts too. Bars are measured stated bars: trajectories REL 1e-3, forces REL 1e-4 (measured fly qfrc_constraint 1.1e-5, qacc_euler 2.9e-5 on engine-evolved states; arena's 3e-3 is its deep 0.03 mm contacts).
  FLY=.work/assets/fly/body.xml
  SP=$($M -c "from kinestride.model import stones; print(stones('.work/spider-body/spider.xml', 'spider_stones.xml'))")
  N=${CVS_COUNT:-64}  # states per body (CVS_COUNT=8: a short probe before the full run)
  for B in $FLY $SP; do
    n=$(basename $B .xml); S=; M_MIN=; EK=3; [ $B = $SP ] && { S=STONES=1; EK=1; M_MIN=CONVEX_MIN=$((N / 16)); }  # the spider's feet hover at the 1 mm margin: its stone contacts toggle within 3 steps, so its evolved states are 1 step on and keep few of them (stone contacts there are gated by the K=20 trajectory; the force check needs 1/16 of the states to hold one)
    env CONVEX=1 REL=1e-3 $S $M -m kinestride.model check $B $X/cvs-$n-20 actuated $N 20 || { echo "FAIL convexstep $n K=20"; fail=$((fail+1)); }
    env CONVEX=1 REL=1e-3 $S SAVE_STATES=$X/cvs-$n.npz $M -m kinestride.model check $B $X/cvs-$n-3 actuated $N $EK || { echo "FAIL convexstep $n K=$EK"; fail=$((fail+1)); }
    env CONVEX=1 REL=${CVS_FREL:-1e-4} $S $M_MIN STATES=$X/cvs-$n.npz $M -m kinestride.model check $B $X/cvs-$n-f actuated $N || { echo "FAIL convexstep $n forces on engine states"; fail=$((fail+1)); }
  done
  # the gate must fail with the EPA contact normal flipped, on the fly and on the spider (32 states each: a flipped normal moves every kept state)
  ~/iTools/bin/mutant-check kinestride/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    env CONVEX=1 REL=1e-3 $M -m kinestride.model check $FLY $X/mutant actuated ${CVS_MUTANT_COUNT:-32} 20 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    env CONVEX=1 REL=1e-3 STONES=1 $M -m kinestride.model check $SP $X/mutant actuated ${CVS_MUTANT_COUNT:-32} 20 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all convexstep checks"; exit 0
fi
if [ "$stage" = cube ]; then  # closed 40 mm cube of planes (floor, 4 walls, ceiling; 330 pairs), states on every face
  C=$($M -c "from kinestride.model import cube; print(cube('${BODY:-.work/assets/fly/body.xml}'))")
  $M -m kinestride.model check $C $X/cube-$(basename ${BODY:-body.xml} .xml) actuated 64 || { echo "FAIL cube"; fail=$((fail+1)); }
  # the gate must fail when the bounding-sphere skip drops pairs that touch (rbound added instead of subtracted)
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/(pos2 - pos1)).reduce_add() - mf\[pf + 12\] > margin/(pos2 - pos1)).reduce_add() + mf[pf + 12] > margin/' -- \
    $M -m kinestride.model check $C $X/mutant actuated 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cube checks"; exit 0
fi
if [ "$stage" = terrain_exact ]; then  # exact heightfield contact in the step (GJK + EPA per cell prism on the terrain function) on bumpy seeded terrain vs MuJoCo's OWN hfield geom (not the plane twin), fly and spider
  # bar REL=4e-4 (default 1e-5): the engine takes node heights from the float32 terrain function, MuJoCo from its float32 table
  # (hfield_check ... func: 1e-7 mm apart, 1e-4 of a contact depth); measured fly qacc 2.1e-4, others <= 3.4e-5. The gate must fail with the contact normal flipped
  for B in ${BODY:-.work/assets/fly/body.xml .work/spider-body/spider.xml}; do
    N=$($M -c "from kinestride.model import terrain; print(terrain('$B', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
    for st in contact actuated; do
      HFEXACT=1 REL=4e-4 $M -m kinestride.model check $N $X/terrain-exact-$(basename $B .xml)-$st $st 64 || { echo "FAIL terrain_exact $(basename $B .xml) $st"; fail=$((fail+1)); }
    done
  done
  N=$($M -c "from kinestride.model import terrain; print(terrain('.work/assets/fly/body.xml', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
  ~/iTools/bin/mutant-check kinestride/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    env HFEXACT=1 REL=4e-4 $M -m kinestride.model check $N $X/mutant contact 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all terrain_exact checks"; exit 0
fi
if [ "$stage" = terrain ]; then  # heightfield ground (tangent plane) on a tilted ramp vs MuJoCo on the same tilted plane
  for B in ${BODY:-.work/assets/fly/body.xml .work/spider-body/spider.xml}; do
    read T P < <($M -c "from kinestride.model import terrain; print(*terrain('$B'))")
    # spider bar 3e-5, measured (main 6e6d0e9, deterministic over reruns): qfrc_constraint 1.19e-5 = 6.2 x its float32 floor 1.92e-6
    # (the open spider qfrc gap, also on the flat floor), qLD 9.3e-6, qacc 5.4e-6, the rest below 6e-6. Fly stays at the 1e-5 default.
    case $B in *spider*) rel=3e-5;; *) rel=${REL:-1e-5};; esac
    ORACLE=$P REL=$rel $M -m kinestride.model check $T $X/terrain-$(basename $B .xml) actuated 64 || { echo "FAIL terrain $(basename $B .xml)"; fail=$((fail+1)); }
  done
  # the gate must fail with the slope's normal tilted the wrong way, on each body at its own bar
  for B in .work/assets/fly/body.xml .work/spider-body/spider.xml; do
    read T P < <($M -c "from kinestride.model import terrain; print(*terrain('$B'))")
    case $B in *spider*) rel=3e-5;; *) rel=1e-5;; esac
    ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/normal = V3(-hg\[1\], -hg\[2\], 1, 0)/normal = V3(hg[1], hg[2], 1, 0)/' -- \
      env ORACLE=$P REL=$rel $M -m kinestride.model check $T $X/mutant-$(basename $B .xml) actuated 64 || fail=$((fail+1))
  done
  # MEASURED, not a gate: tangent plane vs MuJoCo's own mjc_ConvexHField on bumpy terrain (0.3 mm, 8 mm wavelength,
  # 4 octaves; heightfield sampled every 0.31 mm). The tangent plane is not that algorithm; the stand gate is live check
  N=$($M -c "from kinestride.model import terrain; print(terrain('.work/assets/fly/body.xml', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
  $M -m kinestride.model check $N $X/terrain-noise actuated 64 2>&1 | grep -E "max rel err|ncon" | sed 's/^\(PASS\|FAIL\)/MEASURED/' || true
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all terrain checks"; exit 0
fi
if [ "$stage" = groom ]; then  # the 7 sphere-sphere grooming pairs (front-leg rubs on head, antennae, each other)
  $M kinestride/groom.py $X/groom.npz || { echo "FAIL groom states"; exit 1; }
  STATES=$X/groom.npz $M -m kinestride.model check .work/assets/fly/body.xml $X/groom actuated 64 || { echo "FAIL groom"; fail=$((fail+1)); }
  # the gate must fail with the sphere-sphere normal reversed
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/normal = dif \/ len/normal = -dif \/ len/' -- \
    env STATES=$X/groom.npz $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant actuated 64 || fail=$((fail+1))
  # and with noslip skipped (the fly ships noslip 5; MuJoCo runs it)
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/^        noslip(mi, mf, res, f)$/        pass/' -- \
    env STATES=$X/groom.npz $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant actuated 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all groom checks"; exit 0
fi
if [ "$stage" = equality ] || [ "$stage" = equality_contact ]; then  # connect, weld, joint equalities (models/eq_test.xml: random
  # states violate them; Jdot v needs velocities) and, for equality_contact, with an active sphere-sphere contact and noslip 5
  # in the same Newton solve (mixed H rows); both vs mj_step, float32-floor bars as the contact stage
  E=kinestride/models/eq_test.xml; [ "$stage" = equality_contact ] && E=kinestride/models/eq_contact.xml
  $M -m kinestride.model check $E $X/$stage equality 64 || { echo "FAIL $stage"; fail=$((fail+1)); }
  S=kinestride/smooth.mojo
  # the gate must fail with: the anchor sign flipped, the weld rotation conjugate dropped, the joint polynomial sign flipped,
  # the Jdot v correction sign flipped, the reference acceleration dropped
  for mut in 's/var p0 = rot(B0\[1\], m3(mf, ef))/var p0 = rot(B0[1], -m3(mf, ef))/' \
             's/var nq1 = Q4(qb\[0\], -qb\[1\], -qb\[2\], -qb\[3\])/var nq1 = Q4(qb[0], qb[1], qb[2], qb[3])/' \
             's/cpos\[0\] -= mf\[ef + 25\] \* dif/cpos[0] += mf[ef + 25] * dif/' \
             's/jdv\[k\] = jd0\[k\] - jd1\[k\]/jdv[k] = jd0[k] + jd1[k]/' \
             's/ - get(res, f, OEQA + rbase + k)/ - 0 * get(res, f, OEQA + rbase + k)/'; do
    ~/iTools/bin/mutant-check $S "$mut" -- $M -m kinestride.model check $E $X/mutant equality 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all $stage checks"; exit 0
fi
if [ "$stage" = cross ]; then  # coupling across two trees in one model (sphere and capsule pairs, connect/weld/joint
  # equalities, spatial + fixed tendons and a tendon limit) vs mj_step, at two placements
  W0=$X/cross/w0.xml
  for dx in 0 2; do
    WX=$X/cross/w$dx.xml
    $M -c "from kinestride.model import shifted; print(shifted('kinestride/models/cross_world.xml', '$WX', $dx))" || { echo "FAIL cross shift $dx"; fail=$((fail+1)); }
    CONVEX=1 STONES=1 NEAR_STATES=1 CROSS_MIN=16 $M -m kinestride.model check $WX $X/cross/e$dx equality 64 || { echo "FAIL cross equality $dx"; fail=$((fail+1)); }
    $M -m kinestride.model check $WX $X/cross/t$dx tendon_limit 64 || { echo "FAIL cross tendon $dx"; fail=$((fail+1)); }
  done
  S=kinestride/smooth.mojo
  # the gate must fail with: geom1's side using geom2's root com; the cross-tree shift dropped in the sphere and convex
  # pairs; the frame shift dropped in equality's sh1; the tree shift dropped in tjac and tpoint
  for mut in \
    's/var off1 = cpos - com1/var off1 = cpos - com/' \
    's/^                pos1 = pos1 + (origin(mi, qpos, f, Int(mi\[GBODY + g1\])) - origin(mi, qpos, f, b))/                pos1 = pos1/' \
    's/^        pos1 = pos1 + (origin(mi, qpos, f, Int(mi\[GBODY + g1\])) - origin(mi, qpos, f, b))/        pos1 = pos1/' \
    's/var sh1 = o1 - frame/var sh1 = V3(0)/'; do
    CONVEX=1 STONES=1 NEAR_STATES=1 CROSS_MIN=16 ~/iTools/bin/mutant-check $S "$mut" -- $M -m kinestride.model check $W0 $X/cross/mutant equality 64 || fail=$((fail+1))
  done
  for mut in \
    's/0) + (origin(mi, qpos, f, bi) - frame)/0)/' \
    's/b\[0\] + (origin(mi, qpos, f, bi) - frame)/b[0]/'; do
    ~/iTools/bin/mutant-check $S "$mut" -- $M -m kinestride.model check $W0 $X/cross/mutant tendon_limit 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail cross checks"; exit 1; }
  echo "PASS all cross checks"; exit 0
fi
if [ "$stage" = xfrc ]; then  # MuJoCo xfrc_applied (mj_xfrcAccumulate): 4 random sparse wrench slots per state (ctrl.bin rows after
  # the mocap rows, -DNXF) vs mj_step with d.xfrc_applied: fly actuated (contacts, servos), spider step, a push across trees
  F=.work/assets/fly/body.xml; W=kinestride/models/cross_world.xml
  XFRC=4 $M -m kinestride.model check $F $X/xfrc/fly actuated 64 || { echo "FAIL xfrc fly"; fail=$((fail+1)); }
  XFRC=4 $M -m kinestride.model check kinestride/models/spider_test.xml $X/xfrc/spider step 64 || { echo "FAIL xfrc spider"; fail=$((fail+1)); }
  XFRC=4 CONVEX=1 STONES=1 NEAR_STATES=1 $M -m kinestride.model check $W $X/xfrc/cross equality 64 || { echo "FAIL xfrc cross"; fail=$((fail+1)); }
  S=kinestride/smooth.mojo
  # the gate must fail with: the push not applied (today's behaviour), the force sign flipped, the point term
  # (xipos - subtree_com) x force sign flipped, the point at xpos instead of xipos
  for mut in \
    's/^            xfrc(mi, ctrl, res, f)$/            pass/' \
    's/var fo = V3(ctrl\[(row + 1)/var fo = -V3(ctrl[(row + 1)/' \
    's/+ cross(off, fo)$/- cross(off, fo)/' \
    '/var off = V3(get(res, f, OXI/s/OXI + 3 \* (b - 1)/3 * (b - 1)/g'; do
    ~/iTools/bin/mutant-check $S "$mut" -- env XFRC=4 $M -m kinestride.model check kinestride/models/spider_test.xml $X/xfrc/mutant step 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail xfrc checks"; exit 1; }
  echo "PASS all xfrc checks"; exit 0
fi
if [ "$stage" = mocap ]; then  # two mocap bodies (random mocap_pos/quat per state, quats unnormalised) welded/connected to the animal
  E=kinestride/models/eq_mocap.xml
  $M -m kinestride.model check $E $X/mocap mocap 64 || { echo "FAIL mocap"; fail=$((fail+1)); }
  # the gate must fail with every mocap body reading the first one's pose, and with the quaternion components swapped
  for mut in 's/var mrow = NU + 7 \* mid/var mrow = NU + 7 * 0/' \
             's/xquat = Q4(ctrl\[(mrow + 3) \* F + f\], ctrl\[(mrow + 4) \* F + f\]/xquat = Q4(ctrl[(mrow + 4) * F + f], ctrl[(mrow + 3) * F + f]/'; do
    ~/iTools/bin/mutant-check kinestride/smooth.mojo "$mut" -- $M -m kinestride.model check $E $X/mutant mocap 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all mocap checks"; exit 0
fi
if [ "$stage" = gravcomp ]; then  # body gravcomp (mj_gravcomp) on the fly and the spider test body (body_gravcomp 1, .5, 0, .8 in turn)
  # and on models/gravcomp_act.xml, where joints with actgravcomp take the actuator route (engine_forward.c:770; needs nu > 0)
  S=kinestride/smooth.mojo; G=kinestride/models/gravcomp_act.xml; mkdir -p $X/gc
  FG=$($M -c "from kinestride.model import with_gravcomp; print(with_gravcomp('${BODY:-.work/assets/fly/body.xml}', '$X/gc/fly.xml'))")
  SG=$($M -c "from kinestride.model import with_gravcomp; print(with_gravcomp('kinestride/models/spider_test.xml', '$X/gc/spider.xml'))")
  for B in $FG $SG $G; do for st in rne step; do
    $M -m kinestride.model check $B $X/gc/$(basename $B .xml)-$st $st 64 || { echo "FAIL gravcomp $(basename $B .xml) $st"; fail=$((fail+1)); }
  done; done
  $M -m kinestride.model check $G $X/gc/act actuated 64 || { echo "FAIL gravcomp actuator route"; fail=$((fail+1)); }
  # the gate must fail with gravcomp skipped, with its sign flipped, with the actuator route dropped and with actgravcomp ignored
  ~/iTools/bin/mutant-check $S 's/^        gravcomp(mi, mf, res, f)$/        pass/' -- $M -m kinestride.model check $FG $X/mutant rne 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $S 's/-(mf\[MASS + i\] \* gc)/(mf[MASS + i] * gc)/' -- $M -m kinestride.model check $FG $X/mutant rne 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $S 's/res\[(OQACT + d) \* F + f\] += get(res, f, OGC + d)/res[(OQACT + d) * F + f] += 0 * get(res, f, OGC + d)/' -- $M -m kinestride.model check $G $X/mutant actuated 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $S '0,/if mi\[CGC + j\] != 0:/s//if mi[CGC + j] != 9:/' -- $M -m kinestride.model check $G $X/mutant rne 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all gravcomp checks"; exit 0
fi
if [ "$stage" = sptendon ]; then  # spatial tendons (mj_tendon, mju_wrap): site-site, pulleys, sphere and cylinder wraps with sidesites (inside the
  # sphere too: wrap_inside), a fixed tendon beside them, a tendon servo; fixture models/tendon_spatial.xml (random states, then the actuated path
  # on rollout states) and the fly and spider with three spatial tendons added (model.with_spatial_tendons), vs mj_step on the tendon outputs
  # (ten_length, ten_velocity, dense ten_J) and everything the tendon spring and damper reach
  S=kinestride/smooth.mojo; T=kinestride/models/tendon_spatial.xml; mkdir -p $X/st
  FT=$($M -c "from kinestride.model import with_spatial_tendons; print(with_spatial_tendons('${BODY:-.work/assets/fly/body.xml}', '$X/st/fly.xml'))")
  ST=$($M -c "from kinestride.model import with_spatial_tendons; print(with_spatial_tendons('kinestride/models/spider_test.xml', '$X/st/spider.xml'))")
  for B in $T $FT $ST; do
    $M -m kinestride.model check $B $X/st/$(basename $B .xml) sptendon 64 || { echo "FAIL sptendon $(basename $B .xml)"; fail=$((fail+1)); }
  done
  $M -m kinestride.model check $T $X/st/act sptendon_act 64 || { echo "FAIL sptendon_act"; fail=$((fail+1)); }
  # the gate must fail with: the wrap side flipped, the tangent solutions swapped, the moment arm sign flipped, the pulley divisor dropped,
  # the cylinder's height correction dropped, the inside wrap's angle sign flipped
  for mut in 's/good = ux \* side\[0\] + uy \* side\[1\]/good = -(ux * side[0] + uy * side[1])/' \
             's/var sgn: Float32 = 1 if i == 0 else -1/var sgn: Float32 = -1 if i == 0 else 1/' \
             's/tjac(mi, qpos, res, f, row, bb, pb, dif, 1 \/ divisor, frame)/tjac(mi, qpos, res, f, row, bb, pb, dif, -1 \/ divisor, frame)/' \
             's/divisor = mf\[TNF + w0\]/divisor = 1/' \
             's/r1\[2\] = p0\[2\] + (p1\[2\] - p0\[2\]) \* (L0 + wlen) \/ (L0 + wlen + L1)/r1[2] = p0[2]/' \
             's/ang = asin64(z) - asin64(A \* z)/ang = asin64(z) + asin64(A * z)/'; do
    ~/iTools/bin/mutant-check $S "$mut" -- $M -m kinestride.model check $T $X/mutant sptendon 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all sptendon checks"; exit 0
fi
if [ "$stage" = tendon_limit ]; then  # tendon limit rows (mj_instantiateLimit, engine_core_constraint.c:1277) on the equality block, unilateral in Newton:
  # models/tendon_spatial.xml with every tendon limited to the 35..65 % quantiles of its length over nearby states (model.with_tendon_limits:
  # 4.8 limit rows per state on average, 63/64 states carry one), twice: MuJoCo's default solref_lim 0.02 (stiff rows: float32 leaves
  # qfrc_constraint within 2.4e-4 of its max on the CPU and 3.9e-4 on the card, qacc 1.3e-5, bar REL=5e-4; tolerance makes no difference, solref does) and solref_lim 0.1
  # with tolerance 0 (default bar); the fly and the spider with three spatial tendons limited the same way (soft) at REL=2e-3: the fly's
  # qacc reaches 1e7 on its light rotational dofs, so float32 resolves J qacc in a row's jar to ~1 rad/s^2, 1e-3 of a 5 N limit force
  # (measured: fly qfrc_constraint 1.37e-3, qacc 1.42e-5, a 1e-6 input perturbation moves MuJoCo by 3e-7, so not conditioning; spider ten_J
  # 1.15e-5); vs mj_step with contacts and actuation off, as the equality
  # stage (its qacc_euler bar). Precision is gated on the fixtures, coverage on the animals; mutants on the soft fixture.
  S=kinestride/smooth.mojo; mkdir -p $X/tl
  TL=$($M -c "from kinestride.model import with_tendon_limits; print(with_tendon_limits('kinestride/models/tendon_spatial.xml', '$X/tl/stiff.xml'))")
  TS=$($M -c "from kinestride.model import with_tendon_limits; print(with_tendon_limits('kinestride/models/tendon_spatial.xml', '$X/tl/soft.xml', solref=0.1, tolerance=0))")
  FT=$($M -c "from kinestride.model import with_spatial_tendons, with_tendon_limits; print(with_tendon_limits(with_spatial_tendons('${BODY:-.work/assets/fly/body.xml}', '$X/tl/fly0.xml'), '$X/tl/fly.xml', solref=0.1, tolerance=0))")
  ST=$($M -c "from kinestride.model import with_spatial_tendons, with_tendon_limits; print(with_tendon_limits(with_spatial_tendons('kinestride/models/spider_test.xml', '$X/tl/spider0.xml'), '$X/tl/spider.xml', solref=0.1, tolerance=0))")
  REL=5e-4 $M -m kinestride.model check $TL $X/tl/stiff tendon_limit 64 || { echo "FAIL tendon_limit stiff"; fail=$((fail+1)); }
  $M -m kinestride.model check $TS $X/tl/soft tendon_limit 64 || { echo "FAIL tendon_limit soft"; fail=$((fail+1)); }
  for B in $FT $ST; do
    REL=2e-3 $M -m kinestride.model check $B $X/tl/$(basename $B .xml) tendon_limit 64 || { echo "FAIL tendon_limit $(basename $B .xml)"; fail=$((fail+1)); }
  done
  # the gate must fail with: the limit side flipped, the margin dropped, the row made bilateral (force while jar >= 0), the Jacobian sign dropped
  for mut in 's/var dist = side \* (mf\[ef + 24\] - L)/var dist = -side * (mf[ef + 24] - L)/' \
             's/cpos\[0\] = dist - mf\[ef + 25\]/cpos[0] = dist/' \
             's/if rbase + k >= NEQB and jar >= 0:/if False:/' \
             's/= -side \* ten_j(mi, mf, res, f, t, Int(mi\[CEQC + choff + c\])) if on/= side * ten_j(mi, mf, res, f, t, Int(mi[CEQC + choff + c])) if on/'; do
    ~/iTools/bin/mutant-check $S "$mut" -- $M -m kinestride.model check $TS $X/mutant tendon_limit 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all tendon_limit checks"; exit 0
fi
if [ "$stage" = joint_limit ]; then  # hinge/slide joint limit rows (mj_instantiateLimit) on the equality block, unilateral in Newton: the fly and
  # the spider with every hinge/slide limited to the 35..65 % quantiles of its qpos over nearby states (model.with_joint_limits), contacts
  # off, solref_limit 0.1 and tolerance 0, vs mj_step (REL=2e-3 as tendon_limit's animals: the fly's light rotational dofs). The shipped fly's
  # own two proboscis limits now export through the same rows.
  S=kinestride/smooth.mojo; mkdir -p $X/jl
  FJ=$($M -c "from kinestride.model import with_joint_limits; print(with_joint_limits('${BODY:-.work/assets/fly/body.xml}', '$X/jl/fly.xml', solref=0.1, tolerance=0))")
  SJ=$($M -c "from kinestride.model import with_joint_limits; print(with_joint_limits('kinestride/models/spider_test.xml', '$X/jl/spider.xml', solref=0.1, tolerance=0))")
  for B in $FJ $SJ; do
    REL=2e-3 $M -m kinestride.model check $B $X/jl/$(basename $B .xml) joint_limit 64 || { echo "FAIL joint_limit $(basename $B .xml)"; fail=$((fail+1)); }
  done
  # the gate must fail with: the limit side flipped, the margin dropped, the Jacobian sign flipped
  for mut in 's/var dist = side \* (mf\[ef + 24\] - qpos/var dist = -side * (mf[ef + 24] - qpos/' \
             's/res\[(jb + c) \* F + f\] = -side if on and/res[(jb + c) * F + f] = side if on and/'; do
    REL=2e-3 ~/iTools/bin/mutant-check $S "$mut" -- $M -m kinestride.model check $SJ $X/mutant joint_limit 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all joint_limit checks"; exit 0
fi
if [ "$stage" = flight_contacts ]; then  # the Flybody flight model's own contacts: 842 automatic capsule/ellipsoid/cylinder/sphere body-body pairs
  # (model.with_auto_pairs: the same pairs and mixed parameters as explicit pairs; tools/auto_pairs_check.py proves them against MuJoCo's dynamic
  # pairs; up to 5 contacts per capsule/cylinder pair from mjc_Convex's perturbation search, two for parallel capsules), condim 1 (frictionless rows), noslip 5, pyramidal. Narrowphase
  # (convex_check, float64): capsule-capsule, sphere-capsule, sphere-cylinder analytic (exact), the rest GJK + EPA with the sphere/capsule core
  # trick (dist 1e-9, normal 1.5e-4: both sides stop at the 1e-6 ccd tolerance; cases with a centre inside the other geom are left out, EPA's
  # answer there is tolerance-arbitrary on both sides), and every contact MuJoCo reports on 64 near-rest states. Engine vs mj_step: random
  # airborne states (RANDOM_STATES=1, qvel up to 100, ~26 contacts each, default bar) and near-rest states (NEAR_STATES=1, the model's own
  # resting overlaps, ~10 contacts each, bar 3e-3: six states flip the active set of a contact carrying < 0.3 % of the max force, median 9e-7;
  # qacc_euler 2.3e-3 on the card). Both draws reject states with a geom centre inside the other geom (model.reject_degenerate: EPA is
  # tolerance-arbitrary there, so MuJoCo, the CPU build and the card disagree at 1e-2).
  FL=${FLIGHT:-.work/assets/flight}; S=kinestride/smooth.mojo; C=kinestride/convex.mojo; mkdir -p $X/fc
  [ -f $FL/flight.xml ] || ${FLIGHT_PY:?FAIL flight body missing: run tools/assets.sh} tools/flight_export.py $FL
  PYTHONPATH=. $M tools/auto_pairs_check.py $FL/flight.xml $X/fc/flight_pairs.xml 64 || { echo "FAIL auto_pairs oracle"; fail=$((fail+1)); }
  FC=$X/fc/flight_pairs.xml
  for kind in capsule-cylinder capsule-ellipsoid ellipsoid-ellipsoid ellipsoid-cylinder sphere-ellipsoid capsule-capsule sphere-capsule sphere-cylinder; do
    FLIGHT_PAIRS=$FC $M -m kinestride.convex_check $kind 1500 f64 1e-9 1 || { echo "FAIL narrowphase $kind"; fail=$((fail+1)); }
  done
  FLIGHT_PAIRS=$FC $M -m kinestride.convex_check flight 64 f64 || { echo "FAIL narrowphase on flight poses"; fail=$((fail+1)); }
  CONVEX=1 RANDOM_STATES=1 VSCALE=${VSCALE:-100} $M -m kinestride.model check $FC $X/fc/random actuated 64 || { echo "FAIL flight contacts random states"; fail=$((fail+1)); }
  CONVEX=1 NEAR_STATES=1 REL=${FC_NEAR_REL:-3e-3} $M -m kinestride.model check $FC $X/fc/near actuated 64 || { echo "FAIL flight contacts near states"; fail=$((fail+1)); }
  # the gate must fail with: the capsule support's end flipped (engine, random states), the frictionless normal row given the pyramid's row (engine),
  # the row threshold moved to margin + gap (the pre-3.9 convention; near states, one contact per state goes missing), the sphere-cylinder
  # cap normal unflipped (narrowphase), the sphere-capsule clamp dropped (narrowphase)
  ~/iTools/bin/mutant-check $C 's/var end = V(0, 0, self.size\[1\] if ld\[2\] >= 0 else -self.size\[1\], 0)/var end = V(0, 0, -self.size[1] if ld[2] >= 0 else self.size[1], 0)/' -- \
    env CONVEX=1 RANDOM_STATES=1 VSCALE=${VSCALE:-100} $M -m kinestride.model check $FC $X/mutant actuated 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $S 's/            res\[jadr \* F + f\] = jn$/            res[jadr * F + f] = jn + f0 * j1/' -- \
    env CONVEX=1 RANDOM_STATES=1 VSCALE=${VSCALE:-100} $M -m kinestride.model check $FC $X/mutant actuated 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $S 's/    if dist >= imargin:  # mj_setContact/    if dist >= imargin + gap:  # mj_setContact/' -- \
    env CONVEX=1 NEAR_STATES=1 REL=${FC_NEAR_REL:-3e-3} $M -m kinestride.model check $FC $X/mutant actuated 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $C 's/cs.normal\[0\] = (-n)/cs.normal[0] = n/' -- \
    env FLIGHT_PAIRS=$FC $M -m kinestride.convex_check sphere-cylinder 600 f64 1e-9 3 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $C 's/var x = min(sz2\[1\], max(-sz2\[1\], dot(ax2, vec)))/var x = dot(ax2, vec)/' -- \
    env FLIGHT_PAIRS=$FC $M -m kinestride.convex_check sphere-capsule 600 f64 1e-9 3 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all flight_contacts checks"; exit 0
fi
if [ "$stage" = sleep ]; then  # sleep (mj_sleep in mj_advance, mj_wake; mjENBL_SLEEP): the fly with sleeping enabled at a tolerance that puts
  # about half of the standing states to sleep within K=20 steps (servos held at the current angle, HOLD=1; MuJoCo zeroes qvel and freezes qpos
  # at the mjMINAWAKE-th quiet check), vs 20 chained mj_step: the asleep sets must agree exactly, the awake states within the K-step bar
  F=.work/assets/fly/body.xml; S=kinestride/smooth.mojo
  SL=$($M -c "from kinestride.model import with_sleep; print(with_sleep('$F', '$X/sleep/fly.xml', ${SLEEP_TOL:-5.0}))")
  HOLD=1 REL=${SLEEP_REL:-1e-3} $M -m kinestride.model check $SL $X/sleep/fly actuated 64 20 || { echo "FAIL sleep fly K=20"; fail=$((fail+1)); }
  # the gate must fail with: the dof_length weight dropped, the quiet-check count off by one (the sleepers' qpos bar), the velocity not zeroed.
  # The asleep skip has no observable here (a stepped sleeper re-zeroes itself: same state, more work), so it carries no mutant.
  for mut in 's/quiet = quiet and (mf\[SLPF + 1 + v\] \* abs(qvel\[v \* F + f\])/quiet = quiet and (abs(qvel[v * F + f])/' \
             's/if c >= 10:/if c >= 9:/' \
             's/qvel\[v \* F + f\] = 0  # asleep/qvel[v * F + f] *= 1  # asleep/'; do
    ~/iTools/bin/mutant-check $S "$mut" -- env HOLD=1 REL=${SLEEP_REL:-1e-3} $M -m kinestride.model check $SL $X/mutant actuated 64 20 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all sleep checks"; exit 0
fi
if [ "$stage" = loop ]; then  # K steps per launch, state kept on the card. K=1 vs mj_step (one step: the accuracy
  # gate); one K=10 launch must equal ten K=1 launches byte for byte (the loop). K=10 vs ten chained mj_step is not a
  # gate: the engine's ~5e-6 per-step float32 error grows ~12x in 10 steps on contact-rich states (diag/k_floor.py)
  F=.work/assets/fly/body.xml
  # one step is not a chained trajectory: the chaotic-state filter (for K>1 at wide bars) flags 52/64 states at this 1e-5 bar
  CHAOS_SEEDS=0 $M -m kinestride.model check $F $X/fly actuated 64 1 || { echo "FAIL fly loop K=1"; fail=$((fail+1)); }
  ~/iTools/bin/mojo-cached build $(cat $X/fly/sizes.txt) -DFLIES=64 -DSTAGE=7 -DK=10 -DWAVE=${WAVE:-0} kinestride/smooth.mojo -o $X/fly/k10
  cp $X/fly/state10.bin $X/fly/k1x10.bin && $X/fly/k10 $X/fly/
  cmp $X/fly/state.bin $X/fly/k1x10.bin && echo "PASS K=10 launch == 10 K=1 launches, byte-equal" || { echo "FAIL loop K=10 vs 10 x K=1"; fail=$((fail+1)); }
  # the gate must fail with the velocity write-back dropped (each step restarts from the launch's qvel)
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/qvel\[v \* F + f\] += mf\[TSTEP\]/qvel[v * F + f] += 0 * mf[TSTEP]/' -- \
    env CHAOS_SEEDS=0 $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant actuated 64 1 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all loop checks"; exit 0
fi
if [ "$stage" = fluid ]; then  # Flybody flight model: mj_fluid (ellipsoid wings + inertia boxes) inside qfrc_passive, wing velocities up to flapping rate
  FL=${FLIGHT:-.work/assets/flight}
  [ -f $FL/flight.xml ] || ${FLIGHT_PY:?FAIL flight body missing: run tools/assets.sh} tools/flight_export.py $FL
  VSCALE=${VSCALE:-100} $M -m kinestride.model check $FL/flight.xml $X/flight rne 64 || { echo "FAIL flight fluid"; fail=$((fail+1)); }
  # each mutant must fail: Kutta lift sign, Magnus sign, added-mass sign, inertia-box drag sign
  for mut in 's/    fo += cross(circ, v)/    fo -= cross(circ, v)/' 's/    fo += cross(w, v) \* (magnus/    fo -= cross(w, v) * (magnus/' \
             's/    var fo = cross(lin_mom, w)/    var fo = -cross(lin_mom, w)/' 's/        fr\[3\] -= 0.5 \* rho/        fr[3] += 0.5 * rho/'; do
    ~/iTools/bin/mutant-check kinestride/smooth.mojo "$mut" -- \
      env VSCALE=${VSCALE:-100} $M -m kinestride.model check $FL/flight.xml $X/mutant rne 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all fluid checks"; exit 0
fi
if [ "$stage" = loopk ]; then  # engine vs ITSELF, no MuJoCo: one K=10 launch == ten K=1 launches byte for byte, on BODY (default fly)
  # with ENGINE_OPT options (integrator, solver). One-step MuJoCo parity is the separate per-option stages (integrator, loop)
  B=${BODY:-.work/assets/fly/body.xml}; D=$X/loopk-$(basename $B .xml)
  ST=actuated; [ "$(basename $B)" = spider_test.xml ] && ST=step  # the spider test body has no actuators
  ENGINE_ONLY=1 $M -m kinestride.model check $B $D $ST 64 1 | tail -1
  ~/iTools/bin/mojo-cached build $(cat $D/sizes.txt) -DFLIES=64 -DSTAGE=$($M -c "from kinestride.model import STAGES; print(STAGES['$ST'][0])") -DK=10 -DWAVE=${WAVE:-0} kinestride/smooth.mojo -o $D/k10
  cp $D/state10.bin $D/k1x10.bin && $D/k10 $D/ >/dev/null
  cmp $D/state.bin $D/k1x10.bin && echo "PASS loopk $(basename $B .xml) ${ENGINE_OPT:-defaults}: K=10 launch == 10 K=1 launches, byte-equal (engine vs engine, not MuJoCo)" || { echo "FAIL loopk $(basename $B .xml)"; exit 1; }
  exit 0
fi
if [ "$stage" = integrator ]; then  # implicitfast (mj_implicitSkip) vs MuJoCo with opt.integrator=implicitfast, fly + spider + 2-animal world
  F=.work/assets/fly/body.xml; SP=kinestride/models/spider_test.xml
  SB=.work/spider-body/spider.xml; I=$X/integrator; KV=0.01  # KV: the shipped servos have kv 0 = Euler's qH; 0.01 makes h gear^2 kv ~ M's diagonal
  echo "COVERAGE integrator=implicitfast (3): fly step + actuated (airborne states, kv=$KV, shipped kv=0 in step), spider_test step, world fly+spider at 8 and 200 mm"
  echo "COVERAGE not covered: standalone free body (midpoint, refused at export), implicit (full), RK4, tendon-damping and fluid derivatives (refused), spring-only tendons (accepted, no velocity derivative, no parity case), contact states (MEASURED only)"
  # fluid derivatives (mjd_ellipsoidFluid, mjd_inertiaBoxFluid blocks of qDeriv) on the Flybody flight model, step fields under both integrators;
  # mutants: the ellipsoid viscous-drag diagonal sign, the inertia-box linear drag sign (both leave the forward forces alone)
  FL=${FLIGHT:-.work/assets/flight}
  for integ in 2 3; do
    VSCALE=${VSCALE:-100} ENGINE_OPT=integrator=$integ,cone=0 $M -m kinestride.model check $FL/flight.xml $I/fluid$integ step 64 || { echo "FAIL fluid derivative integrator=$integ"; fail=$((fail+1)); }
  done
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/    e0\[0\] -= lincoef; e1\[1\] -= lincoef; e2\[2\] -= lincoef/    e0[0] += lincoef; e1[1] += lincoef; e2[2] += lincoef/' -- \
    env VSCALE=${VSCALE:-100} ENGINE_OPT=integrator=2,cone=0 $M -m kinestride.model check $FL/flight.xml $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/            B\[(3 + k) \* 7\] = -3 \* pi \* diam \* mu/            B[(3 + k) * 7] = 3 * pi * diam * mu/' -- \
    env VSCALE=${VSCALE:-100} ENGINE_OPT=integrator=3,cone=0 $M -m kinestride.model check $FL/flight.xml $I/mutant step 64 || fail=$((fail+1))
  # tendon damping under both implicit integrators (mjd_passive_vel's tendon term, the J' damping J block of qDeriv on D's / M's pattern): models/tendon_spatial.xml
  # (five spatial + one fixed tendon, every one damped) at random states; mutant: the block's sign
  T=kinestride/models/tendon_spatial.xml
  for integ in 2 3; do
    ENGINE_OPT=integrator=$integ $M -m kinestride.model check $T $I/tendon$integ sptendon 64 || { echo "FAIL tendon damping implicit integrator=$integ"; fail=$((fail+1)); }
  done
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/res\[(OQLU + e) \* F + f\] += h \* damp \* jk/res[(OQLU + e) * F + f] -= h * damp * jk/' -- \
    env ENGINE_OPT=integrator=2 $M -m kinestride.model check $T $I/mutant sptendon 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/res\[(OQH + e) \* F + f\] += h \* damp \* jk/res[(OQH + e) * F + f] -= h * damp * jk/' -- \
    env ENGINE_OPT=integrator=3 $M -m kinestride.model check $T $I/mutant sptendon 64 || fail=$((fail+1))
  export ENGINE_OPT=integrator=3,kv=$KV
  A=$($M -m kinestride.model airborne $F $I/fly | tail -1)
  STATES=$A $M -m kinestride.model check $F $I/fly actuated 64 || { echo "FAIL fly actuated implicitfast"; fail=$((fail+1)); }
  ENGINE_OPT=integrator=3 $M -m kinestride.model check $F $I/fly-step step 64 || { echo "FAIL fly step implicitfast"; fail=$((fail+1)); }
  ENGINE_OPT=integrator=3 $M -m kinestride.model check $SP $I/spider step 64 || { echo "FAIL spider step implicitfast"; fail=$((fail+1)); }
  # the gate must fail with the actuator velocity term of qH (qDeriv) added instead of subtracted
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/-= h \* mf\[af + 8\] \* mf\[af + 8\] \* mf\[af + 3\]/+= h * mf[af + 8] * mf[af + 8] * mf[af + 3]/' -- \
    env STATES=$A $M -m kinestride.model check $F $I/mutant actuated 64 || fail=$((fail+1))
  # islands: two animals in one model. Each animal's dofs get their own qH terms; the engine must match MuJoCo on the merged model
  # at 200 mm (two islands) and 8 mm (the spider's origin 8 mm away), and fail when the spider uses the fly's origin
  for off in 200 8; do
    W=$($M -c "from kinestride.model import combine; print(combine('$F', '$SB', ($off, 0, 0), 'iworld$off'))")
    AW=$($M -m kinestride.model airborne $W $I/world$off | tail -1)
    STATES=$AW $M -m kinestride.model check $W $I/world$off actuated 64 || { echo "FAIL implicitfast world at $off mm"; fail=$((fail+1)); }
    if [ $off = 200 ]; then
      ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/    var r = Int(mi\[ROOT + b\])/    var r = Int(mi[ROOT + 1])/' -- \
        env STATES=$AW $M -m kinestride.model check $W $I/mutant actuated 64 || fail=$((fail+1))
    fi
  done
  # MEASURED, not a gate: contact states (active-set float32 noise, fly kv 0.01: qacc_euler 2.4e-5, qfrc_constraint 1.2e-4 vs a floor-lifted bar of 1.3e-5, 6.4e-5)
  $M -m kinestride.model check $F $I/fly-contact actuated 64 2>&1 | grep -E "max rel err" | grep -E "qacc|qfrc_constraint" | sed 's/^\(PASS\|FAIL\)/MEASURED/' || true
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all integrator checks (implicitfast)"; exit 0
fi
if [ "$stage" = implicit ]; then  # full implicit (mj_implicit: qDeriv with mjd_rne_vel, LU) and the implicitfast midpoint of standalone free bodies, vs MuJoCo
  F=.work/assets/fly/body.xml; SP=kinestride/models/spider_test.xml; MP=kinestride/models/midpoint.xml
  SB=.work/spider-body/spider.xml; I=$X/implicit; KV=0.01
  echo "COVERAGE integrator=implicit (2): fly step (shipped kv 0) + actuated (airborne states, kv=$KV), spider_test step (free, ball, hinge, slide), world fly+spider at 8 and 200 mm"
  echo "COVERAGE integrator=implicitfast (3) midpoint: four standalone free bodies (models/midpoint.xml: simple, aligned, coupled, coupled and rotated), contact-free; contact eligibility in the midpoint_contact part"
  echo "COVERAGE tendon damping derivative: models/tendon_spatial.xml step under integrator 2 and 3, sign mutants; tendon actuator velocity term (tkv) under both, sign mutants"
  echo "COVERAGE fluid derivatives: Flybody flight.xml step under integrator 2 and 3, drag-sign mutants"
  echo "COVERAGE not covered: contact states under full implicit (MEASURED only, active-set float32 noise as the implicitfast gate), WAVE=1"
  # tendon actuator velocity feedback (mjd_actuator_vel: gear^2 biasprm[2] J' J, Codex review B6): models/tendon_spatial.xml's servo with tkv
  for integ in 2 3; do
    ENGINE_OPT=integrator=$integ,tkv=0.5 $M -m kinestride.model check kinestride/models/tendon_spatial.xml $I/tact$integ sptendon_act 64 || { echo "FAIL tendon actuator implicit integrator=$integ"; fail=$((fail+1)); }
    ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/\* F + f\] -= h \* g2b \* jk/* F + f] += h * g2b * jk/' -- \
      env ENGINE_OPT=integrator=$integ,tkv=0.5 $M -m kinestride.model check kinestride/models/tendon_spatial.xml $I/mutant sptendon_act 64 || fail=$((fail+1))
  done
  export ENGINE_OPT=integrator=2
  $M -m kinestride.model check $F $I/fly-step step 64 || { echo "FAIL fly step implicit"; fail=$((fail+1)); }
  $M -m kinestride.model check $SP $I/spider step 64 || { echo "FAIL spider step implicit"; fail=$((fail+1)); }
  export ENGINE_OPT=integrator=2,kv=$KV
  A=$($M -m kinestride.model airborne $F $I/fly | tail -1)
  STATES=$A $M -m kinestride.model check $F $I/fly actuated 64 || { echo "FAIL fly actuated implicit"; fail=$((fail+1)); }
  # the gate must fail without the Coriolis derivative (mjd_rne_vel's term of qLU), with the damping term of qLU flipped, and without Dcacc inherited from the parent
  export ENGINE_OPT=integrator=2
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/+= h \* (fr \* cd).reduce_add()/+= 0 * h * (fr * cd).reduce_add()/' -- \
    $M -m kinestride.model check $F $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/+= h \* (fr \* cd).reduce_add()/+= 0 * h * (fr * cd).reduce_add()/' -- \
    $M -m kinestride.model check $SP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/Int(mi\[DDG + v\])) \* F + f\] += h \* mf\[DAMP + v\]/Int(mi[DDG + v])) * F + f] -= h * mf[DAMP + v]/' -- \
    $M -m kinestride.model check $SP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/res\[(ODCA + 6 \* badr + r) \* F + f\] = get(res, f, ODCA + 6 \* padr + r)/res[(ODCA + 6 * badr + r) * F + f] = 0/' -- \
    $M -m kinestride.model check $SP $I/mutant step 64 || fail=$((fail+1))
  # islands: two animals in one model (own LU rows each), at 200 mm (two islands) and 8 mm
  export ENGINE_OPT=integrator=2,kv=$KV
  for off in 200 8; do
    W=$($M -c "from kinestride.model import combine; print(combine('$F', '$SB', ($off, 0, 0), 'imworld$off'))")
    AW=$($M -m kinestride.model airborne $W $I/world$off | tail -1)
    STATES=$AW $M -m kinestride.model check $W $I/world$off actuated 64 || { echo "FAIL implicit world at $off mm"; fail=$((fail+1)); }
  done
  # midpoint (implicitfast, mj_midpoint)
  export ENGINE_OPT=integrator=3
  $M -m kinestride.model check $MP $I/mid step 64 || { echo "FAIL midpoint step"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/if not mid_eligible(mi, res, f, b):/if True:/' -- \
    $M -m kinestride.model check $MP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/var dw_body = rot(iquat, 2 \* delta)/var dw_body = rot(iquat, delta)/' -- \
    $M -m kinestride.model check $MP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/var du = rot(iquat, 2 \* (x - cross(delta, r_com)))/var du = rot(iquat, 2 * x)/' -- \
    $M -m kinestride.model check $MP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/force \/ mass + rot(rot_x2i, m3(mf, GRAV))/force \/ mass/' -- \
    $M -m kinestride.model check $MP $I/mutant step 64 || fail=$((fail+1))
  # midpoint_contact: p and q touch in about half the states (live contact: ordinary implicitfast step), r never does (always midpoint)
  MC=kinestride/models/midpoint_contact.xml
  $M -m kinestride.midpoint_states $I/mc-states.npz 64
  STATES=$I/mc-states.npz $M -m kinestride.model check $MC $I/mc contact 64 || { echo "FAIL midpoint contact"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/if not mid_eligible(mi, res, f, b):/if True:/' -- \
    env STATES=$I/mc-states.npz $M -m kinestride.model check $MC $I/mutant contact 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/or Int(mi\[GBODY + Int(mi\[CPH + 11 \* p + 1\])\]) == b:/or False:/' -- \
    env STATES=$I/mc-states.npz $M -m kinestride.model check $MC $I/mutant contact 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all implicit checks"; exit 0
fi
if [ "$stage" = tendon ]; then  # Flybody flight model, airborne (no contacts) with a spring and damper on its two fixed tendons: full step, actuators incl. trntype 3
  FL=${FLIGHT:-.work/assets/flight}
  [ -f $FL/flight.xml ] || ${FLIGHT_PY:?FAIL flight body missing: run tools/assets.sh} tools/flight_export.py $FL
  A=$($M -c "from kinestride.model import airborne_mjcf; print(airborne_mjcf('$FL/flight.xml'))")
  $M -m kinestride.model check $A $X/airborne actuated 64 || { echo "FAIL flight tendon"; fail=$((fail+1)); }
  # each mutant must fail: tendon length sign, passive coef sign, actuator moment sign, spring dead zone dropped
  for mut in 's/        length += c \* qpos/        length -= c * qpos/' 's/\] += mf\[TNF + w\] \* frc/] -= mf[TNF + w] * frc/' \
             's/+= gear \* mf\[TNF + w\] \* force/-= gear * mf[TNF + w] * force/' 's/var x = lv\[0\] - hi if lv\[0\] > hi else (lv\[0\] - lo if lv\[0\] < lo else Float32(0))/var x = lv[0] - hi/'; do
    ~/iTools/bin/mutant-check kinestride/smooth.mojo "$mut" -- $M -m kinestride.model check $A $X/mutant actuated 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all tendon checks"; exit 0
fi
if [ "$stage" = rk4 ]; then  # RK4 (mj_RungeKutta) vs MuJoCo with opt.integrator=RK4: fly, spider, fly + spider world
  F=.work/assets/fly/body.xml; SP=kinestride/models/spider_test.xml
  SB=.work/spider-body/spider.xml; I=$X/rk4
  echo "COVERAGE integrator=RK4 (1): fly step (contact-free, dt 1e-6: h B / M 0.059, a wrong weight still moves qacc_euler by far more than 1e-5; dt 1e-5 leaves cdof_dot at 1.4e-5 float32 noise), spider_test step; fly actuated on standing states (contacts + servos + adhesion, dt 1e-7); world fly+spider at 8 and 200 mm (dt 1e-7)"
  echo "COVERAGE not covered: contact and servo parity at dt >= 1e-6 (stiff contacts: MuJoCo's own ncon flips between its 4 forward passes and float32 follows chaotically: dt 1e-6 qfrc_constraint 4.9e-5 vs 3.9e-5 bar, dt 1e-5 ncon off), RK4 on the shipped dt 1e-4 (explicit dof damping is unstable in MuJoCo itself: h B / M 5.9), actuator velocity terms, qH (stale in MuJoCo after RK4, not compared)"
  ENGINE_OPT=integrator=1,dt=1e-6 $M -m kinestride.model check $F $I/fly-step step 64 || { echo "FAIL fly step rk4"; fail=$((fail+1)); }
  ENGINE_OPT=integrator=1,dt=1e-6 $M -m kinestride.model check $SP $I/spider step 64 || { echo "FAIL spider step rk4"; fail=$((fail+1)); }
  # the gate must fail with the RK4 weights wrong (F1 weight 2 -> 1)
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/(get(res, f, ORKF + v) + 2 \* get(res, f, ORKF + NV + v)/(get(res, f, ORKF + v) + 1 * get(res, f, ORKF + NV + v)/' -- \
    env ENGINE_OPT=integrator=1,dt=1e-6 $M -m kinestride.model check $F $I/mutant step 64 || fail=$((fail+1))
  export ENGINE_OPT=integrator=1,dt=1e-7
  A=$(env -u ENGINE_OPT $M -m kinestride.model airborne $F $I/fly 0 | tail -1)  # standing states from an Euler rollout (RK4 cannot roll out the shipped model)
  STATES=$A $M -m kinestride.model check $F $I/fly actuated 64 || { echo "FAIL fly actuated rk4"; fail=$((fail+1)); }
  # the gate must fail with the stage-1 velocity step dropped (X_1 qvel = v0 + h A F0 -> v0)
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/qvel\[v \* F + f\] = get(res, f, ORKV + v) + h \* c \* get(res, f, ORKF + s \* NV + v)/qvel[v * F + f] = get(res, f, ORKV + v)/' -- \
    env ENGINE_OPT=integrator=1,dt=1e-6 $M -m kinestride.model check $F $I/mutant step 64 || fail=$((fail+1))
  # islands: two animals in one model, at 200 mm (two islands) and 8 mm; the gate must fail when the spider uses the fly's origin
  for off in 200 8; do
    W=$($M -c "from kinestride.model import combine; print(combine('$F', '$SB', ($off, 0, 0), 'rworld$off'))")
    AW=$(env -u ENGINE_OPT $M -m kinestride.model airborne $W $I/world$off 0 | tail -1)
    STATES=$AW $M -m kinestride.model check $W $I/world$off actuated 64 || { echo "FAIL rk4 world at $off mm"; fail=$((fail+1)); }
    if [ $off = 200 ]; then
      ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/    var r = Int(mi\[ROOT + b\])/    var r = Int(mi[ROOT + 1])/' -- \
        env STATES=$AW $M -m kinestride.model check $W $I/mutant actuated 64 || fail=$((fail+1))
    fi
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all rk4 checks"; exit 0
fi
if [ "$stage" = cone ]; then  # elliptic friction cone on the walking fly: Newton (noslip 0) and Newton + noslip 5 (QCQP over the friction ellipse), MuJoCo run to convergence
  for ns in 0 5; do
    E=$($M -c "from kinestride.model import elliptic; print(elliptic('.work/assets/fly/body.xml', $ns))")
    $M -m kinestride.model check $E $X/elliptic$ns actuated 64 || { echo "FAIL elliptic cone noslip $ns"; fail=$((fail+1)); }
  done
  E0=$($M -c "from kinestride.model import elliptic; print(elliptic('.work/assets/fly/body.xml', 0))")
  E5=$($M -c "from kinestride.model import elliptic; print(elliptic('.work/assets/fly/body.xml', 5))")
  # Newton mutants (noslip 0): cone force tangent sign, Dm without (1 + mu^2), line-search slope sign, top-zone bound doubled
  for mut in 's/force = V3(f0, -f0 \/ T \* U\[1\] \* fr\[1\], -f0 \/ T \* U\[2\] \* fr\[2\], 0)/force = V3(f0, f0 \/ T * U[1] * fr[1], f0 \/ T * U[2] * fr[2], 0)/' \
             's/(mu \* mu \* (1 + mu \* mu))/(mu * mu)/' 's/e0 += Dm \* (N - mu \* T) \* (N1 - mu \* T1)/e0 -= Dm * (N - mu * T) * (N1 - mu * T1)/' \
             's/if N >= mu \* T or (T <= 0 and N >= 0):/if N >= 2 * mu * T or (T <= 0 and N >= 0):/'; do
    ~/iTools/bin/mutant-check kinestride/smooth.mojo "$mut" -- $M -m kinestride.model check $E0 $X/mutant actuated 64 || fail=$((fail+1))
  done
  # noslip mutants (noslip 5): QCQP residual cross term sign (tangent axes share mu, so an axis swap is an equivalent mutant), dual-finish sign, friction update dropped
  for mut in 's/res0 - (a00 \* o1 + a01 \* o2)/res0 - (a00 * o1 - a01 * o2)/' 's/var df = get(res, f, OFRN + row) - get(res, f, OFRC + row)/var df = get(res, f, OFRN + row) + get(res, f, OFRC + row)/' \
             's/res\[(OFRN + r + 1) \* F + f\] = n\[0\]/res[(OFRN + r + 1) * F + f] = o1/'; do
    ~/iTools/bin/mutant-check kinestride/smooth.mojo "$mut" -- $M -m kinestride.model check $E5 $X/mutant actuated 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cone checks"; exit 0
fi
if [ "$stage" = cone_convex ]; then  # elliptic friction cone on convex pairs: the fly in the arena (box walls, ellipsoid stones, CONVEX=1), Newton noslip 0 and 5,
  # MuJoCo run to convergence (model.elliptic), one-step fields on 64 states moved into walls and stones. The contact tail (emit_contact) is shared by every pair kind,
  # so the cone port is the same code; this gate is what lets export accept the pair. Mutant: the cone force tangent sign (the cone gate's first mutant) on convex states.
  for ns in 0 5; do
    E=$($M -c "from kinestride.model import elliptic; print(elliptic('.work/assets/fly/body.xml', $ns))")
    env CONVEX=1 $M -m kinestride.model check $E $X/cone-convex$ns actuated 64 || { echo "FAIL cone_convex noslip $ns"; fail=$((fail+1)); }
  done
  E0=$($M -c "from kinestride.model import elliptic; print(elliptic('.work/assets/fly/body.xml', 0))")
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/force = V3(f0, -f0 \/ T \* U\[1\] \* fr\[1\], -f0 \/ T \* U\[2\] \* fr\[2\], 0)/force = V3(f0, f0 \/ T * U[1] * fr[1], f0 \/ T * U[2] * fr[2], 0)/' -- \
    env CONVEX=1 $M -m kinestride.model check $E0 $X/mutant actuated 32 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cone_convex checks"; exit 0
fi
if [ "$stage" = cone_hfield ]; then  # elliptic friction cone on exact-heightfield pairs (HFEXACT=1): the terrain_exact terrain for the fly and the spider, Newton noslip 0 and 5,
  # MuJoCo run to convergence (model.elliptic on the terrain xml), one-step fields on 64 states moved onto the terrain. Same contact tail as cone_convex. Bar REL=5e-4 (terrain_exact uses 4e-4, float32 terrain function vs MuJoCo table): fly noslip 0 qacc_euler 4.55e-4 on three Newton active-set switch states (59, 13, 30), median 5.9e-6, deterministic over two runs (DIAG=1); the other three runs within 1.2e-4.
  # Mutant: the cone force tangent sign (cone_convex's mutant) on the fly's exact-heightfield states.
  for B in .work/assets/fly/body.xml .work/spider-body/spider.xml; do
    N=$($M -c "from kinestride.model import terrain; print(terrain('$B', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
    for ns in 0 5; do
      E=$($M -c "from kinestride.model import elliptic; print(elliptic('$N', $ns))")
      env HFEXACT=1 REL=5e-4 $M -m kinestride.model check $E $X/cone-hfield-$(basename $B .xml)-$ns actuated 64 || { echo "FAIL cone_hfield $(basename $B .xml) noslip $ns"; fail=$((fail+1)); }
    done
  done
  N=$($M -c "from kinestride.model import terrain; print(terrain('.work/assets/fly/body.xml', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
  E0=$($M -c "from kinestride.model import elliptic; print(elliptic('$N', 0))")
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/force = V3(f0, -f0 \/ T \* U\[1\] \* fr\[1\], -f0 \/ T \* U\[2\] \* fr\[2\], 0)/force = V3(f0, f0 \/ T * U[1] * fr[1], f0 \/ T * U[2] * fr[2], 0)/' -- \
    env HFEXACT=1 REL=5e-4 $M -m kinestride.model check $E0 $X/mutant actuated 32 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cone_hfield checks"; exit 0
fi
if [ "$stage" = solver ]; then  # CG and PGS (mj_solCG, mj_solPGS) vs MuJoCo with opt.solver set, next to Newton (the default)
  F=.work/assets/fly/body.xml; SB=.work/spider-body/spider.xml; I=$X/solver
  echo "COVERAGE solver=CG (1) and PGS (0), integrator Euler, pyramidal cone, noslip 5 as shipped: fly actuated on standing states (feet, grooming pairs, servos, adhesion), fly + spider world at 200 and 8 mm, two-fly world at 200 mm (two kinematic trees = two MuJoCo islands: PGS per island with its own shuffle)"
  echo "COVERAGE not covered: equality and friction-loss rows (PGS/CG export refuses them), elliptic cones, solver x non-Euler integrator, contact-free step (the solver is not in it), CG at float32 stop tests (CG runs its iteration cap: MuJoCo's tests fire early in float32, see smooth.mojo), K>0 chains vs MuJoCo (loopk below is engine vs engine)"
  A=$(env -u ENGINE_OPT $M -m kinestride.model airborne $F $I/fly 0 | tail -1)  # standing states from the shipped Euler/Newton rollout
  for SV in 1 0; do
    export ENGINE_OPT=solver=$SV
    STATES=$A $M -m kinestride.model check $F $I/fly$SV actuated 64 || { echo "FAIL fly actuated solver=$SV"; fail=$((fail+1)); }
    for off in 200 8; do
      W=$($M -c "from kinestride.model import combine; print(combine('$F', '$SB', ($off, 0, 0), 'sworld$off'))")
      AW=$(env -u ENGINE_OPT $M -m kinestride.model airborne $W $I/world$off 0 | tail -1)
      STATES=$AW $M -m kinestride.model check $W $I/world$SV$off actuated 64 || { echo "FAIL solver=$SV world at $off mm"; fail=$((fail+1)); }
    done
  done
  # islands: two flies 200 mm apart = two kinematic trees = two MuJoCo islands (own shuffle, own stop). The merged two-fly world differs from its engine
  # twin by 7e-5 in MuJoCo itself (Newton too: ~/Projects/science/fly-connectome/docs/plans/WORLD.md, open), so this test alone runs at REL=2e-4; PGS over both flies' rows as one block (the island loop run over all rows for every root) must fail here
  FW=$($M -c "from kinestride.model import combine; print(combine('$F', '$F', (200, 0, 0), 'ffworld'))")
  AF=$(env -u ENGINE_OPT $M -m kinestride.model airborne $FW $I/ffworld 0 | tail -1)
  for SV in 1 0; do
    REL=2e-4 ENGINE_OPT=solver=$SV STATES=$AF $M -m kinestride.model check $FW $I/ff$SV actuated 64 || { echo "FAIL solver=$SV two-fly world"; fail=$((fail+1)); }
  done
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/            if Int(mi\[ROOT + Int(mi\[CPH + 11 \* rp + 2\])\]) == troot:/            if True:/' -- \
    env REL=2e-4 ENGINE_OPT=solver=0 STATES=$AF $M -m kinestride.model check $FW $I/mutant actuated 64 || fail=$((fail+1))
  # mutants: PGS projection (force clamped at 0 -> min), CG Polak-Ribiere beta sign
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/var new = max(old - rs \* get(res, f, OGA + r), Float32(0))/var new = min(old - rs * get(res, f, OGA + r), Float32(0))/' -- \
    env ENGINE_OPT=solver=0 STATES=$A $M -m kinestride.model check $F $I/mutant actuated 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/var beta = max(num \/ max(den, Float32(1e-15)), Float32(0))/var beta = max(-num \/ max(den, Float32(1e-15)), Float32(0))/' -- \
    env ENGINE_OPT=solver=1 STATES=$A $M -m kinestride.model check $F $I/mutant actuated 64 || fail=$((fail+1))
  for SV in 1 0; do ENGINE_OPT=solver=$SV $0 loopk || fail=$((fail+1)); done  # K=10 vs 10 x K=1, engine vs engine, per solver
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all solver checks (CG, PGS)"; exit 0
fi
B=${BODY:-.work/assets/fly/body.xml}  # BODY=path checks another MJCF (e.g. the spider from models/spider.py)
$M -m kinestride.model check $B $X/body-$(basename $B .xml) "$stage" 64 || { echo "FAIL $(basename $B .xml) $stage"; fail=$((fail+1)); }
if [ "$stage" = contact ] || [ "$stage" = actuated ]; then  # plane-mesh foot pairs; the spider test body has none
  echo "spider: no plane-mesh contact pairs or actuators, $stage stage is fly-only"
else
  $M -m kinestride.model check kinestride/models/spider_test.xml $X/spider "$stage" 64 || { echo "FAIL spider $stage"; fail=$((fail+1)); }
fi
if [ "$stage" = kinematics ]; then  # the gate must fail with the hinge angle's sign flipped
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/var h = 0.5 \* (qpos/var h = -0.5 * (qpos/' -- \
    $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant kinematics 64 || fail=$((fail+1))
fi
if [ "$stage" = crb ]; then  # the gate must fail with the hinge cdof lever arm reversed (anchor - com)
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/var off = com - anchor/var off = anchor - com/' -- \
    $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant crb 64 || fail=$((fail+1))
fi
if [ "$stage" = factorM ]; then  # the gate must fail with the elimination update's sign flipped
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/var scl = -get(res, f, L + adr) \* invD/var scl = get(res, f, L + adr) * invD/' -- \
    $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant factorM 64 || fail=$((fail+1))
fi
if [ "$stage" = rne ]; then  # the gate must fail with the gyroscopic term's sign flipped (cvel x* I cvel)
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/acc) + cross_force(cvel/acc) - cross_force(cvel/' -- \
    $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant rne 64 || fail=$((fail+1))
fi
if [ "$stage" = step ]; then  # the gate must fail with the body-frame angular velocity integrated as world-frame
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/mul(q, Q4(cos(half), axis\[0\] \* s, axis\[1\] \* s, axis\[2\] \* s))/mul(Q4(cos(half), axis[0] * s, axis[1] * s, axis[2] * s), q)/' -- \
    $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant step 64 || fail=$((fail+1))
fi
if [ "$stage" = contact ]; then  # the gate must fail with the contact spring pushing the wrong way
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/- K \* imp \* (dist - imargin)/+ K * imp * (dist - imargin)/' -- \
    $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant contact 64 || fail=$((fail+1))
  # direct solref (-stiffness, -damping; about the shipped 0.02/1 at d_width 0.95): K = -solref[0] / d^2, B = -solref[1] / d, no refsafe
  ENGINE_OPT=solref=-2500:-100 $M -m kinestride.model check .work/assets/fly/body.xml $X/direct contact 64 || { echo "FAIL contact direct solref"; fail=$((fail+1)); }
  ENGINE_OPT=solref=-2500:-100 ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/if ref1 > 0 else -ref1/if ref1 > 0 else ref1/' -- \
    $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant contact 64 || fail=$((fail+1))
fi
if [ "$stage" = actuated ]; then  # adhesion transmission on both pair sides and condim 1 (models/adhesion_sides*.xml, Codex review B5)
  for v in "" _c1; do
    env ${v:+CONVEX=1} NEAR_STATES=1 $M -m kinestride.model check kinestride/models/adhesion_sides$v.xml $X/adh$v actuated 64 || { echo "FAIL adhesion_sides$v"; fail=$((fail+1)); }
  done
  NEAR_STATES=1 ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/ and Int(mi\[GBODY + Int(mi\[CPH + 11 \* p\])\]) != body//g' -- \
    $M -m kinestride.model check kinestride/models/adhesion_sides.xml $X/mutant actuated 64 || fail=$((fail+1))
  env CONVEX=1 NEAR_STATES=1 ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/one = one or Int(mi\[CCND + p\]) == 1/one = one/' -- \
    $M -m kinestride.model check kinestride/models/adhesion_sides_c1.xml $X/mutant actuated 64 || fail=$((fail+1))
fi
if [ "$stage" = actuated ]; then  # the gate must fail with the servo bias (-kp q) dropped
  ~/iTools/bin/mutant-check kinestride/smooth.mojo 's/mf\[af\] \* c + mf\[af + 1\] + mf\[af + 2\] \* length/mf[af] * c + mf[af + 1]/' -- \
    $M -m kinestride.model check .work/assets/fly/body.xml $X/mutant actuated 64 || fail=$((fail+1))
fi
[ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
echo "PASS all $stage checks"
