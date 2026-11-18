# Own GPU body engine

A direct port of MuJoCo's own functions runs the actuated fly, standing on six feet with 44 joint
servos and 6 adhesion pads, at 0.64 us per animal-step for 65,536 animals: 22 times MuJoCo on six CPU
workers. Re-measured 2026-11-18 (`bench.sh run body.xml 7`, MuJoCo figure 70,582 steps/s from 2026-10-02, not re-run):
1,024 animals 45,972 animal-steps/s (0.65x MuJoCo), 16,384 424,841 (6.0x), 65,536 773,606 (11.0x).
Small batches are still slower than the CPU. Every field from kinematics to the step agrees with
MuJoCo within 1e-5 of its scale (the contact force of the unactuated probe within 3 times its float32
input floor), on the fly and, for the smooth stages, on an 8-leg spider test body, and a planted bug
fails the gate at each stage. Not yet ported: noslip and the 496 contact pairs other than the feet.
Our own renderer draws the engine's poses (`RENDERER.md`). Procedure and rules: private note `SKILL`.

## Why a port, when the GPU track is closed

private note `2026-10-03-gpu-body-port` closed GPU physics on 2026-10-02: MuJoCo Warp on gfx1100 reached correct physics
but stayed below six CPU MuJoCo workers (70,582 steps/s). This is a different bet: MuJoCo's functions
ported into Mojo for a fixed body, one GPU thread or block per animal, so every rule is ours to change
(joints, adhesion, new species) and brain and body can share one process.

## Design

```
any MJCF (fly body-v2, spider test model, ...)
   │ kinestride/model.py export: mjModel fields by name -> mi.bin (int32), mf.bin (float32),
   │                                    sizes.txt (-D NBODY NJNT NQ NGEOM NV NC)
   ▼
kinestride/smooth.mojo  -DSTAGE=N  one compiled binary per body type
   per-animal arrays [row][animal], animals fastest; model arrays shared, in cache
   ▼
model.py check STAGE: ~/iTools/bin/mj-oracle random states -> MuJoCo float64 vs engine float32,
                      every output field, error relative to the field's own scale, gate 1e-5
check.sh STAGE:      fly + spider test body + a planted mutant (~/iTools/bin/mutant-check)
```

A new species is a new MJCF file and an export, not new engine code. Refused at export (not ported):
simple dofs; tendon limits are ported (check.sh tendon_limit); tendon equalities are ported (check.sh tendon_eq); runtime eq_active, inactive-at-start equalities included, is ported (check.sh eqact_runtime); the sleep filter is ported for one tree per animal (check.sh sleep); the flight model's own body-body contacts are ported (check.sh flight_contacts: automatic pairs, primitive narrowphase, condim 1). Equality constraints (connect, weld, joint), mocap
bodies, gravcomp and spatial tendons are ported (lane-constr units 1 to 4, `b9e7e5a`, `4c3eab4`, 9e96872; unit 5, full implicit + midpoint, `4450a20`): `check.sh gravcomp`, `check.sh equality | equality_contact | mocap`, fixtures
`kinestride/models/eq_*.xml`.

## Stages

| Stage | MuJoCo 3.9.0 function | Status | Fly max err | Spider max err | Mutant |
|---|---|---|---:|---:|---|
| S0 spike | toy: rigid thorax, 6 kinematic legs, soft feet | speed PASS, stability FAIL | | | |
| S1.1 | mj_kinematics (bodies, inertial frames, geoms, anchors, axes) | PASS `ed60339` | 7.6e-07 | 7.9e-07 | hinge sign, caught |
| S1.2 | mj_comPos + mj_crb (subtree CoM, cinert, cdof, crb, sparse M) | PASS `20e8b51` | 1.3e-06 | 9.4e-07 | hinge lever arm, caught |
| S1.3 | mj_factorM | PASS `798bc33` | 3.0e-06 | 1.7e-06 | elimination sign, caught |
| S1.4 | mj_comVel, mj_passive, mj_rne (qfrc_bias) | PASS `8b87dbc` | 3.2e-07 | 6.8e-07 | gyroscopic sign, caught |
| S1.5 | mj_Euler, implicit dof damping (qH, qacc, qpos) | PASS `8b87dbc` | 3.6e-06 | 2.0e-06 | body w as world w, caught |
| S1 gate | full contact-free step at 65,536 animals | PASS: 0.23 us per step (gate 20) | | | |
| S2 probe | plane-mesh feet (6 pairs), pyramidal cone, mj_solPrimal Newton, step | PASS `53828de`: 2.0 us per step (gate 14) | qacc 3.0e-06, qfrc_constraint 4.0e-05 * | | contact spring sign, caught |
| S2 | noslip, all contact pairs (broadphase), warm start | | | | |
| S3 | mj_transmission + mj_fwdActuation: 44 joint servos, 6 adhesion bodies | PASS `7e66ef6`: 0.64 us per step at 65,536 | qacc 4.0e-06, actuator_force 7.0e-08 | | servo bias dropped, caught |
| Floor | all 55 plane-mesh pairs (every part on the floor) | PASS `3f1b1d5` | qfrc_constraint 1.9e-04 (float32-floor bar 2.1e-04) | | | feet-only sank through the floor in MuJoCo too |
| Loop | K steps per launch, state kept on the card | PASS `8e5654f` | K=1 qvel 4.8e-06 | | qvel write-back dropped | K=10 byte-equal to 10 x K=1 |
| Origin | floating origin per animal (root-relative step) | PASS `670866c` | cube cinert 2.3e-07 (was 1.01e-05) | | | needed for worlds 20-40 mm across |
| Cube | 40 mm cube of 6 planes, 330 pairs, states on every face | PASS `e1731b9` | 25 fields within bars | spider body PASS | bounding-sphere skip sign | hulls shared: 2.90 -> 0.54 MB |
| Spider | generated salticid body (`models/spider.py`) | PARTIAL `bbe2934` | | kinematics..step PASS; contact/actuated qfrc_constraint 1.5e-05..5.2e-05 | | qacc passes 6.5e-06; cause open |
| S6+ equality | mj_instantiateEquality, mj_diagApprox/makeImpedance/referenceConstraint, mj_Jdotv (engine_core_constraint.c:513,1637,2069,3048,973) | PASS (`equality`, `equality_contact`; qacc_euler bar lifted to 1e-4, measured 4.6e-5) | qfrc_constraint 4.9e-6, qacc 1.2e-6 | n/a (no spider equality body) | 5: anchor sign, weld conjugate, joint polynomial sign, Jdot v sign, aref dropped |
| S0+ mocap | mj_kinematics mocap branch (engine_core_smooth.c:86) | PASS (`mocap`) | qfrc_constraint 4.5e-7, qacc 7.4e-7 | n/a | 2: mocap index, quaternion order |
| S4+ gravcomp | mj_gravcomp, mj_passive hand-off, actuator-level gravcomp in mj_fwdActuation (engine_passive.c:817,1012; engine_forward.c:770) | PASS (`gravcomp`: fly, spider, `gravcomp_act.xml` for nu > 0 routing) | qfrc_passive 1.4e-7, qfrc_bias 2.9e-7 | qfrc_passive 2.7e-7, qfrc_bias 6.5e-7 | 4: gravcomp skipped, sign flipped, actuator route dropped, actuatorgravcomp ignored |
| S4+ spatial tendons | mj_tendon (engine_core_smooth.c:922), mju_wrap (engine_util_misc.c:281), tendon spring/damper/servo through ten_J | PASS (`sptendon`: fixture, fly, spider; `sptendon_act`) | ten_length 3.2e-7, ten_J 4.5e-7 | ten_length 1.8e-7, ten_J 1.3e-6 | 6: wrap side, tangent solutions, moment sign, pulley divisor, cylinder height, inside-wrap angle |

64 random states per body: free joint anywhere at random orientation, ball random, hinges and slides
in range, every dof velocity uniform in +-10. The S1.5 reference is MuJoCo's own mj_step with contact,
constraint and actuation disabled; qacc_euler is its velocity change over dt. The spider is `kinestride/models/spider_test.xml` (8 legs; free, ball, hinge, slide),
written to exercise joint types the fly lacks, with a spring and damper on every joint; it is not a
real spider body.

\* qfrc_constraint's bar is 3 times its float32 input floor: MuJoCo itself moves 2.4e-05 when fed the
float32-rounded state the engine receives, because at solref 2e-4 the foot force is about 2.5e7 times
the penetration. No float32 engine can meet 1e-5 there; extra solver iterations do not change ours.
The S2 probe's states come from actuated MuJoCo rollouts (half standing on six feet, half with jittered
servos, one to five feet down), not random poses.

## Speed

| What | 1,024 animals | 16,384 | 65,536 |
|---|---:|---:|---:|
| S0 toy full step, fly-steps/s | 0.47e9 | 5.4e9 | 8.7e9 |
| S1.1 kinematics, per launch | 284 us | 380 us | 1.37 ms |
| S1.2 kinematics + comPos + crb, per launch | 2.36 ms | 3.84 ms | 6.37 ms (97 ns per animal) |
| S1 full contact-free step (S1.1-S1.5), per launch | 3.55 ms | 6.70 ms | 15.0 ms (0.23 us per animal) |
| S2 probe: + foot contacts + Newton, per launch | 32.1 ms | 83.4 ms | 133 ms (2.03 us per animal) |
| same, float32 stopping rules (`bb9596c`) | 18.6 ms | 31.4 ms | 61.9 ms (0.94 us per animal) |
| S3: + 44 servos + 6 adhesion, standing | 13.7 ms | 21.8 ms | 42.2 ms (0.64 us per animal, 22x CPU) |
| floor 55 pairs, K-loop (`3f1b1d5`), 64 animals | 23.6 ms per step | | |
| cube 330 pairs (`e1731b9`), 64 animals | 75-85 ms per step | | |

S0 to S1.2 ran on the Miner-125 LACT profile (1,400 MHz, -125 mV); the full step on Ultimate-AI.
`kinestride/bench.sh` reproduces the last two rows (`bench.sh build MJCF 6`, then `run` under gpu-wait).

Not optimised yet: one thread per animal, every intermediate in global memory (5.8 KB per fly for
kinematics alone), each body reading its parent right after writing it. Fusion keeps intermediates in
registers or LDS; the 74 x 74 factor needs a block per animal.

## S0: four orders of headroom, and a toy that still tips

The toy (no mass matrix, no leg dynamics, no meshes; not comparable to MuJoCo) cleared the 500k
fly-steps/s gate by four orders of magnitude. Four bugs found on the way are now rules in the skill:
foot velocity from the previous step's position pumped tilt modes; the quaternion was integrated with
the body-frame product for a world-frame angular velocity; the gait swept left and right feet in
opposite directions; the swing foot sat lower than the stance foot. With adhesion (40 uN per stance
foot, the body-v2 gain, about 4 body weights) it still tips to 117 degrees; without, it inverts. Open
causes: prescribed legs without compliance, centre of mass at the hip centroid (body-v2: 0.30 mm back,
at the hind coxae), invented gait. S1 uses the real body, so these are not chased in the toy.

## The S2 probe passed: contacts cost 9 times the smooth step

The probe gate was 14 us per animal-step at 65,536 animals. Feet, impedance and the Newton solve take
the step from 0.23 to 2.03 us, so the engine runs 492,000 animal-steps per second, 7 times MuJoCo on
six CPU workers, before any fusion. What is left of S2 adds work: noslip (the fly model runs 5
iterations), the 496 other contact pairs (body, tibiae; a broadphase) and warm start, which cuts
Newton iterations. Then S3 (44 servos, 6 adhesion actuators).

## The S1 gate

The gate was a full contact-free step at 65,536 animals under about 20 us per animal-step, leaving
contacts room to cost several times more and still beat 70,582 steps/s. The step takes 0.23 us:
4.37 million animal-steps per second, before any fusion. Contacts may now cost 60 times the smooth
step and still match the CPU. S2 is the risk: an iterative contact solver is where MuJoCo Warp lost
to the CPU on this card.

## Parameters

A research check (fly-connectome `.work/research-verified`, not kept) compared a Perplexity report with body-v2. Ours: mass
1.026 mg, CoM x -0.30 mm, femur 0.71-0.84 mm, tibia 0.52-0.68 mm, Euler integrator with Newton
solver, solref (2e-4, 1.0) and (2e-3, 1.0), friction 0.4 / 1.0, adhesion gain 40, kp 45 and 2. The
report was wrong on solref damping (1e3), adhesion (at most 1 body weight) and MuJoCo Warp on AMD.

## Fluid forces and fixed tendons (flight model, lane-flight)

Export (`model.fluid`, `model.tendons`) appends two blocks after every existing block, so walking-fly and spider
`mi.bin`/`mf.bin` stay byte-identical to main (`cmp_export.sh`). STAGE 4 `fluid()` ports `mj_fluid`
(engine_passive.c:842): ellipsoid model (added mass, Magnus, Kutta lift, blunt/slender drag, viscous resistance) on the
two wing geoms, inertia-box model on the 130 other bodies with mass. Wrenches are taken to the root's subtree_com, summed
leaf to root into one scratch row per weld body, then `qfrc_fluid[v] = cdof[v] . W[body(v)]`: one O(nbody) pass replaces a
`mj_applyFT` chain walk per body. Pitfall found by the gate: for a body, `ximat` is the inertial frame's rotation, so the
box model's frame is `xquat * body_iquat` (error was 1.7e-3 before, 8.8e-8 after). Gate: `check.sh rne` on the flight MJCF
(`models/flight_export.py`), `check.sh fluid` adds flapping-rate velocities and mutants (Kutta, Magnus, added mass, box
drag signs). Next: tendon kernels (length, spring/damper, actuator trntype 3), then the elliptic cone.
Flight model cost (`bench.sh` stage 5, contact-free step with fluid, 135 bodies): 1,024 animals 3.68 ms, 16,384 animals 6.62 ms, 65,536 animals 17.5 ms per launch (3.74 M animal-steps/s). Walking fly stage 7 at 65,536 unchanged: 65.3 ms.

Fixed tendons (unit 3): `tendon()` (length = sum coef qpos, velocity = sum coef qvel, constant moment = the wrap coefs),
`tendon_passive()` (engine_passive.c:775 spring about [lower, upper] plus damper, into qfrc_passive), actuation trntype 3
(moment gear x coefs). The shipped flight tendons carry no spring or damper, so the gate model `airborne.xml` adds them
(2.0, 0.3, dead zone +-0.04) and drops contacts and the elliptic cone: `check.sh tendon` = `actuated` on it, 25 fields x 64
states within 1e-5, mutants length sign, passive sign, actuator sign, dead zone dropped all caught. Contacts on the flight
model (elliptic cone, geom contype pairs) are unit 4.

Elliptic friction cone (unit 4): contact rows are the contact-frame rows (normal, t1, t2); R0 = (1 - imp) tran / imp, R1 = R0 /
impratio, R2 = R1 f0^2 / f1^2, friction K = 0; mu of the regularized cone = f0 / sqrt(impratio). Constraint update zones
(engine_core_constraint.c:3155): top N >= mu T no force, bottom mu N + T <= 0 quadratic, middle cost 0.5 Dm (N - mu T)^2, Dm = D0 /
(mu^2 (1 + mu^2)); Hessian block as HessianCone; line search as PrimalEval; noslip solNoSlip's elliptic branch with mju_QCQP2
and the projection onto the ellipse. The Hessian is rebuilt every Newton iteration (the cone block moves with jar). Findings:
(1) MuJoCo's own stop (tolerance 1e-8) leaves ~1e-4 of force error on a nonlinear cone solve (the pyramidal solve is exact once
the active set settles), so the gate runs MuJoCo to tolerance 0; (2) the engine's step stop must be 1e-7 for the cone (1e-6
left 4.8e-4; 1e-7 costs 6.1 Newton iterations, 32 line-search evaluations at 64 animals); (3) cone fields carry a wider float32
bar (15 x floor: qfrc_constraint 5.3e-5, qacc_euler 2.7e-5 measured, floor 4.3e-6) because the converged cone solve amplifies
rounding. The flight model itself has no portable contact: its plane has contype 0, its only contacts are 7 body-body pairs
at rest (capsule, ellipsoid, cylinder), which need GJK/EPA; the cone is gated on the walking fly with cone=elliptic.
