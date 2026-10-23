# Body engine

A GPU port of MuJoCo 3.9.0 physics in Mojo, batched: one GPU thread (one 32-lane wave in WAVE builds) per animal.
It reads no MJCF. `model.py export` flattens any MuJoCo model into `mi.bin` (int32) and `mf.bin` (float32), and
the sizes go into `-D` constants, so each body type compiles its own binary. Every ported stage is checked
against MuJoCo itself, plus a mutation that must fail.

Current status: the capability table in `README.md`. Design: `docs/` (ENGINE, ENGINE_OPT, HFIELD_CONTACT,
CONVEX_CONTACT, RENDERER); history in `~/Brain/kinestride/notes/2026-10-03-gpu-body-port.md` and
`~/Brain/kinestride/notes/2026-10-03-engine-finish.md`; app plans (WORLD, BRAIN_BODY_LOOP) in `~/Projects/science/fly-connectome/docs/plans/`.

## Quick start
```
gpu-wait run --priority 90 --timeout 1200 -- kinestride/check.sh step        # parity vs MuJoCo
gpu-wait run --priority 90 --timeout 1200 -- env BODY=path.xml kinestride/check.sh actuated
(fly-connectome) .venv/bin/python -m agent.body_engine.live check                                     # live view self-test
gpu-wait run ... -- (fly-connectome) .venv/bin/python -m agent.body_engine.live watch 10 [cube] [spider]   # watch, K=10 steps/frame
gpu-wait run ... -- (fly-connectome) .venv/bin/python -m agent.body_engine.live fps 10 [world]        # ms per frame
gpu-wait run ... -- (fly-connectome) .venv/bin/python -m agent.body_engine.live watch 10 world-game   # spider + follow camera + game bars + plume text and mini-map, fps printed (fps: same command, 300 frames)
kinestride/bench.sh build MJCF [STAGE]; gpu-wait run ... -- kinestride/bench.sh run MJCF [STAGE]
```

## Pipeline (smooth.mojo stages, each one MuJoCo function group)
| stage | ports | check.sh |
|---|---|---|
| 1 | mj_kinematics (quaternions only, no 3x3 xmat) | `kinematics` |
| 2 | mj_comPos, mj_crb (sparse M, MuJoCo layout) | `crb` |
| 3 | mj_factorM (sparse L'DL) | `factor` |
| 4 | mj_comVel, mj_passive (springs, dampers, fluid, fixed-tendon spring and damper), mj_rne | `rne`, `fluid`, `tendon` (flight model) |
| 5 | mj_Euler, implicit dof damping, mj_integratePos | `step` |
| 6 | plane-mesh contacts, pyramidal or elliptic cone, Newton (warm start), noslip | `contact`, `cube`, `terrain`, `cone` (elliptic) |
| 7 | actuators (joint servos, adhesion, fixed tendons) from ctrl.bin | `actuated`, `loop` |
| + | sphere-sphere grooming pairs; fly + spider in one model | `groom`, `world` |
| viewer | the live loop's core (`viewer.mojo`: engine + renderer buffers on one DeviceContext, `step`, `frame`) and `view.mojo`, the loop with no brain; fly-connectome's live.mojo runs on it | `view` (state after 10 x K=1 == smooth K=10, first frame == render on the start pose, byte-equal; 2 mutants) |
| spectate | camera follows the moving animal (`-DFOLLOW=1`) and energy/health/stamina bars (`-DHUD=1`) in the live frame, frame cost split (`-DPROFILE=1`); view code, no MuJoCo function | `spectate`, `python -m kinestride.spectate_check` |
| world_hud | plume world in the live frame (`-DWORLD=1`): odor sites sampled on the card from world_content's descriptor, concentrations printed in the HUD, sources drawn in a mini-map (fly sites, spider leg I tarsus tips) | `world_hud` (`world_check.py`) |
| flylink3 | the same fly with the host taken out (`live.mojo -DBRAIN=3`, one DeviceContext): resident Shiu brain, DN readout, FlyGym gait kernel on the card, no host copy per step | `flylink3` (`fly_link3.py`), `flygait` (gait kernel vs HybridTurningController), `shiures` (resident brain vs ShiuLive) |
| flylink | brain-driven fly (host transport, `live.mojo -DBRAIN=1`): resident Shiu -> DN drives -> FlyGym gait mapping -> 50 ctrl; walks 3.0 body lengths in 1 s | `flylink` (`fly_link.py`) |
| game | energy/health/stamina/death per animal: standalone module and oracle, and the per-step engine hook (`-DGAME=1`, STAGE 7, comptime-folded out otherwise): work drain `sum abs(qfrc_actuator x qvel) x dt`, stamina, death zeroes servo and adhesion effort; rules at the end of `mf.bin` | `python -m kinestride.game_state_check` (module), `game` (engine vs `game_events.replay` on MuJoCo work, fly + spider, death gate, HUD from the rows) |
| + | brain<->body link: ctrl written each control step (host pipe or on-card brain kernel), sensor packet out; spider gait brain | `spiderlink` |
| + | equality constraints connect, weld, joint (always-active Newton rows, mj_Jdotv correction) | `equality` (alone), `equality_contact` (with an active sphere-sphere contact and noslip 5 in one solve) |
| + | coupling across trees in one model (contact pairs, connect/weld/joint equalities, spatial and fixed tendons, tendon limits): each pair side's Jacobian about its own root, geom1 moved into geom2's tree frame; Newton, pyramidal, Euler only | `cross` |
| + | mesh-mesh pairs (mjc_Convex at margin > 0: one GJK + EPA contact, then the perturbation search, up to 5), geom1's hull per pair (`-DNMM`); margin 0 is refused (MuJoCo's native polygon clipping is not ported); a model's every potential pair joins H's static pattern, so many cross pairs cost every step | `meshmesh` |
| + | joint actuator force limits (jnt_actfrcrange clamp on qfrc_actuator after actuator gravcomp) | `actfrc` (fly, spider and `models/gravcomp_act.xml` via `model.with_actfrc`) |
| + | body gravcomp (mj_gravcomp: -gravity mass gravcomp at xipos through the body's point Jacobian; per joint passive route or, with `actuatorgravcomp`, the actuator route) | `gravcomp` (fly and spider with gravcomp set via `model.with_gravcomp`, plus `models/gravcomp_act.xml` for the actuator route) |
| + | spatial tendons (mj_tendon: site paths, pulley divisors, sphere and cylinder wraps with sidesite via mju_wrap incl. the inside wrap; ten_length, ten_velocity, dense ten_J; tendon spring, damper and servo through the moment row) | `sptendon` (fixture `models/tendon_spatial.xml` random and actuated states, fly and spider with three added spatial tendons via `model.with_spatial_tendons`) |
| + | mocap bodies (mocap_pos/quat per animal in ctrl.bin rows NU..; kinematics; welded or connected to the animal) | `mocap` |
| + | push input: MuJoCo xfrc_applied (mj_xfrcAccumulate through mj_applyFT into qfrc_applied, part of qfrc_smooth), sparse slots per animal in ctrl.bin after the mocap rows (`-DNXF`, layout in `docs/API.md`) | `xfrc` (fly actuated, spider step, a push across trees) |
| + | implicitfast integrator (mj_implicitSkip, `-DINTEGRATOR=3`, comptime) | `integrator` (MuJoCo parity), `loopk` (K=10 vs 10 x K=1, engine only) |
| + | full implicit integrator (mj_implicit: qDeriv incl. `mjd_rne_vel`, no-fill-in LU in `qLU`, `-DINTEGRATOR=2`, comptime) | `implicit` (MuJoCo parity: fly, spider, two-animal world) |
| + | implicitfast midpoint of standalone free bodies (mj_midpoint, runtime eligibility: no live contact) | `implicit` (`models/midpoint.xml`, `midpoint_contact.xml`) |
| + | RK4 integrator (mj_RungeKutta, `-DINTEGRATOR=1`, comptime: 4 forward passes, `rkstage` in smooth.mojo) | `rk4` (MuJoCo parity), `loopk` |
| + | CG (mj_solCG, `-DSOLVER=1`) and PGS (mj_solPGS, `-DSOLVER=0`) beside Newton, comptime | `solver` (MuJoCo parity, fly + worlds + island mutant), `loopk` |
| world content | seeded food/decoy plume descriptors and concentration sampling at supplied antenna XY poses | `python -m kinestride.world_content_check`; GPU fixture `world_content_gpu_check.mojo` (live/render hook pending owner integration) |
| + | jump: spider hydraulic-burst jump variant, take-off speed/angle/distance vs MuJoCo and Nabawy 2018 | `jump` |
| + | native vision: 6 eye cameras on the spider head rendered per animal (retina luminance in the packet), prey body moved by the environment | `spidereye` |
| + | box-mesh and ellipsoid-mesh pairs in the step: MuJoCo native GJK + EPA and mjc_Convex perturbation search (`convex.mojo`, float64 narrowphase), opt-in `CONVEX=1` | `convex` (narrowphase vs mj_collision, CPU), `arena` (step: original arena, states in walls and stones), `convexstep` (20 chained steps, fly in its arena and spider on stones, vs mj_step; forces on engine-evolved states) |
| + | exact heightfield contact `mjc_ConvexHField` (GJK + EPA per cell prism on the terrain function), opt-in `HFEXACT=1`; tangent plane stays the default | `hfield` (narrowphase vs MuJoCo hfield, CPU), `terrain_exact` (step, bumpy terrain, fly and spider), `cone_hfield` (elliptic cone on the same terrain, noslip 0 and 5, bar 5e-4) |

## Supported / missing
Supported: free, ball, hinge and slide joints; plane-mesh and sphere-sphere contact; box-mesh and ellipsoid-mesh contact (GJK + EPA, `CONVEX=1`; on by default in the fly link, `fly_link.prepare(convex=True)`, gated in the walking loop by `check.sh stone` and `check.sh wall`; memory grows with the number of convex pairs, see docs/CONVEX_CONTACT.md); exact heightfield contact (`HFEXACT=1`, 16 contacts per foot pair); endless terrain (`world.mojo`,
contacted as a tangent plane, wrong where it bends inside one foot); many animals per model, each with its own
floating origin; K steps per launch (K=10 byte-equal to 10 x K=1); standalone seeded plume content with a shared
food/decoy descriptor, presence/depletion state and concentration-only antenna sampler (coordinates in world mm;
`maps.py` oracle in metres, converted at 1000 mm/m).

The flyloop Shiu worker also supports resident continuous advances: one uploaded neural graph and persistent state
can be advanced in successive tick windows, including beyond tick 32,000. Its int32 last-spike state costs two
additional bytes per neuron versus the earlier int16 state; the refractory state stays uint8. See
`~/Projects/science/fly-connectome/docs/plans/BRAIN_BODY_LOOP.md` and fly-connectome `agent/verify_shiu_continuous.py` for the split-run gate. The one-process sensory,
neural, body, and renderer path remains pending owner integration.

Fluid forces (opt density, viscosity, wind: ellipsoid model on geoms with fluidshape, inertia-box model on every other body with mass,
`mj_fluid`) are ported in STAGE 4 for the Flybody flight model, comptime-folded out (`NFLROW` unset) for walking bodies. The elliptic friction cone (`opt.cone elliptic`, comptime `ELLIPTIC`, export sets it) is ported for the exported plane-mesh and
sphere pairs: three contact-frame rows per contact (R1 = R0 / impratio, R2 = R1 f0^2 / f1^2), the cone's top / bottom / middle
zones, the cone Hessian block (rebuilt each Newton iteration), the line-search cone terms and noslip as a QCQP over the friction
ellipse. Gate `check.sh cone` on the walking fly with `model.elliptic` (cone elliptic, noslip 0 and 5, MuJoCo to tolerance 0). Export refuses the elliptic cone with RK4 or implicitfast (untested).

Fixed tendons (joint wraps with coefs; length, velocity, spring with dead zone, damper, actuator trntype 3) are ported: export block
`model.tendons` (NTEN/NWRAP; wrap arrays last so spatial wraps append), kernels `tendon()`/`tendon_passive()`/`actuation()`. Gate
`check.sh tendon` runs the flight model airborne (`model.airborne`: no contacts, spring and damper added to its tendons). `python -m kinestride.flight check` flies it: open-loop flapping from Flybody's wing-beat pattern generator, engine vs MuJoCo over 22 beats, half-beat mutant must fail; `watch` shows it in mpv.

Equality constraints (connect, weld, joint, tendon; body and site semantics) and mocap bodies are ported (`model.eq_rows`/`eq_block`/`mocap_block`, last export blocks;
`equality()` in smooth.mojo; NEQ and NMOCAP comptime, absent for walking bodies). Export refuses equality with CG/PGS, elliptic cones, a non-Euler integrator,
adhesion on a body with a gap pair (excluded contacts in the gap band transmit adhesion in MuJoCo; refused), flex equalities and coupling across two trees outside Newton + pyramidal + Euler (`check.sh cross`). Every equality is exported, inactive at start too (`eq_active0`, the 13th int per equality of `eq_block`): without `-DEQACT` the engine holds
`eq_active0`, with it the host's ctrl.bin rows switch any equality per animal between launches (D = 0 for an inactive one, as MuJoCo drops its rows); `check.sh eqact_runtime`
(`model.with_inactive`: every other equality inactive at start; eq_test, the spider's fixed-tendon equalities within 1e-5, the fly's spatial-tendon ones within 2e-3, EQACT unset and
random; `eqact_rollout`: 12 K=1 launches x 64 animals, the host flipping a third of the equalities every 3 launches, qpos_next 4.7e-7, qacc_euler 5.7e-5 of its max; 2 mutants).
Gates use states near qpos0 (`model.nearby`; `EQSPREAD=far` for metre-sized violations, MEASURED); qacc_euler's bar is lifted to 1e-4 (float32 cancellation in D (J qacc - aref), measured 4.6e-5).

Gravcomp (`body_gravcomp`, `model.gravcomp_block`, `gravcomp()` in smooth.mojo, comptime `GRAVCOMP`) is ported including `jnt_actgravcomp`: those joints' share goes to
qfrc_actuator (engine_forward.c:770), which MuJoCo only does when the model has actuators (nu > 0; with nu = 0 the compensation of such joints is lost in MuJoCo too, and here).
Joint-level actuator force limits (`jnt_actfrcrange`, `model.actfrc_block`, comptime `NAFRC`) clamp qfrc_actuator at each limited joint's first dof after the actuator gravcomp
(clampVec, engine_forward.c:788), as MuJoCo does only when nu > 0; `check.sh actfrc` (fly, spider and `models/gravcomp_act.xml` through `model.with_actfrc`: gravcomp on,
ranges binding on both sides in every state, states from the unlimited model's rollouts; fly and spider contact-free, so their own contact float32 floors stay out of this gate; 5 mutants).

Spatial tendons (`model.spatial()`, `spatial()` in smooth.mojo, comptime `SPATIAL`, appended right after the fixed-tendon arrays; with SPATIAL every tendon, fixed or spatial, takes
its length and dense moment row from `spatial()` and tendon spring, damper and servos read those rows) are ported: site paths, pulleys, sphere and cylinder wraps with sidesites. The inside wrap
(sidesite inside the sphere) runs its Newton loop in float64: in float32 asin(z) near 1 cannot reach the loop's 1e-6 tolerance (3 of 64 gate states failed), and the GPU has no float64
sin/cos, so only the loop is float64. Joint limits (hinge and slide, `mj_instantiateLimit`) are the same unilateral rows: `dist = side (range - qpos)`, `J = -side` on the dof, exported whenever the model already has a constraint block (contacts, equalities, tendon limits) under Newton, pyramidal and Euler, else dropped with a printed note; the shipped fly carries 4 (its two proboscis joints); `check.sh joint_limit` (fly and spider, every hinge/slide limited to its 35..65 % qpos quantiles, REL 2e-3, 2 mutants). Dof friction loss (`mj_instantiateFriction`) rides the same block between the bilateral and the limit rows (MuJoCo's efc order; `NEQB..NEQL`, comptime-folded out when absent): `J = 1` on the dof, pos 0 so `aref = -B J qvel`, `R` from `dof_solref`/`dof_solimp` and `dof_invweight0`, Huber cost in Newton (quadratic while `-R floss < jar < R floss`, force `-+floss` outside; `update`, `peval`, `hessian` and the Newton active-set test all switch on it); Newton, pyramidal, Euler and noslip 0 only (noslip over friction rows is not ported); `check.sh frictionloss` (fly and spider, every joint's frictionloss at a quantile of its unbounded force so rows sit in all three zones, within 1e-5, 3 mutants). Tendon friction loss follows the dof rows in the same range (`J = ten_J`, `tendon_solref_fri`/`tendon_solimp_fri`, `tendon_invweight0`); `check.sh tendon_friction` (tendon_spatial.xml, the spider with one fixed tendon, fly and spider with three spatial tendons; frictionloss at the median of each row's single-row force; within 1e-5, the spider 2e-5 for its spatial ten_J's own 1.15e-5; 2 mutants). Zone coverage comes from the final `jar` (`tools/friction_zones.py`): MuJoCo's `efc_state` after `mj_forward` is stale on friction rows (a row reported linear carries `efc_force = -jar/R`). Tendon limits (`mj_instantiateLimit`) are rows on the equality block, one per side, unilateral in Newton (`check.sh tendon_limit`: fixture at MuJoCo's default solref 0.02 within 5e-4 and at solref 0.1 within 1e-5, fly and spider within 2e-3, 4 mutants). Tendon equalities (`mjEQ_TENDON`, `mj_instantiateEquality`) are bilateral rows on the equality block, type 5 in `eq_block` (3 and 4 are the limit rows): `L0 - length0_0 - data[0] - poly(L1 - length0_1)`, `J = J0 - deriv J1`, `tendon_invweight0` summed for `dA`, no Jdot v term (MuJoCo's `mj_Jdotv` covers connect and weld only); `check.sh tendon_eq` (`model.with_tendon_equalities`: one two-tendon quartic and one single-tendon offset, solref 0.1, tolerance 0; the tendon fixture and the spider with two fixed tendons within 1e-5, fly and spider spatial tendons within 2e-3, 3 mutants). Coupling two spatial tendons of the fixture (length0 0.25 m, spread 8e-4 over nearby states) puts `L - length0` at float32 resolution: qacc 5.6e-5 (CPU, Newton to 0), so the fixture couples a spatial tendon to its fixed one. Direct solref (-stiffness, -damping) runs on every row: equality, joint and tendon limit rows follow `mj_makeImpedance` (K = -solref[0] / d_width^2, B = -solref[1] / d_width, refsafe on the standard format only; a mixed format is replaced by the default 0.02 1 at export, as `getsolparam` does); `check.sh solref_direct` (ENGINE_OPT `rowsolref`: eq_test at -2500 -100 and mixed 0.1 -20 within 1e-5, fly and spider joint limits and the spider's tendon limits at -100 -20 within 2e-3, the tendon fixture within 1e-5, 3 mutants); contacts: `check.sh contact` at direct (-2500, -100).

Port queue: `port_queue.tsv` (driven by `~/iTools/bin/mj-port` source / brief / gate / land, receipts in `.work/port/`, the panel's port tab). Sleep (`mj_sleep` in `mj_advance`, `mjENBL_SLEEP`): one animal is one tree and its own island, so a sleeping animal skips its step until the host writes a velocity; `check.sh sleep` (fly with sleep allowed on the root, tolerance 5, 32/64 standing states asleep after 20 held steps, asleep sets equal, sleepers' qpos within 1e-5, 3 mutants). Flight contacts (`mj_collideGeoms` on the model's own body-body pairs): `model.with_auto_pairs` writes MuJoCo's automatic pairs (its static filters and `mj_contactParam` mixing) as explicit pairs, `tools/auto_pairs_check.py` proves them against the dynamic pairs; the engine collides sphere, capsule, ellipsoid and cylinder pairs (capsule-capsule, sphere-capsule, sphere-cylinder analytic; the rest GJK + EPA with the sphere/capsule core trick), condim 1 rows, and keeps a row while `dist < margin` with the narrowphase at `margin + gap` (3.9's convention; the fly's zero gap never told the two apart). `check.sh flight_contacts`: 842 pairs, narrowphase per kind in float64, random airborne states within the float32 floors and near-rest states within 3e-3, 5 mutants. Capsule/cylinder pairs get mjc_Convex's perturbation search (up to 5 contacts), as MuJoCo with multiccd on; EPA multicontact itself only ever fires on box and mesh faces. Not ported: condim 4 and 6, an explicit pair whose geoms sit on two animals.
Solvers: Newton (default), CG and PGS are ported for the pyramidal cone on plane-mesh and sphere pairs (export refuses elliptic cones and convex pairs with them); equality and friction-loss rows are Newton only (export refuses them with CG/PGS). Sleep (enable bit 16) is off in every model here: not needed.
implicit and implicitfast: the actuator velocity term covers joint and tendon actuators (gear^2 biasprm[2] J'J onto D's / M's pattern; `check.sh implicit` with `ENGINE_OPT=tkv`, sign mutants). `mjd_passive_vel` is ported in full: tendon damping (J' damping J onto D's / M's pattern; tendon_spatial under both integrators, sign mutants) and the fluid blocks (`mjd_ellipsoidFluid`: Magnus, Kutta, drag, torque, added mass; `mjd_inertiaBoxFluid`: diagonal; Flybody flight.xml step under both integrators, drag-sign mutants), all in `check.sh implicit`. Cost on 64 flight animals (130 inertia boxes + 2 wings): Euler 2.9 ms, implicit 35 ms (6.7 without the fluid blocks), implicitfast 31 ms per launch; the (k, p) entry comes from exported NV x NV tables (`pair_table`), the rest is the per-body chain collection and the per-pair products. Sleep (enable bit 16) is off in every model here and not ported.

Full implicit (`implicit_factor`, `implicit_solve` in smooth.mojo; note MuJoCo's numbering: implicit is 2, implicitfast 3, so `-DINTEGRATOR=2`): qLU = M - h qDeriv in MuJoCo's dof-dof sparsity D (a dof's row is its body's B row: the whole branch, ancestors, own dofs and subtree), `qDeriv` = dof dampers + actuator velocity term + `mjd_rne_vel`. The RNE derivative keeps one 6-vector per B entry for cvel, cacc and cfrc_body;
MuJoCo's 6x6 products are linear in that vector and are written as spatial operations (crossMotion, mulInertVec, crossForce). The factor is `mju_factorLUSparse` (reverse order, no fill-in on a tree), the solve `mju_solveLUSparse`. Layout: `model.implicit_block` appends B, D and `mapM2D` as the last ints; scratch rows `OQLU`, `ODCV`, `ODCA`, `ODCF` (fly: 21,348 floats, 85 KB per animal, 5.6 GB at 65,536: a non-walking mode; `WAVE=1` untested).
Midpoint (`midpoint`, `midpoint_newton`, `mid_eligible`): under implicitfast a childless free body with no live contact on it is integrated by `mj_midpoint`; the engine solves for the velocity change (w_mid - w) so that `(new - old) / h` does not cancel in float32 (9e-5 before, 1e-6 after), and the next position uses MuJoCo's qvel_mid. MuJoCo's `simple` dofs (ipos 0, iquat identity on such a body) have diagonal-only M rows: `crb` stops at `M_rownnz`. A contact pair across two free bodies is refused (two floating origins): pair each with a static geom. A static world sphere now subtracts the animal's floating origin like the plane does.

Brain graph selection and motor-neuron annotation are outside the physics kernels. The MCNS Shiu graph reuses the same CPU/GPU LIF kernels as FlyWire; `agent.verify_mcns` checks exact per-cell spike-count parity on the 165,733-neuron, 6,239,854-edge graph. The FlyWire-compatible DN and leg-MN selectors live in `agent.mcns_mapping` and are documented with coverage in `~/Projects/science/fly-connectome/docs/plans/MCNS.md`.

## Files
```
model.py     export (INTS/FLOATS field order), oracle helpers, standing states
smooth.mojo  physics stages 1-7
agent/mcns.py, agent/mcns_mapping.py, agent/verify_mcns.py   MCNS graph, name-based DN/leg-MN selectors, and Shiu CPU/GPU parity oracle
convex.mojo  GJK + EPA + perturbation contacts (MuJoCo engine_collision_gjk.c, mjc_Convex), float32 or -DCCD64=1 float64
convex_check.py, convex_test.mojo   narrowphase gate vs mj_collision (check.sh convex; the flight twin's primitive pairs and its near-rest poses, check.sh flight_contacts) and its CPU driver; hfield_check.py, hfield_test.mojo   the same for the exact heightfield (check.sh hfield)
convex_gpu.mojo, convex_bench.sh   narrowphase on the card: ns per pair class, VGPR/scratch/LDS;  row_diag.py   engine contact rows vs MuJoCo's efc rows (DUMP=1 build)
world.mojo, world.py   terrain function (model.world_height_grad runs world.mojo's CLI `world IN OUT N`; world.py is the oracle of `python -m kinestride.world diff`), world MJCFs (planes, cube, ramp)
assets.mojo, assets.py   world-part meshes (icosphere, box and ellipsoid geoms, edge splitter) for model.meshes; assets.py is the oracle of assets_check.py
city.mojo, city_check.mojo   endless city at 1 m units: 64 m cells keyed on the global cell (origin rebased by whole cells), lots with box or polar r(t) = R (1 + a cos(k t + phi)) footprints and setback tiers; city_hit (DDA, slab, polar march), city_height; `check.sh city` (point march, rebase at 10^6 m, streets, golden `city_golden.bin`, 4 mutants)
world_content.mojo/.py seeded plume descriptor and concentration sampler; check, GPU fixture and scenario TOML
game_state.mojo, game_events.py, game_state_check.py  compact game state, CPU event replay, standalone gate;  game_check.py  the engine hook's gate (`check.sh game`)
render.mojo  ray-march renderer on engine poses;  live.mojo, live.py  engine + renderer in one process
stand.sh     live check + upright on ramp and bumps;  check.sh  parity gate per stage;  bench.sh  timing at 1k/16k/65k;  phases.sh  per-phase timing
render_check.py, terrain_render_check.py, verify_render_leak.py   renderer gates
../probes/rt_bvh_probe.mojo   compile probe: Mojo emits RDNA3 `image_bvh_intersect_ray` (gfx1100); `llvm_intrinsic` cannot pass the <3 x float> operands, so the vector is built with `llvm.insertelement` and the call is `pop.call_llvm_intrinsic`. Compile only: node layout and descriptor not yet run on the card
spectate.mojo, spectate_check.py   live.mojo's spectator kernels on the card (`follow`: camera from the engine's own root pose rows; `hud`: bars, alpha 254) and their gate (hooks-off byte-equal, moving clip, HUD pixels, cost split, leak)
follow_camera.mojo   standalone root-and-radius framing for landscape and portrait targets; `hud.mojo`   pure game-bar and world-source mini-map layout helpers; `spectator_check.py` checks both and includes an off-center-target mutant (builds under `kinestride/.work/spectator_check/`)
models/      generated bodies (spider.py), eq_test.xml, eq_contact.xml, eq_mocap.xml (equality and mocap gate bodies), gravcomp_act.xml (gravcomp actuator-route gate body), tendon_spatial.xml (spatial tendon gate body), flight_export.py (Flybody flight MJCF, flight venv);  groom.py  grooming pairs
world_check.py, watch_world.py   plume-world gate; `live.py watch K world-game` (follow camera, game bars, plume text and mini-map, fps) and its check
fly_brain.mojo, fly_gait.mojo, fly_link3.py, fly_gait.py   BRAIN=3: DN readout + drives + FlyGym gait kernel (`fly_gait.mojo`, float64, parity-gated, with `fly_link.Gait`'s drive slew and trim) beside the resident brain of `kinestride/shiu_resident.mojo`; `fly_link3.py` runs the fly gates on one DeviceContext
fly_link.py   brain-driven fly in live (BRAIN=1 host transport: resident Shiu, flyloop's gait mapping, settled FlyGym start) and its gate (`check.sh flylink`)
spider_brain.mojo, spider_gait.py, spider_link.py   spider brain<->body link: gait kernel (`-DBRAIN=2`) and its numpy oracle, gate + timing (`live.mojo -DBRAIN=1|2`, `~/Projects/science/fly-connectome/docs/plans/SPIDER_BRAIN.md`)
../shiu_worker.mojo, ../shiu.py, ../verify_shiu_continuous.py   resident Shiu advance; deterministic split-run and reset-mutant gate
spider_jump.py   spider jump: ctrl script, gate (models/spider.py `build(jump=True)`)
eyes.mojo, spider_eyes.py   spider eyes (`live.mojo -DEYES=1`): per-animal eye cameras from the head pose, retinas [animal][eye][64x64] uint8, prey path from env.bin; eye table, analytic pinhole reference and gate (models/spider.py `build(prey=True)`)
../probes/s0.mojo, proto_*.mojo, newton_prof.sh   early prototypes, Newton rocprof probe
```

## Live view gotchas

- `spectate_check.build` names binaries by a hash of every engine `*.mojo` plus `sizes.txt`/`rsizes.txt` (efbfc0c): before, a re-export with other sizes reused the old binary, which exited 1 with no message at the first ctrl.

- `spectate_check.build` names binaries by a hash of every engine `*.mojo` plus `sizes.txt`/`rsizes.txt` (efbfc0c): before, a re-export with other sizes reused the old binary, which exited 1 with no message at the first ctrl.
- The fly MJCF ships `noslip_iterations="5"`; with it MuJoCo itself trembles (root |qvel| 1.09 vs 0.025 with 0). The live
  view exports noslip 0 but takes its standing pose from the shipped noslip-5 model: with noslip 0 the settling rollout
  lays the fly on its belly, and a lying fly passes a stillness gate (`74485d5`). `check.sh` parity keeps noslip 5.
- On terrain the start pose is lifted 0.05 mm above the highest ground under the animal's own bodies; the old "highest
  ground within 4 mm" dropped the fly 0.57 mm on a 0.1 ramp, and it slid 0.7 mm and lay down, in MuJoCo too.
- `live.py upright` clearance must exceed half the flat value and 0.2 mm, so a flat run that itself lies down fails.
- In the standing pose the head (rostrum) is 0.07 mm under the floor: only plane-mesh pairs collide, and it has none
  that push it out. The thorax body has two mesh geoms; the gate reads the first.

## Rules
A new body is a new MJCF file, not new code. A speed number counts only from a build that passes `check.sh step`.
Every GPU run goes through `gpu-wait`.

RK4: the shipped dt 1e-4 is unstable for explicit RK4 in MuJoCo itself (dof damping, h B / M = 5.9 against a limit of 2.8), so `check.sh rk4` runs
contact-free steps at `dt=1e-6` (a wrong weight still moves qacc_euler far above 1e-5) and contacts + servos at `dt=1e-7` (stiff contacts make MuJoCo's own ncon flip between its four forward passes at larger dt). MuJoCo's d after an RK4 step holds the LAST forward pass (X3), so the engine's rows do too; qH is stale there and not compared.

CG and PGS (`check.sh solver`): PGS stops unconverged at 100 sweeps in MuJoCo, so its result depends on the visit order (a fixed-seed pcg32 Fisher-Yates shuffle, ported exactly, one island per kinematic tree with its own shuffle) and on `qacc_warmstart`: the oracle there uses a fresh MjData per state (`model.reference_fresh`). CG's two stop tests fire after ~37 iterations in float32 against ~80 in float64 and leave qacc 2e-3 off, so CG runs its iteration cap (line search ends it at alpha 0): about 2.4x Newton's iterations. The two-fly island test runs at `REL=2e-4` (merged vs separate worlds differ by 7e-5 in MuJoCo itself, Newton included); the PGS island mutant must fail it.

`check.sh implicit` and `check.sh integrator` state their own coverage and runs on airborne states: with contact rows, active-set float32 noise (qfrc_constraint
1e-4, qacc_euler 2e-5 at kv 0.01, over the floor-lifted bars) hides the integrator; the contact result is printed as MEASURED. `reject_check.py` gates the export refusals.
