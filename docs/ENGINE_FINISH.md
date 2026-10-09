# Finish the body engine (2026-10-03)

Four gaps keep the engine at PARTIAL, and only 7 of the 496 missing contact pairs matter for the worlds we run:
the rest are the legacy arena's walls and stones, already replaced by planes, terrain and cones.

## Context
After v0.2.0 the engine (MuJoCo 3.9.0 port, `kinestride/smooth.mojo`) lists four gaps in
`docs/CAPABILITIES.md`: noslip, the non-foot contact pairs, one floating origin per model, and a slow, static live
view. Scope agreed 2026-10-03: grooming pairs plus worlds only (the 440 box/ellipsoid wall and stone pairs need
GJK/EPA), noslip, per-animal origin, live speed and a following camera.

Fly pairs: 55 mesh-plane (ported), 275 box-mesh and 165 ellipsoid-mesh (legacy arena, MuJoCo-only), 7
sphere-sphere (grooming). Spider: 50 mesh-plane (ported).

## Steps
Each step follows the body-engine skill: port, gate, mutant, bench, commit with numbers.
```
1 per-animal origin ─► 2 grooming pairs ─► 3 noslip ─► 4 live speed + follow cam
  (S-M)                (XS-S)              (M)          (M)
```
1. **Per-animal origin.** DONE (see git log: per-animal origin, 8 and 200 mm PASS).
   Plan text: `origin()` today returns the first free joint for the whole model. Each body uses the
   origin of its own tree (`body_rootid` to its free joint's qpos; world-attached trees and world geoms use 0).
   Touches `step()` kinematics, `contacts()` (a world geom's position expressed in the animal's frame) and
   `absolute()`. Gate: `check.sh world` with the spider at 8 mm and at 200 mm, all 25 fields PASS, every existing
   stage still PASS; mutant (the spider uses the fly's origin) fails at 200 mm. Correction: the float32 INPUT floor
   cannot drop, it comes from qpos stored as float32 world coordinates; the origin removes the engine's own error
   on top of it.
2. **Grooming pairs.** DONE (`120dfb6`, `eaae6a8`: check.sh groom PASS, H gets its own fill pattern). Port `mjc_SphereSphere` (engine_collision_primitive.c) as a second pair kind: one
   contact, normal along the centre line, condim 3, the same impedance path, a Jacobian spanning two dof chains.
   Export refuses any other unported pair kind loudly (today it filters silently). Gate: `check.sh actuated` with
   the 7 grooming pairs live, on states where legs touch the head. Mutant: normal sign.
3. **Noslip.** DONE (`59abe8d`, `30e3575`: all stages PASS with noslip 5; the change term must use d1 = -d0 in float32). Port `mj_solNoSlip` (engine_solver.c `solNoSlip`): projected Gauss-Seidel over each contact's
   friction rows after Newton, `ARdiaginv` from the existing factor, `noslip_iterations` from the model, then qacc
   from the new forces. Remove the "noslip 0" override from every oracle. Gate: `check.sh actuated` and `world`
   with noslip 5 as the fly model ships. Mutant: skip the projection. Bench the cost of 5 sweeps per step.
4. **Live speed and follow camera.** Profile one live frame (`rocprof-kernels`, `phases.sh`) for the engine and
   render shares, then cut in order: one block per animal for small batches (BODY_ENGINE_OPT step 2 measured 10x
   at 1k animals), render only what changed. Camera target and distance from the animals' positions on the card
   each frame, smoothed, no host round trip. Gate: at least 25 fps on the world at K=10, `live check` frame and
   leak on the world build, frames 1, 300 and 3000 looked at.

5. **Integrators and solvers (lane-solver, 2026-10-05).** Comptime-selected from opt.integrator / opt.solver, binaries for
   other models unchanged. Done: implicitfast (`mj_implicitSkip`, engine_forward.c:1864). For the fly it equals Euler's qH
   (damping only, servos kv 0), so the gate overrides kv (`ENGINE_OPT=integrator=3,kv=0.01`) and uses airborne states
   (contact rows add float32 active-set noise above the bars; reported MEASURED). Refused at export: tendon damping, fluid.
   Full implicit (lane-constr unit 5, `mj_implicit`: `mjd_rne_vel` + LU, `-DINTEGRATOR=2`) and the implicitfast midpoint of standalone free bodies
   (`mj_midpoint`) are ported, `check.sh implicit`. Not needed: sleep (enable bit 16 off in every model).
   RK4 (`mj_RungeKutta`): `step` runs four times from `smooth()`, X1..X3 live in the qpos/qvel tensors, X0 restored after the last pass; MuJoCo's
   d holds the last forward pass, so does the engine (qH stale, not compared). Gate dt 1e-6 contact-free, 1e-7 with contacts (see README).
   CG (`mj_solCG`, Polak-Ribiere over the Newton line search, preconditioner M^-1) and PGS (`mj_solPGS`: pcg32 shuffle, one island per tree, warmstart
   from qacc_warmstart or zero by dual cost) are comptime `SOLVER`; PGS leaves Newton's representation (OST, OJAR, OQFC) so noslip continues unchanged.
   Sleep/island filter: not needed (enable bit 16 off in all our models); constraint islands are only solver-convergence structure and PGS follows them.

## Out of scope
GJK/EPA (box, ellipsoid, mesh-mesh) and the legacy arena's 440 pairs; MuJoCo's merged-vs-separate spider force
difference (7.3e-5, open, logged in WORLD.md).

## Verification
Per step as above; at the end every `check.sh` stage, `world`, `terrain`, `live.py upright` on the world build, the
live world looked at following both animals, and release v0.3.0.
