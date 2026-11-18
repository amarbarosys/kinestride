# Kinestride

65,536 articulated bodies, each with 74 degrees of freedom, contacts and 50 actuators, step together in 66.7 ms
on one AMD RX 7900 XTX. Kinestride is batched GPU physics in Mojo: one GPU thread per body, any MuJoCo MJCF model
as input, and every stage checked against MuJoCo 3.9.0 with a planted bug that the check must catch.

## What it is

- **Batched.** One GPU thread per body. A launch steps the whole crowd: 66.7 ms at 65,536 bodies
  (`docs/releases/v0.13.0.md`).
- **Any MJCF model.** `kinestride/model.py export` flattens a MuJoCo model into two tables (`mi.bin` int32,
  `mf.bin` float32). The sizes become compile-time constants, so each body compiles a kernel shaped for it.
  A `.blend` path loads directly through the blender-mjcf add-on (`kinestride/blend.py`, cached under `.work/`).
  Export refuses what the engine cannot simulate instead of dropping it.
- **Checked.** Each ported MuJoCo function is a file named after its original and cites the source it follows.
  Each output field must agree with MuJoCo within 1e-5 of its scale on random states, and a planted mutant of
  the function must fail the gate.

| area | ported and checked |
|---|---|
| dynamics | kinematics, composite inertia, sparse mass-matrix factor, RNE bias, springs, dampers, gravity compensation |
| integrators | Euler, RK4, implicit, implicitfast (+ midpoint for free bodies) |
| contacts | plane and heightfield vs mesh; sphere, capsule, ellipsoid, cylinder; box and convex meshes, mesh vs mesh (GJK + EPA, multi-contact); automatic body-body pairs; pyramidal and elliptic cones |
| constraints | Newton (default), CG, PGS; connect, weld, joint equalities; joint and tendon limits; solref on contact, equality and limit rows; equalities switched on and off per body at run time |
| actuators | joint servos, fixed and spatial tendons, adhesion; joint-level force limits |
| more | push input, per-body sleep, mocap bodies, a ray-marching renderer with eye cameras |

Not yet ported: noslip, and contact pairs beyond those listed above. The stage-by-stage status is in
`docs/ENGINE_REFERENCE.md`; the stable surface (files, rows, step) is `docs/API.md`.

## Quick start

Requirements: Linux, an AMD RDNA3 GPU with ROCm (gfx1100 is the only target run so far), and the Python
environment from `pyproject.toml` (uv: Mojo 1.2.0, MAX 26.7.0, MuJoCo 3.9.0).

```
tools/smoke.sh                     # CPU only, no assets: doc links, export, full-step build
kinestride/check.sh step           # GPU: one stage against MuJoCo, plus its planted mutant
tools/assets.sh                    # test bodies into .work/ (not in git)
tools/check_all.sh                 # GPU: every stage, one job each
```

`check.sh` takes `.work/check.lock` per worktree, so run GPU stages one at a time. A passing stage prints
`PASS`; a failing one prints `FAIL <step>` and exits non-zero.

From another project: `uv pip install -e /path/to/kinestride` (Python: `kinestride.model`), and Mojo builds
with `-I /path/to/kinestride/kinestride` (`import smooth`, `import render`, `import world`).

## Benchmarks

`kinestride/bench.sh build MJCF 7`, then `kinestride/bench.sh run MJCF 7` on the GPU. Batch sizes are 1,024,
16,384 and 65,536 bodies. Stage 5 is the contact-free step; stage 6 adds feet, contacts and Newton. Run
`check.sh` first: a speed number from an unchecked build is not reported.

| figure | value | source |
|---|---|---|
| 65,536 bodies, full step, one launch | 66.7 ms | `docs/releases/v0.13.0.md` |
| 65,536 bodies, actuated fly (6 feet, 44 servos, 6 adhesion pads) | 0.64 us per body-step | `docs/ENGINE.md` |
| same workload, MuJoCo on six CPU workers | 22 times slower | `docs/ENGINE.md` |

The optimisation history, with the byte budget and the rejected steps, is in `docs/ENGINE_OPT.md`.

## How it compares to MuJoCo

MuJoCo is the reference and the oracle. Kinestride ports its functions one at a time and checks each against it.
Scope, speed and when each one fits are in [docs/VS_MUJOCO.md](docs/VS_MUJOCO.md).

## Where it came from

Kinestride began as the GPU body engine of an insect-brain project. The decision to port MuJoCo's functions
by hand, after an earlier GPU route was closed, is in [docs/HISTORY.md](docs/HISTORY.md).

## Layout

```
kinestride/   engine: one .mojo file per MuJoCo function, model.py (export), check.sh, bench.sh
kinestride/models/   test MJCFs kept in git (spider_test.xml); fly and other bodies come from tools/assets.sh
tools/        CPU smoke, assets, check_all, Blender rig, dataset helpers
docs/         API, engine reference, optimisation history, renderer, releases
skills/       agent skill for porting a stage (body-engine)
```

What the engine runs, and what it refuses, is listed in [docs/CAPABILITIES.md](docs/CAPABILITIES.md).

## For coding agents

Porting a MuJoCo stage, writing a stage gate, or timing the engine: read `skills/body-engine/SKILL.md`.
Claude Code: `ln -s "$PWD/skills/body-engine" .claude/skills/body-engine`.

## License

Apache-2.0, see `LICENSE`. Portions are translated from MuJoCo (Apache-2.0, DeepMind); see `NOTICE`.
