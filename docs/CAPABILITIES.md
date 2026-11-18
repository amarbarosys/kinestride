# Kinestride capabilities

What the engine does today, with the commit or doc each claim comes from. Stages and gates are checked against MuJoCo 3.9.0. Limits are in `VS_MUJOCO.md`.

## Engine

A batched GPU body engine. MuJoCo's own functions are ported into Mojo for a fixed body, one thread (or wave) per animal, so every rule can change. The actuated fly (44 joint servos, 6 adhesion pads) runs on the GPU. An 8-leg spider test body covers the smooth stages (`docs/ENGINE.md`).

Build path (`docs/API.md`):

```
python -m kinestride.model export MJCF OUT
mojo build $(cat OUT/sizes.txt) -DFLIES=N -DSTAGE=7 kinestride/smooth.mojo -o step
```

Each model becomes two tables (`mi.bin`, `mf.bin`) and its own binary. Export refuses what the engine cannot simulate.

## Stages ported

| Stage | MuJoCo 3.9.0 function | Status (commit) |
|---|---|---|
| S1.1 to S1.5 | kinematics, comPos/crb, factorM, comVel/passive/rne, Euler | PASS, `ccd9695`, `019783a`, `83469eb`, `fc0abdb` |
| S2 probe | plane-mesh feet (6 pairs), pyramidal cone, Newton solve | PASS, `01c4c57` (2.0 us per step) |
| S3 | transmission, actuators: 44 joint servos, 6 adhesion bodies | PASS, `dd2f42c` (0.64 us per step at 65,536) |
| Floor | all 55 plane-mesh pairs | PASS, `00a58e6` |
| Loop | K steps per launch, state kept on the card | PASS, `c5c478c` (K=10 byte-equal to 10 x K=1) |
| Origin | floating origin per animal | PASS, `a03e700` |
| Cube | 40 mm cube of 6 planes, 330 pairs | PASS, `5acde4e` |
| Spider | generated salticid body | PARTIAL, `615d2af` (contact/actuated qfrc_constraint 1.5e-05 to 5.2e-05; cause open) |
| Equality | connect, weld, joint, tendon equalities, runtime `eq_active` | PASS, `equality`, `equality_contact` (see `docs/ENGINE.md`; `3b67c67`, `ca9c0b9`) |
| Mocap | mocap bodies | PASS, `mocap` (`docs/ENGINE.md`) |
| Gravcomp | `mj_gravcomp`, actuator-level gravcomp | PASS, `gravcomp` (`docs/ENGINE.md`) |
| Spatial tendons | `mj_tendon`, `mju_wrap` | `docs/ENGINE.md` stage table |

Tolerances: each field is checked against MuJoCo within 1e-5 of its scale; where float32 cannot reach that, the bar is the measured float32 floor (for example qfrc_constraint 1.9e-04 against a 2.1e-04 bar in the Floor stage).

## Solvers

- Newton with the MuJoCo iteration count (S2 probe, `01c4c57`). The float32 stopping rules that cut line-search cost are in `docs/ENGINE_OPT.md` (Step 0, `32d27c2`).
- PGS over equality, friction-loss, limit, elliptic and convex rows, with islands across trees (`cdb3c92`).
- CG over the same rows (`d4d61a2`).
- Elliptic cone under RK4 and implicitfast (`13ab72f`).
- Torsional and rolling friction, condim 4 and 6, pyramidal and elliptic (`df46fbc`).
- Dof and tendon friction-loss rows, Huber cost in Newton (`53e6000`, `acf6154`).
- Direct solref on equality and joint/tendon limit rows (`a5912f3`). Joint actuator force limits (`5488855`).

## Contacts

- Plane-mesh feet (S2 probe, `01c4c57`).
- Margin-0 box-mesh and mesh-mesh contacts by MuJoCo's polygon clipping (`0548782`). Mesh-mesh through `mjc_Convex` (`22643a5`).
- Box-mesh and ellipsoid-mesh through the native GJK and EPA, with the perturbation search for 0, 1 or 5 contacts (`docs/CONVEX_CONTACT.md`).
- Exact heightfield contact, one GJK and EPA per cell prism, up to 50 contacts per pair (`docs/CONVEX_CONTACT.md`). A tangent-plane approximation was planned first (`docs/HFIELD_CONTACT.md`); the exact path is the one the convex doc records as measured.
- Contact pairs across two trees in one model (`4fcfec8`) and between two articulated animals (`7f705b5`).
- Contact list out for audio or game hooks (`-DCLIST=k`, `5234636`; `docs/API.md`).
- Not ported: plane-capsule, and the flight model's automatic body-body contacts (capsule, cylinder, sphere supports are missing; `docs/CONVEX_CONTACT.md`). Noslip and the contact pairs other than the feet are still open (`docs/ENGINE.md`).

## Batch sizes and throughput

Per-launch numbers from the engine docs and optimisation baseline (`docs/ENGINE_OPT.md`, baseline at `01c4c57`, 2026-10-03):

| Batch | Per launch | Per animal-step | vs MuJoCo CPU (70,582 steps/s) |
|---|---:|---:|---:|
| 1,024 | 32.1 ms | 31.4 us | 0.45x |
| 16,384 | 83.4 ms | 5.09 us | 2.8x |
| 65,536 | 133 ms | 2.03 us | 7.0x |

Later: the S3 stage runs 0.64 us per animal-step at 65,536 (`dd2f42c`, `docs/ENGINE.md`). Step 0 of the optimisation plan cut the 1,024-animal launch from 32.1 ms to 18.6 ms (`32d27c2`). Re-measured 2026-11-18 at stage 7: 1,024 animals 22.3 ms per launch (0.65x MuJoCo), 16,384 38.6 ms (6.0x), 65,536 84.7 ms, 1.29 us per animal-step (11.0x); table in `docs/VS_MUJOCO.md`.

The Godot crowd frame timings at 1k, 16k and 65k are in `a1f1cc3` (no numbers are copied here).

## Motion and tracking

- Any BVH skeleton is retargeted onto one 21-body, 70 kg character, with spike repair and limb planting (`docs/MOTION.md`).
- Tracking matches MuJoCo's own run of the same loop to 1.2e-7 m mean body position over 800 frames, 64 animals (`check.sh track`, `docs/MOTION.md`).
- Dataset driver: every animal its own clip, push and fall (`e07afc5`, `docs/MOTION.md`).
- Retarget measured at 2.8M to 3.65M frames/s, against 23.8k to 25.1k skeletons/s in Godot 4.7.2 on the same machine (`docs/MOTION.md`).

## Muscles

A compact excitation-driven muscle model on the fixed-tendon path: activation and fatigue with exact exponential integration, bounded force curves, MJCF annotation `muscle_<actuator>` (`docs/MUSCLE.md`, `1876173`, `1b10bf6`). It is not MuJoCo's gain/bias/dyntype model. Game stamina drains from muscle work (`b7d24bf`).

## Checks

- `check.sh STAGE` per stage, with a planted mutant that must fail. Mutants caught are listed per stage in `docs/ENGINE.md`.
- `check.sh convex`, `hfield`, `arena`, `terrain_exact`, `convexstep`, each with mutants (`docs/CONVEX_CONTACT.md`).
- `check.sh bvh` (14 planted bugs, each caught), `motion`, `track`, `dataset`, `clist`, `gravcomp`, `equality`, `mocap` (`docs/MOTION.md`, `docs/API.md`).
- `check_all` builds every mutant on the CPU first (`fead62e`).
- `smoke.sh`, a CPU check a fresh checkout can run (`1a2f5b3`).

## Rendering and live loop

- Renderer draws the engine's poses (`docs/RENDERER.md`). Sun shadows, SSAO, AgX tone mapping and bloom are in the commit log (`385de27`, `b4a185f`, `2117e26`).
- Live loop: `viewer.Viewer` with the app's own controller (`docs/API.md`, `8dff5a6`).
- Window and UI: SDL3 window, FreeType glyph atlas, immediate-mode UI on the GPU (`bea8dfe`, `73ee055`, `fc74f14`).

## Integrations

- Godot: a C ABI over the engine (create, step, poses, ctrl, `c6b7f4d`), a GPU launch from inside Godot (`dbea402`), KinestrideWorld and NpcMind classes (`7373a16`), MultiMesh transforms (`2853ec8`), npc-mind brains as a C shared library (`e682e31`), crowd and NPC scenes (`c3eef4b`).
- Blender: `.blend` load through a cached export (`1addcc3`), a 14-body humanoid saved as `.blend` (`984c2d5`).
- npc-mind NPCs walk as spider bodies (`86573e2`).

## Platform

Tested on gfx1100 (RDNA3) only so far. Build: Mojo 1.2.0, MAX 26.7.0 (`a6aff17`).

> Commits before `e788c9b` (2026-10-09) are in the fly-connectome repo, where the engine started; kinestride history begins at `e788c9b`.
