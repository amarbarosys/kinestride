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
| Equality | connect, weld, joint, tendon equalities, runtime `eq_active` | PASS, `equality`, `equality_contact` (see `docs/ENGINE.md`; `9d593b6`, `e7260c2`) |
| Mocap | mocap bodies | PASS, `mocap` (`docs/ENGINE.md`) |
| Gravcomp | `mj_gravcomp`, actuator-level gravcomp | PASS, `gravcomp` (`docs/ENGINE.md`) |
| Spatial tendons | `mj_tendon`, `mju_wrap` | `docs/ENGINE.md` stage table |

Tolerances: each field is checked against MuJoCo within 1e-5 of its scale; where float32 cannot reach that, the bar is the measured float32 floor (for example qfrc_constraint 1.9e-04 against a 2.1e-04 bar in the Floor stage).

## Solvers

- Newton with the MuJoCo iteration count (S2 probe, `01c4c57`). The float32 stopping rules that cut line-search cost are in `docs/ENGINE_OPT.md` (Step 0, `32d27c2`).
- PGS over equality, friction-loss, limit, elliptic and convex rows, with islands across trees (`2e87a24`).
- CG over the same rows (`eafad87`).
- Elliptic cone under RK4 and implicitfast (`3a897e3`).
- Torsional and rolling friction, condim 4 and 6, pyramidal and elliptic (`612ff59`).
- Dof and tendon friction-loss rows, Huber cost in Newton (`b631139`, `9a94224`).
- Direct solref on equality and joint/tendon limit rows (`c1e6cdf`). Joint actuator force limits (`5686d69`).

## Contacts

- Plane-mesh feet (S2 probe, `01c4c57`).
- Margin-0 box-mesh and mesh-mesh contacts by MuJoCo's polygon clipping (`2f552a2`). Mesh-mesh through `mjc_Convex` (`05d8954`).
- Box-mesh and ellipsoid-mesh through the native GJK and EPA, with the perturbation search for 0, 1 or 5 contacts (`docs/CONVEX_CONTACT.md`).
- Exact heightfield contact, one GJK and EPA per cell prism, up to 50 contacts per pair (`docs/CONVEX_CONTACT.md`). A tangent-plane approximation was planned first (`docs/HFIELD_CONTACT.md`); the exact path is the one the convex doc records as measured.
- Contact pairs across two trees in one model (`9016296`) and between two articulated animals (`76873dc`).
- Contact list out for audio or game hooks (`-DCLIST=k`, `b26adc4`; `docs/API.md`).
- Not ported: plane-capsule, and the flight model's automatic body-body contacts (capsule, cylinder, sphere supports are missing; `docs/CONVEX_CONTACT.md`). Noslip and the contact pairs other than the feet are still open (`docs/ENGINE.md`).

## Batch sizes and throughput

Per-launch numbers from the engine docs and optimisation baseline (`docs/ENGINE_OPT.md`, baseline at `01c4c57`, 2026-10-03):

| Batch | Per launch | Per animal-step | vs MuJoCo CPU (70,582 steps/s) |
|---|---:|---:|---:|
| 1,024 | 32.1 ms | 31.4 us | 0.45x |
| 16,384 | 83.4 ms | 5.09 us | 2.8x |
| 65,536 | 133 ms | 2.03 us | 7.0x |

Later: the S3 stage runs 0.64 us per animal-step at 65,536 (`dd2f42c`, `docs/ENGINE.md`). Step 0 of the optimisation plan cut the 1,024-animal launch from 32.1 ms to 18.6 ms (`32d27c2`). Re-measured 2026-11-18 at stage 7: 1,024 animals 22.3 ms per launch (0.65x MuJoCo), 16,384 38.6 ms (6.0x), 65,536 84.7 ms, 1.29 us per animal-step (11.0x); table in `docs/VS_MUJOCO.md`.

The Godot crowd frame timings at 1k, 16k and 65k are in `2884ab4` (no numbers are copied here).

## Motion and tracking

- Any BVH skeleton is retargeted onto one 21-body, 70 kg character, with spike repair and limb planting (`docs/MOTION.md`).
- Tracking matches MuJoCo's own run of the same loop to 1.2e-7 m mean body position over 800 frames, 64 animals (`check.sh track`, `docs/MOTION.md`).
- Dataset driver: every animal its own clip, push and fall (`382f1b0`, `docs/MOTION.md`).
- Retarget measured at 2.8M to 3.65M frames/s, against 23.8k to 25.1k skeletons/s in Godot 4.7.2 on the same machine (`docs/MOTION.md`).

## Muscles

A compact excitation-driven muscle model on the fixed-tendon path: activation and fatigue with exact exponential integration, bounded force curves, MJCF annotation `muscle_<actuator>` (`docs/MUSCLE.md`, `a2b9851`, `98d4f63`). It is not MuJoCo's gain/bias/dyntype model. Game stamina drains from muscle work (`77cc7f2`).

## Checks

- `check.sh STAGE` per stage, with a planted mutant that must fail. Mutants caught are listed per stage in `docs/ENGINE.md`.
- `check.sh convex`, `hfield`, `arena`, `terrain_exact`, `convexstep`, each with mutants (`docs/CONVEX_CONTACT.md`).
- `check.sh bvh` (14 planted bugs, each caught), `motion`, `track`, `dataset`, `clist`, `gravcomp`, `equality`, `mocap` (`docs/MOTION.md`, `docs/API.md`).
- `check_all` builds every mutant on the CPU first (`5c8a4e6`).
- `smoke.sh`, a CPU check a fresh checkout can run (`bf489e6`).

## Rendering and live loop

- Renderer draws the engine's poses (`docs/RENDERER.md`). Sun shadows, SSAO, AgX tone mapping and bloom are in the commit log (`ccfb09d`, `05d77fe`, `08c1853`).
- Live loop: `viewer.Viewer` with the app's own controller (`docs/API.md`, `c657634`).
- Window and UI: SDL3 window, FreeType glyph atlas, immediate-mode UI on the GPU (`6066ba9`, `9e146f9`, `c6b53a6`).

## Integrations

- Godot: a C ABI over the engine (create, step, poses, ctrl, `c7af891`), a GPU launch from inside Godot (`fb579b8`), KinestrideWorld and NpcMind classes (`d31eddd`), MultiMesh transforms (`9229756`), npc-mind brains as a C shared library (`02767ec`), crowd and NPC scenes (`5eeaef1`).
- Blender: `.blend` load through a cached export (`49c8cc0`), a 14-body humanoid saved as `.blend` (`6050930`).
- npc-mind NPCs walk as spider bodies (`e465f99`).

## Platform

Tested on gfx1100 (RDNA3) only so far. Build: Mojo 1.2.0, MAX 26.7.0 (`6443f95`).

> Commits before `7a119b3` (2026-10-09) are in the fly-connectome repo, where the engine started; kinestride history begins at `7a119b3`.
