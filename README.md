# Kinestride

65,536 articulated bodies, each with 74 degrees of freedom, contacts and 50 actuators, step together in 66.7 ms
on one AMD RX 7900 XTX. Kinestride is batched GPU physics in Mojo: one GPU thread per body, any MuJoCo MJCF model
as input, and every stage checked against MuJoCo 3.9.0 with a planted bug that the check must catch.

## Any MJCF model becomes its own binary

`kinestride/model.py export` flattens a MuJoCo model into two tables (`mi.bin` int32, `mf.bin` float32). The sizes
become compile-time constants, so a fly, a spider or a game character each compiles a kernel shaped for it.
A body built in Blender gets here through the blender-mjcf add-on (parented meshes become bodies, named Empties
become joints; armatures, masses and actuators are not exported, so those are added to the XML by hand).
Export refuses what the engine cannot simulate instead of dropping it: flex,
flex equalities, coupling across two trees outside Newton + pyramidal + Euler.

| area | ported and checked |
|---|---|
| dynamics | kinematics, composite inertia, sparse mass-matrix factor, RNE bias, springs, dampers, gravity compensation |
| integrators | Euler, RK4, implicit, implicitfast (+ midpoint for free bodies) |
| contacts | plane and heightfield vs mesh; sphere, capsule, ellipsoid, cylinder; box and convex meshes, mesh vs mesh (GJK + EPA, multi-contact); automatic body-body pairs; pyramidal and elliptic cones; noslip |
| constraints | Newton (default), CG, PGS; connect, weld, joint equalities; joint and tendon limits; standard and direct solref on contact, equality and limit rows; equalities switched on and off per animal at run time (eq_active); pairs, tendons and equalities across trees |
| actuators | joint servos, fixed and spatial tendons, adhesion; joint-level force limits (actuatorfrcrange) |
| more | push input (xfrc_applied), fluid forces, per-body sleep, mocap bodies, a game layer (energy, health, stamina), a ray-marching renderer with eye cameras |

## The step is limited by memory bandwidth, not arithmetic

At 65,536 bodies a launch reads 40 GB and writes 16 GB of DRAM, about 87 % of the card's bandwidth. Each body
reads 611 KB per step against a 141 KB footprint, because the Newton solver re-reads its factors and vectors every
iteration (`docs/ENGINE_OPT.md`, byte budget). Full implicit integration costs 85 KB of scratch per body
(5.6 GB at 65,536), so it is not yet a mode for large crowds.

## Running it

```
tools/smoke.sh                     # CPU only, no assets: doc links + export + full-step build (29 s cold)
python -m kinestride.assets_check  # CPU: assets.mojo (world-part meshes) vs its oracle assets.py, 10 cases exact
tools/assets.sh                    # test bodies into .work/ (fly, flight model, spider; kept out of git)
kinestride/check.sh step           # one stage vs MuJoCo (under gpu-wait on this machine)
tools/check_all.sh                 # every stage, one preemptible gpu-wait job each
kinestride/bench.sh build MJCF 7   # then: gpu-wait run --timeout 1200 -- kinestride/bench.sh run MJCF 7
tools/blender_check.sh            # Blender rig -> blender-mjcf add-on -> check step (under gpu-wait)
```

`.venv` comes from `pyproject.toml` (uv: Mojo 1.1.0, MAX 26.6.0, MuJoCo 3.9.0). The MuJoCo oracle is
`~/iTools/science/mj-oracle`.

From another project: `uv pip install -e ~/Projects/mojo/kinestride` (Python: `kinestride.model`), and Mojo builds
with `-I ~/Projects/mojo/kinestride/kinestride` (`import smooth`, `import render`, `import world`). fly-connectome is
the first consumer: its connectome brains, live viewer and game scenes drive Kinestride bodies.

## Watching a body live

`kinestride/viewer.mojo` holds the live loop's engine and renderer (one DeviceContext, poses rendered without a copy);
`kinestride/view.mojo` runs it with no brain, ctrl held, frames to stdout for mpv. An app keeps its own loop and calls
`Viewer.step` and `Viewer.frame` around its own controller: fly-connectome's `live.mojo` drives brains this way
(`check.sh view`: the loop equals smooth and render byte for byte).

## What is not here yet

A window and input for game use (fly-connectome has one, through mojo-imkit); the engine has only run on gfx1100.

Reference: `docs/API.md` (the stable surface: files, rows, step), `docs/ENGINE_REFERENCE.md` (stage by stage), `docs/ENGINE_OPT.md` (performance history),
`docs/RENDERER.md`, `docs/HFIELD_CONTACT.md`, port queue `kinestride/port_queue.tsv`. License notes: `NOTICE`.

Port routine (porting a MuJoCo stage, gates, timing): skill `.claude/skills/body-engine/SKILL.md`.
Status and next steps: board `~/Brain/kinestride/whiteboard.md` (symlink `.claude/checkpoints/whiteboard.md`, gitignored).
