# Kinestride history

## Origin

Kinestride came out of fly-connectome, a fly body simulation. The body engine was a direct port of MuJoCo's functions into Mojo for a fixed body, one thread or wave per animal (`docs/ENGINE.md`).

Before this repo existed, the GPU track was checked on MuJoCo Warp. On 2026-10-02 that track was closed: MuJoCo Warp on gfx1100 reached correct physics but stayed below six CPU MuJoCo workers at 70,582 steps/s (`docs/ENGINE.md`, citing the Brain note `2026-10-03-gpu-body-port.md`). The engine port started from there.

The baseline for the optimisation work was measured on 2026-10-03 at `01c4c57`, the S2 probe (`docs/ENGINE_OPT.md`). That commit is cited in the docs but is not among the first commits of this repo. The repo log begins at the root commit `e788c9b`, on 2026-10-09.

## Repo start (2026-10-09)

- `e788c9b` feat: Kinestride, batched GPU physics for articulated bodies in Mojo. Root commit.
- `285eff9` docs: body-engine skill moves here from fly-connectome.
- `4453051` fix: meshes(world=...) imported kinestride.assets, which stayed in fly-connectome.
- `3afda6e` fix(tools): assets.sh stops copying MuJoCo sources from fly-connectome.
- `4b1e666` fix(tools): flight_export takes the fly-connectome checkout as an argument.
- `a8fa82c` docs(release): v0.1.0, the first release as its own repo.
- `e48ceca` docs(release): v0.2.0, the look round.
- `024a05d` docs(release): v0.3.0, the fly looks like MuJoCo's.

## Releases

Tags from `git tag` (all tags use the form `vX.Y.Z`; each was tagged with the message "kinestride vX.Y.Z"):

| Tag | Date | Release commit | Note |
|---|---|---|---|
| v0.1.0 | 2026-10-09 | `a8fa82c` | first release as its own repo |
| v0.2.0 | 2026-10-09 | `e48ceca` | the look round |
| v0.3.0 | 2026-10-09 | `024a05d` | the fly looks like MuJoCo's |
| v0.4.0 to v0.11.0 | not in the sources | | tags exist; release commits not cited here |
| v0.8.0 | not in the sources | | motion check cited in `docs/MOTION.md` (v0.8.0: 0.090 m at ASSIST=0.02; 31-body character 61 ms) |
| v0.12.0 | 2026-10-23 | `300a57e` | |
| v0.12.1 | 2026-10-26 | `0d74b91` | |
| v0.13.0 | 2026-10-26 | `4717e20` | |

Dates and hashes for v0.12.0 to v0.13.0 are from the release commits in `git log`.

## Milestones

- **2026-10-09**: repo start, first release (v0.1.0), look round (v0.2.0), render and GUI (v0.3.0). Tools added: `doclinks.sh` (`a0ebadf`), `smoke.sh` (`1a2f5b3`), `blender_check.sh` (`574833f`).
- **2026-10-12**: motion and tracking land. BVH to MJCF character (`f0be825`), tracking loop (`07ead2e`), dataset driver (`e07afc5`). Engine contacts across two trees (`4fcfec8`), mesh-mesh pairs (`22643a5`), push input (`ea849f3`), per-animal equality switch (`3b67c67`). City renderer (`d58505a`).
- **2026-10-16 to 10-17**: Living World (`aeb1e15`, `32a3fe0`, six species in `32716d6`). Stateful tendon muscle engine (`1876173`) and muscle gating (`1b10bf6`), game stamina from muscle work (`b7d24bf`). Tone mapping and sky scattering (`b4a185f`, `59621b4`).
- **2026-10-22**: smooth macro surface (`5d1d950`), trails, gulls, aurora. Table-backed heightfield HTAB (`061b052`).
- **2026-10-23**: contacts and solvers. Equalities, friction-loss rows, PGS over all row kinds (`cdb3c92`), CG (`d4d61a2`), elliptic cone under RK4 (`13ab72f`), margin-0 box-mesh and mesh-mesh by polygon clipping (`0548782`). `bisect_stage.sh` (`02eb64f`).
- **2026-10-26**: union walk tried, measured 16 to 27% slower, and reverted (`0ff4946`, reverts `c74099c`, `cab9511`; `6a17b3f` records the kernel at 128 VGPRs). npc-mind NPCs walk as spider bodies (`86573e2`). npc_live peak RSS down from 1,580 MB to 184 MB (`624a27a`).
- **2026-10-27**: snapshot and restore of qpos, qvel, act, warm start (`eeebaaf`). Contact list out and contact events ring (`5234636`).
- **2026-10-29**: build moved to Mojo 1.2.0 and MAX 26.7.0 (`a6aff17`). Every check.sh stage in its own tree copy, GPU tests in parallel (`4098ed9`).
- **2026-10-30**: Godot integration. Kinestride GPU launch from inside Godot (`dbea402`), C ABI (`c6b7f4d`), KinestrideWorld and NpcMind classes (`7373a16`), MultiMesh transforms (`2853ec8`), npc-mind brains as a C shared library (`e682e31`), crowd and NPC scenes (`c3eef4b`), crowd frame timings at 1k / 16k / 65k (`a1f1cc3`). Blender rig as a 14-body humanoid saved as `.blend` (`984c2d5`).

## Engine stage milestones (from `docs/ENGINE.md`)

- S1.1 to S1.5: kinematics, CoM and CRB, factorM, passive and RNE, Euler (`ccd9695`, `019783a`, `83469eb`, `fc0abdb`).
- S2 probe: plane-mesh feet, Newton (`01c4c57`, 2.0 us per step).
- S3: 44 joint servos, 6 adhesion bodies, 0.64 us per step at 65,536 (`dd2f42c`).
- Floor, Loop, Origin, Cube, Spider (partial), equality, mocap, gravcomp: see `docs/ENGINE.md` for commits.

## Still open

- Noslip and the contact pairs other than the feet.
- Spider stage (partial, cause open).
- cone_hfield (red).
- Small-batch performance at 1,024 animals (`docs/ENGINE_OPT.md`).
- Planting speed in the motion pipeline, against a 100x target (`docs/MOTION.md`).

> Commits before `e788c9b` (2026-10-09) are in the fly-connectome repo, where the engine started; kinestride history begins at `e788c9b`.
