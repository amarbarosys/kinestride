# Kinestride history

## Origin

Kinestride came out of fly-connectome, a fly body simulation. The body engine was a direct port of MuJoCo's functions into Mojo for a fixed body, one thread or wave per animal (`docs/ENGINE.md`).

Before this repo existed, the GPU track was checked on MuJoCo Warp. On 2026-10-02 that track was closed: MuJoCo Warp on gfx1100 reached correct physics but stayed below six CPU MuJoCo workers at 70,582 steps/s (`docs/ENGINE.md`, citing the Brain note `2026-10-03-gpu-body-port.md`). The engine port started from there.

The baseline for the optimisation work was measured on 2026-10-03 at `01c4c57`, the S2 probe (`docs/ENGINE_OPT.md`). That commit is cited in the docs but is not among the first commits of this repo. The repo log begins at the root commit `7a119b3`, on 2026-10-09.

## Repo start (2026-10-09)

- `7a119b3` feat: Kinestride, batched GPU physics for articulated bodies in Mojo. Root commit.
- `4b4367f` docs: body-engine skill moves here from fly-connectome.
- `72d9431` fix: meshes(world=...) imported kinestride.assets, which stayed in fly-connectome.
- `51b4d03` fix(tools): assets.sh stops copying MuJoCo sources from fly-connectome.
- `adeb34f` fix(tools): flight_export takes the fly-connectome checkout as an argument.
- `4441d74` docs(release): v0.1.0, the first release as its own repo.
- `6591ede` docs(release): v0.2.0, the look round.
- `370b0d2` docs(release): v0.3.0, the fly looks like MuJoCo's.

## Releases

Tags from `git tag` (all tags use the form `vX.Y.Z`; each was tagged with the message "kinestride vX.Y.Z"):

| Tag | Date | Release commit | Note |
|---|---|---|---|
| v0.1.0 | 2026-10-09 | `4441d74` | first release as its own repo |
| v0.2.0 | 2026-10-09 | `6591ede` | the look round |
| v0.3.0 | 2026-10-09 | `370b0d2` | the fly looks like MuJoCo's |
| v0.4.0 to v0.11.0 | not in the sources | | tags exist; release commits not cited here |
| v0.8.0 | not in the sources | | motion check cited in `docs/MOTION.md` (v0.8.0: 0.090 m at ASSIST=0.02; 31-body character 61 ms) |
| v0.12.0 | 2026-10-23 | `f80b0d6` | |
| v0.12.1 | 2026-10-26 | `fa7b5fb` | |
| v0.13.0 | 2026-10-26 | `1805c08` | |

Dates and hashes for v0.12.0 to v0.13.0 are from the release commits in `git log`.

## Milestones

- **2026-10-09**: repo start, first release (v0.1.0), look round (v0.2.0), render and GUI (v0.3.0). Tools added: `doclinks.sh` (`f09e6b0`), `smoke.sh` (`bf489e6`), `blender_check.sh` (`3248a3f`).
- **2026-10-12**: motion and tracking land. BVH to MJCF character (`9981459`), tracking loop (`01fb255`), dataset driver (`382f1b0`). Engine contacts across two trees (`9016296`), mesh-mesh pairs (`05d8954`), push input (`5ed0206`), per-animal equality switch (`9d593b6`). City renderer (`2c3a1af`).
- **2026-10-16 to 10-17**: Living World (`b94e901`, `ba22b87`, six species in `9e1edb9`). Stateful tendon muscle engine (`a2b9851`) and muscle gating (`98d4f63`), game stamina from muscle work (`77cc7f2`). Tone mapping and sky scattering (`05d77fe`, `5b7a2fd`).
- **2026-10-22**: smooth macro surface (`8558fe8`), trails, gulls, aurora. Table-backed heightfield HTAB (`0ee9ab5`).
- **2026-10-23**: contacts and solvers. Equalities, friction-loss rows, PGS over all row kinds (`2e87a24`), CG (`eafad87`), elliptic cone under RK4 (`3a897e3`), margin-0 box-mesh and mesh-mesh by polygon clipping (`2f552a2`). `bisect_stage.sh` (`3b8d5f5`).
- **2026-10-26**: union walk tried, measured 16 to 27% slower, and reverted (reverts `203d421`, `604ad48`; `a6be7f7` records the kernel at 128 VGPRs). npc-mind NPCs walk as spider bodies (`e465f99`). npc_live peak RSS down from 1,580 MB to 184 MB (`7209cf9`).
- **2026-10-27**: snapshot and restore of qpos, qvel, act, warm start (`8052f39`). Contact list out and contact events ring (`b26adc4`).
- **2026-10-29**: build moved to Mojo 1.2.0 and MAX 26.7.0 (`6443f95`). Every check.sh stage in its own tree copy, GPU tests in parallel (`6a28e43`).
- **2026-10-30**: Godot integration. Kinestride GPU launch from inside Godot (`fb579b8`), C ABI (`c7af891`), KinestrideWorld and NpcMind classes (`d31eddd`), MultiMesh transforms (`9229756`), npc-mind brains as a C shared library (`02767ec`), crowd and NPC scenes (`5eeaef1`), crowd frame timings at 1k / 16k / 65k (`2884ab4`). Blender rig as a 14-body humanoid saved as `.blend` (`6050930`).

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

> Commits before `7a119b3` (2026-10-09) are in the fly-connectome repo, where the engine started; kinestride history begins at `7a119b3`.
