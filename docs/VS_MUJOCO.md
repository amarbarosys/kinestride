# Kinestride vs MuJoCo

Where the engine is ahead of MuJoCo on the CPU, why, and where it is not. Figures come from `docs/ENGINE.md`, `docs/ENGINE_OPT.md`, and `docs/MOTION.md`, with the commit that measured each one.

## Where it is better

**Batched GPU throughput.** The actuated fly, standing on six feet with 44 servos and 6 adhesion pads, runs at 0.64 us per animal-step for 65,536 animals (`dd2f42c`, `docs/ENGINE.md`). That is 22 times MuJoCo on six CPU workers (`docs/ENGINE.md`). The 22x follows from MuJoCo's 70,582 steps/s (`docs/ENGINE_OPT.md`): 65,536 animal-steps take about 928 ms on that CPU, against about 42 ms on the card. Equivalent rate: 1 / 0.64 us, about 1.56M animal-steps/s (derived from the same figure).

**Small batches (open).** `docs/ENGINE.md` says the engine is ahead of MuJoCo "already at 1,024 animals". The optimisation baseline measured 0.45x at 1,024 (32.1 ms per launch, `01c4c57`). Step 0 brought that to 18.6 ms (`32d27c2`). MuJoCo at 70,582 steps/s needs about 14.5 ms for 1,024 animals, so 18.6 ms is still behind. The "ahead at 1,024" claim is not confirmed by a later measurement in the sources.

**Determinism.** Two properties are measured:
- K steps per launch match K single-step launches byte for byte: K=10 is byte-equal to 10 x K=1 (`c5c478c`, `docs/ENGINE.md`).
- The optimisation rule requires two runs to be byte-equal (`docs/ENGINE_OPT.md`, Targets). Those are the rules, not a separate measurement.

**Every rule is ours to change.** MuJoCo's functions are ported, so joints, adhesion and new species are engine code and not a MuJoCo option (`docs/ENGINE.md`, Why a port). A new species is a new MJCF file and an export, not new engine code (`docs/ENGINE.md`, Design).

**Brain and body in one process.** npc-mind brains and the world run in one shared C library (`e682e31`), with crowd scenes in Godot (`c3eef4b`). Crowd frame timings at 1k, 16k and 65k are in `a1f1cc3` (no numbers are copied here).

**Written in Mojo.** The engine and its checks are Mojo, built with Mojo 1.2.0 and MAX 26.7.0 (`a6aff17`). The Python side is the export and the MuJoCo oracle only (`docs/API.md`).

**Motion pipeline.** Retarget runs at 2.8M to 3.65M frames/s, against 23.8k to 25.1k skeletons/s for Godot 4.7.2's RetargetModifier3D on the same machine (`docs/MOTION.md`). Tracking matches MuJoCo's own run of the same loop to 1.2e-7 m (`docs/MOTION.md`).

**A different GPU bet.** MuJoCo Warp on gfx1100 reached correct physics but stayed below six CPU MuJoCo workers at 70,582 steps/s. That track was closed on 2026-10-02 (`docs/ENGINE.md`, citing the Brain note). The port is a different bet: MuJoCo's functions, not MuJoCo Warp.

## Where it is not better, or not yet

- **Hardware.** Only gfx1100 has been measured (`docs/ENGINE.md` measurements and `docs/ENGINE_OPT.md` use the card). Other GPUs are untested.
- **Small batches.** See above: 1,024 animals is not confirmed as faster than MuJoCo on the CPU.
- **Coverage.** Noslip and the contact pairs other than the feet are not ported (`docs/ENGINE.md`). The spider stage is partial (`615d2af`; qfrc_constraint 1.5e-05 to 5.2e-05, cause open).
- **Flight model.** Its automatic body-body contacts need capsule, cylinder and sphere supports, which are missing (`docs/CONVEX_CONTACT.md`).
- **Plane-capsule.** Not ported (`docs/MOTION.md`, Bones are convex meshes).
- **Float32 limits.** Some checks cannot reach 1e-5 in float32. Measured bars are used instead: arena REL 3e-3, convexstep REL 1e-3, terrain_exact REL 4e-4 (`docs/CONVEX_CONTACT.md`). Box-mesh contact counts show 1.5% mismatches against MuJoCo in float32 (`docs/CONVEX_CONTACT.md`).
- **Chaotic states.** Chained runs with contact are chaotic; `convexstep` drops states whose own MuJoCo trajectory is sensitive, at most half (`docs/CONVEX_CONTACT.md`).
- **Heightfield.** The tangent-plane approximation is wrong where terrain bends inside one foot (`docs/HFIELD_CONTACT.md`). The exact path is recorded in `docs/CONVEX_CONTACT.md`.
- **Muscles.** The muscle model is not MuJoCo's gain/bias/dyntype model (`docs/MUSCLE.md`). It cannot be compared to MuJoCo's muscle numbers.
- **Motion planting.** Retarget is 112x to 146x Godot, but planting is 24x against a 100x target (`docs/MOTION.md`).
- **Balance.** The servos alone do not yet balance the BVH clips; the root assist does much of the work (`docs/MOTION.md`, v0.8.0: 0.090 m at ASSIST=0.02).
- **Red stage.** cone_hfield is red (status from the coordinator; no stage row in the sources).
- **Checks need local tools.** `check.sh` calls mojo-cached, mutant-check, mj-oracle and gpu-wait, which are not shipped in this repo (`docs/ENGINE.md`, Stages; `docs/HFIELD_CONTACT.md`, Verification).
- **Test data.** CMU clips are not in git and come from `tools/assets.sh` (`docs/MOTION.md`).

> Commits before `e788c9b` (2026-10-09) are in the fly-connectome repo, where the engine started; kinestride history begins at `e788c9b`.
