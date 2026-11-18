# Kinestride vs MuJoCo

Where the engine is ahead of MuJoCo on the CPU, why, and where it is not. Figures come from `docs/ENGINE.md`, `docs/ENGINE_OPT.md`, and `docs/MOTION.md`, with the commit that measured each one.

## Where it is better

**Batched GPU throughput.** The actuated fly, standing on six feet with 44 servos and 6 adhesion pads, runs at 0.64 us per animal-step for 65,536 animals (`dd2f42c`, `docs/ENGINE.md`). That is 22 times MuJoCo on six CPU workers (`docs/ENGINE.md`). The 22x follows from MuJoCo's 70,582 steps/s (`docs/ENGINE_OPT.md`): 65,536 animal-steps take about 928 ms on that CPU, against about 42 ms on the card. Equivalent rate: 1 / 0.64 us, about 1.56M animal-steps/s (derived from the same figure).

**Measured 2026-11-18** (`kinestride/bench.sh run body.xml 7`, actuated fly, RX 7900 XTX, K=1; MuJoCo figure 70,582 steps/s on six CPU workers from 2026-10-02, not re-run):

| animals | per launch | animal-steps/s | vs MuJoCo CPU |
|---:|---:|---:|---:|
| 1,024 | 22.3 ms | 45,972 | 0.65x |
| 16,384 | 38.6 ms | 424,841 | 6.0x |
| 65,536 | 84.7 ms | 773,606 | 11.0x |

The 0.64 us / 22x figure above is from `dd2f42c` (feet-only contacts); stage 7 now measures 1.29 us at 65,536, the same at kinestride's first commit `7a119b3` (85.2 ms vs 84.7 ms per launch), so the current number is 11x. Likely cause, not bisected: all 55 plane-mesh floor pairs (`00a58e6`) replaced the 6 foot pairs. The engine is behind MuJoCo at 1,024 animals.

**Determinism.** Two properties are measured:
- K steps per launch match K single-step launches byte for byte: K=10 is byte-equal to 10 x K=1 (`c5c478c`, `docs/ENGINE.md`).
- The optimisation rule requires two runs to be byte-equal (`docs/ENGINE_OPT.md`, Targets). Those are the rules, not a separate measurement.

**Every rule is ours to change.** MuJoCo's functions are ported, so joints, adhesion and new species are engine code and not a MuJoCo option (`docs/ENGINE.md`, Why a port). A new species is a new MJCF file and an export, not new engine code (`docs/ENGINE.md`, Design).

**Brain and body in one process.** npc-mind brains and the world run in one shared C library (`02767ec`), with crowd scenes in Godot (`5eeaef1`). Crowd frame timings at 1k, 16k and 65k are in `2884ab4` (no numbers are copied here).

**Written in Mojo.** The engine and its checks are Mojo, built with Mojo 1.2.0 and MAX 26.7.0 (`6443f95`). The Python side is the export and the MuJoCo oracle only (`docs/API.md`).

**Motion pipeline.** Retarget runs at 2.8M to 3.65M frames/s, against 23.8k to 25.1k skeletons/s for Godot 4.7.2's RetargetModifier3D on the same machine (`docs/MOTION.md`). Tracking matches MuJoCo's own run of the same loop to 1.2e-7 m (`docs/MOTION.md`).

**A different GPU bet.** MuJoCo Warp on gfx1100 reached correct physics but stayed below six CPU MuJoCo workers at 70,582 steps/s. That track was closed on 2026-10-02 (`docs/ENGINE.md`, citing the Brain note). The port is a different bet: MuJoCo's functions, not MuJoCo Warp.

## Where it is not better, or not yet

- **Hardware.** Only gfx1100 has been measured (`docs/ENGINE.md` measurements and `docs/ENGINE_OPT.md` use the card). Other GPUs are untested.
- **Small batches.** See above: at 1,024 animals the engine runs 0.65x MuJoCo on the CPU; the crossover lies between 1,024 and 16,384 (not measured).
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

> Commits before `7a119b3` (2026-10-09) are in the fly-connectome repo, where the engine started; kinestride history begins at `7a119b3`.
