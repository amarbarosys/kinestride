# Whole-body GPU migration

The complete model is exportable. A faithful AMD GPU runtime replacement is **UNVERIFIED**.

## Verified portable model

`agent/export_body.py` exports the complete embodied world using the installed FlyGym exporter, retains the 0.0001-second physics timestep, and restores full-precision compiled body frames and inertials. The initial export failed its mechanical check because serialized inferred inertias changed. The corrected export passes all checked arrays and names at relative tolerance 1e-10 and absolute tolerance 1e-12.

Receipt: `.work/agent/gpu-body-port/body-v2/manifest.json`. Model: adjacent `body.xml`, mesh assets and `initial-state.npz`.

Counts: 71 bodies, 69 joints, 75 position coordinates, 74 velocity coordinates, 50 actuators, 113 geoms, 69 meshes and 502 explicit contact pairs. Checks cover body inertias and frames, joint limits and damping, geometry and mesh arrays, contact-pair friction/solver settings, actuator transmissions/gains and simulation options. Units remain the existing FlyGym mm-based convention. This is a MuJoCo export/reload check, not cross-engine dynamics equivalence. CPG/reflex controller code and runtime snapshots need a separate adapter.

## Backend compatibility evidence

[Genesis installation documentation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html) describes an AMD ROCm backend. Downloaded PyPI candidate version 1.4.3 was inspected without installing it or launching a GPU workload. Package and importer hashes are recorded in `.work/agent/gpu-body-port/backend-audit.json`.

Its `genesis/utils/mjcf.py` selects joint transmissions at line 420 and tendon transmissions at lines 429–432. It does not parse BODY adhesion transmissions. Its geometry parser processes collision masks and explicit exclusions at lines 765–802 but does not import positive `mj.pair_*` contact pairs. This model uses BODY adhesion and explicit contact pairs, so direct MJCF import cannot establish faithful behavior. These findings concern the importer, not a claim that custom Genesis extensions are impossible.

## Remaining implementation and acceptance

1. Implement preserved explicit contacts and adhesion in the selected AMD GPU solver; verify every required feature before enabling a viewer.
2. Map joint names, position actuators, contact observations, controller state, reset and cameras without reordering or dropping model features.
3. CPU preflight the adapter, then queue a short explicit AMD backend run with device evidence and no silent physics fallback.
4. Compare contact forces, inertia, pose and gait against the retained baseline; rerun the frozen 20-seed floor/wall suite and actual GPU UI reset.
5. Measure single-fly full-loop latency and rendering before switching the normal launcher or retiring its current physics backend.

## Completed baseline

Independent engineering verification: `.work/agent/p1-body-gate/full-verification-cpu-ui.json`, PASS. All 20 60-second floor trials had upright fraction 1.0; 19/20 five-second wall trials escaped, meeting the frozen 90% threshold. Seed 12 timed out. 130,000 exchanges were finite. Actual CPU body/brain reset and embodied learning replay passed. GPU UI verification remains UNVERIFIED after three input-automation failures, not a demonstrated reset failure. These results do not establish biological validity or complete neural-loop acceptance.

## 2026-10-02 gfx1100 probe update

The ROCm Warp fork at commit `9d0b6cf` built for `gfx1100` with
`--hip-arch=gfx1100` (Warp 1.13.0+rocm.0), and its smoke kernel passed through `gpu-wait`.
The ROCm mujoco_warp fork at `9229bb9` includes BODY transmission support. The supplied body
still fails conversion because `noslip_iterations=5` is unsupported; after setting it to zero,
conversion also rejects 275 explicit box-mesh pairs with margin 0.001 under MULTICCD. A diagnostic
copy with those margins zeroed converts and steps once with finite qpos, maximum qpos difference
0.0003205 from CPU, and six finite 40 N adhesion forces. At 1,000 steps its qpos contains NaN,
while CPU qpos remains finite. Stable GPU stepping is therefore UNVERIFIED. These diagnostic model
changes are not the supplied gate body and do not establish equivalent physics.

The upstream Warp PyPI project is CUDA-only; the ROCm fork's supported GPU list names gfx942, not
gfx1100. The inspected MJX-JAX transmission enum does not include BODY. A survey of 25 fly
simulation repositories found no GPU fly-physics run. Its notes report alextitonis' shuffled model
at 0.87 bits versus real at 0.83, a slightly better matched control in the FLM study, and a
FlySweeper saved run of 60/100 (a separate earlier run reported 76/100, under a distinct run).
These are contextual reports, not directly comparable outcomes.

Receipts: `.work/agent/gfx1100-probe/{REPORT.md,result.json,probe.log,INCOMPAT.md}` and
`.work/briefs/gpu-port/PLAN.md`; upstream snapshots under `~/Projects/imports/GpuPhysics/`;
field survey `~/Projects/imports/FlyAssets/SURVEY.md`. The in-progress probe directory was not edited.

## Upstream FlyGym GPU benchmark on gfx1100: stop, 2026-10-02

FlyGym ships its own MuJoCo Warp path (`flygym/src/flygym/warp/simulation.py`, `GPUSimulation`) and
benchmark (`flygym_demo/benchmark/time_gpu_simulation.py`). Run as shipped on the ROCm Warp fork
(warp 1.13.0+rocm.0, mujoco_warp 3.8.1 ROCm), recorded joint-angle replay, per-leg adhesion from the
CPG phase schedule, 200 steps, 0.1 ms:

| Worlds | Steps/s | Finite worlds |
| ---: | ---: | ---: |
| 1 | 206 | 1/1 |
| 4 | 162 | 0/4 |
| 64 | 2,267 | 0/64 |
| 1,024 | 26,825 | 0/1,024 |
| 4,096 | 43,433 | 0/4,096 |

CPU reference: 6 workers on the gate body, 70,582 steps/s. The upstream model, not our export, goes
nonfinite as soon as more than one world runs, so the failure is in this ROCm stack (a single world
stays finite), and even the invalid throughput at 4,096 worlds is below the CPU. The humanoid
benchmark on the same stack is finite and fast (2.19M steps/s at 8,192 worlds), so the card works;
the fly workload does not. Port stopped; CPU MuJoCo remains the physics backend.
Receipts: `.work/agent/gfx1100-probe/{result.json,REPORT.md,step7_flygym_results.json}`.

## Root cause of the gfx1100 NaNs, 2026-10-02 (driver investigation)

Three defects in the ROCm forks, all from tuning for gfx942 (CDNA, 64-lane wavefronts), found by
stepping CPU-Warp and GPU-Warp side by side on identical state:

| # | Defect | Evidence | Probe fix |
|---|---|---|---|
| 1 | `warp/native/tile_reduce.h:14` hardcodes `WP_TILE_WARP_SIZE 64` for every HIP target; gfx1100 (RDNA3) compiles wave32 | `wp.tile_sum` of 256 ones returns **128** at block_dim 64/128/256 on gfx1100, 256 after the fix (`tile_sum_test.py`) | define 32 (`tile_reduce-wave32.patch`) |
| 2 | mujoco_warp `io.py:321` forces Cholesky block dims to 64 on HIP; its own comment says values above one wavefront "produce NaNs", and on wave32 64 is two wavefronts; the batched Cholesky assumes "one 64-lane wavefront per world" | with 1 and 2 fixed, 1/2/4 identical worlds stay finite 400 steps and agree within 7e-8 | `MJW_BLOCKDIM_*=32`, `MJW_BATCHED_CHOLESKY=0` (`wave32-all.env`) |
| 3 | `collision_driver.py:797` "AMD Opt 5" caches the first step's broadphase pairs forever on any HIP device; a fly spawned in the air caches an empty list and never contacts the floor | GPU nacon 0 while CPU-Warp lands with 8-10 contacts; with the cache cleared per step GPU z, contacts and constraints match CPU-Warp at every sampled step to 200 | clear `d._bvh_cached_ctx` before each collision |

Defect 3 affects every AMD GPU, not just RDNA. Defects 1 and 2 are RDNA (wave32) specific.

FlyGym recorded-motion benchmark, all three fixes, 1,000 steps, correct contacts:

| Worlds | Steps/s | Finite |
| ---: | ---: | ---: |
| 1 | 110 | 1/1 |
| 64 | 5,326 | 64/64 |
| 256 | 19,060 | 256/256 |
| 1,024 | 45,760 | 1,024/1,024 |
| 4,096 | crash: GPU memory access fault | |

CPU reference 70,582 steps/s. Correct GPU physics now runs, but stays below six CPU workers up to
1,024 worlds and faults at 4,096 (likely contact/constraint buffer sizing, not investigated).
Decision unchanged: CPU MuJoCo remains the backend. The three defects are reportable upstream to
ROCm/warp and ROCm/mujoco_warp with the receipts in `.work/agent/gfx1100-probe/`
(`tile_sum_test.py`, `diag_pair.py`, `diag_contact.py`, `diag_worlds.py`, `step8.log`).

## GPU speed items 1-4 on gfx1100, 2026-10-02

Direct timing loop (`.work/agent/gfx1100-probe/bench_gpu.py`, FlyGym benchmark model exported to
`flygym_bench.mjb`, keyframe 0, held controls), all three earlier fixes applied. `mjwarp-testspeed`
cannot run on HIP as shipped: its outer `ScopedCapture` collides with the fork's own HIP graph and
multi-stream syncs ("Cannot synchronize stream while graph capture is active").

**1. Profile (1,024 worlds, EventTracer):** the constraint solver is 87% of step time (6.73 of 7.72 s;
line search 2.25 s). Solver iterations averaged 58 and often hit the 100 cap. CPU MuJoCo (float64)
converges in 2.3 iterations on the same start. Cause: the model tolerance is `1e-8`, below float32
resolution (eps 1.2e-7), so the float32 GPU solver can never declare convergence.

**2. Knobs:** tolerance `1e-6` brings GPU iterations to 4.0-4.5, matching CPU behaviour; `MJW_SOLVER_NCHECK=10`
gave no gain. Contacts 8-12 per world, height 1.082-1.085, consistent with CPU MuJoCo 1.0843.

| Worlds | Steps | Steps/s | Finite | Repeat |
| ---: | ---: | ---: | ---: | --- |
| 256 | 400 | 35,500 | 256/256 | 2 runs identical |
| 512 | 400 | 62,453 | 512/512 | |
| **768** | **1,000** | **74,625 / 74,460** | **768/768** | **2 runs, identical z and iterations** |
| 1,024 | 400 | 31,700-46,200 | 66-1,000 of 1,024, varies by run | nondeterministic |
| 2,048 / 4,096 | 400 | n/a | 0 | |

CPU reference: 70,582 steps/s (six workers). **768 worlds is the first correct GPU configuration that
beats the CPU, by about 6%.** Above roughly 1,000 worlds identical worlds diverge into NaN at random;
larger contact buffers make it worse (1,024 worlds at nconmax 128 / njmax 512: 0 finite) and restoring
the sort kernels' default block sizes does not fix it. A fourth defect scales with total buffer size;
not found yet.

**3. 4,096 crash:** with explicit nconmax 32 / njmax 256 (FlyGym defaults 500/500) it no longer faults,
but runs nonfinite, the same >1,000-world defect as above.

**4. hipGraph (`WP_HIP_GRAPH_ENABLE=1`):** runs but breaks physics: 27/512 finite, mean height 3.16
(flies leave the floor). Not usable.

Decision unchanged for production: CPU MuJoCo stays the backend. A 6% win at a fixed 768 worlds does not
justify a forked, patched stack; the >1,000-world defect is where any real speedup lives.
GPU time for this section: about 14 GPU-min (`gpu-wait stats --days 1`: 45+ jobs, 22+ GPU-min total
across the probe).

## Above-768-world NaNs: partial root cause, track closed, 2026-10-02

Investigated by the opencode lane (`.work/briefs/gpu-bug4/REPORT.md`, scripts and receipts under
`.work/agent/gfx1100-probe/bug4/`). Corrections to the sections above: the failure is **not** a 1,024
boundary. Worlds also diverged at 768 (one run, step 232), 1,000 and 1,023; more worlds only raise the
odds. The two 768-world clean runs above were not a guarantee.

Mechanism with evidence: the fork caches one `SolverContext` per Data (`solver.py:156-188`) and on reuse
re-zeros 4 of about 24 `wp.empty` scratch fields, relying on "overwritten before use". Re-zeroing all of
them restored 1,024/1,024 finite worlds at about 5 solver iterations in 1 of 3 runs; synchronising the
device every solver iteration did the same in 1 of 2 runs; the unmodified fork failed 3 of 3. So stale
scratch plus a synchronisation fault is the dominant mechanism, the specific field is not isolated, and a
second defect remains (one run blew up in integration with a healthy solver). The report's cross-stream
explanation (`solver.py:3333`) only runs with the CG solver; this model uses Newton, so that line is not
the sync fault here.

**Track closed.** The remaining work is repairing the ROCm fork, and even repaired the measured gain over
six CPU workers is single-digit percent. CPU MuJoCo is the physics backend.

## Buffer bytes per world, 2026-10-03

`.work/agent/gfx1100-probe/mem_audit.py` (CPU Warp device, 1,024 worlds; CPU MuJoCo over 2,000 steps
peaks at 19 contacts and 76 constraint rows):

| nconmax / njmax | Data MB | KB per world |
| --- | ---: | ---: |
| 500 / 500 (FlyGym default) | 230.4 | 230 |
| 32 / 128 (bench default) | 83.8 | 84 |
| 16 / 64 | 69.1 | 69, below the 19 / 76 peak: overflows |

32 / 128 is the floor with headroom. At 84 KB per world, 4,096 worlds need about 340 MB of a 24 GB card, so
memory is not what limits the world count; the >768-world NaNs are the stale-scratch defect above.
`efc.J_colind` is allocated at the dense `efc.J` size (8.5 MB each), the one remaining waste.

## Own engine, 2026-10-03

A separate track ports MuJoCo's functions into our own Mojo engine instead of running MuJoCo Warp: `BODY_ENGINE.md`. CPU MuJoCo stays the backend until that engine passes its S1 gate.
