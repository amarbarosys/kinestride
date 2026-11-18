---
name: body-engine
description: Port MuJoCo functions into Kinestride, the batched GPU body engine (kinestride/), one function at a time, for any MJCF body (fly, spider, game characters). Use before porting a MuJoCo stage, writing or changing a stage gate, timing the engine, or adding a species.
metadata:
  short-description: "kinestride: port a MuJoCo stage, gate it against MuJoCo, time it"
  owner: amarbaro.org
  license: Apache-2.0
  origin: own
---

# body-engine

Kinestride is an articulated-body engine: MuJoCo 3.9.0's equations ported to Mojo, one GPU thread or block
per body, each stage checked against MuJoCo itself.

## Layout

```
kinestride/
  model.py           export any MJCF -> mi.bin (int32) + mf.bin (float32) + sizes.txt (-D flags); check STAGE
  <stage>.mojo       one MuJoCo function per file, a port of its C source (file and version cited in the docstring)
  check.sh           every stage on the fly and the spider test body, plus a mutant; run one stage at a time
  bench.sh           timing at 1,024 / 16,384 / 65,536 bodies
  models/            test bodies kept in git (spider_test.xml); fly and others via tools/assets.sh into .work/
```

The C source each port reads is the MuJoCo tag `3.9.0`, fetched into `.work/mujoco-src/`. Read it there.

## Porting a stage: five steps, in this order

1. **Read** the C function at the pinned version. List every mjModel field it reads and every mjData field it writes.
2. **Export** the new mjModel fields: append to `INTS` / `FLOATS` in `model.py` and add the matching comptime
   offsets in the stage's `.mojo`. The order in `model.py` is the file format; both sides change in one commit.
3. **Port** the loop with MuJoCo's names, its operand order and its special cases (zero angle, null quaternion,
   world body). Restructure for speed only after the stage passes, as a separate change.
4. **Check**: add the stage to `STAGES` (function, output fields) and run `kinestride/check.sh STAGE`. Gate: every
   field, on the fly and the spider, 64 random states, relative error <= 1e-5 x max(1, |value|).
5. **Mutate**: plant one bug the stage could plausibly have (a sign, a dropped term, a swapped frame). The gate
   must fail. Then commit, with the timing numbers in the body if the change is about speed.

A stage starts only when the stages it reads from pass: kinematics, comPos, crb, factorM, rne / passive, solve,
Euler, then collision, constraint, solver, then actuators.

## Rules that cost us before

- **Random states, never rest.** At rest, velocity terms are zero and orientation is identity; frame and lag bugs
  hide there. The oracle draws free joints at random orientation.
- **Every gate must be able to fail.** A gate checks the property it claims (upright means tilt, walking means
  distance) and must fail on its mutant before its PASS counts.
- **World vs body frame, written down.** MuJoCo quaternions are w, x, y, z. A free joint's qpos is position (3)
  plus quaternion (4); its qvel is linear (world, 3) plus angular (body, 3). `q' = q + dt/2 w q` for a world-frame
  w, `q w` for a body-frame w. Mixing the two spins bodies off.
- **No lagged derivatives inside a force.** A velocity taken from the previous step's position, fed into a damping
  or friction force, pumps energy. Use the state's own velocity.
- **Explicit stiffness limit.** Semi-implicit Euler is unstable at fly scale unless joint damping is implicit and
  the constraint solver runs. Port MuJoCo's implicit joint damping and solver; do not tune around them.
- **float32 vs MuJoCo float64.** Expect about 1e-6 relative error; 1e-5 is the gate. Bit-identity with MuJoCo is not
  a goal. Bit-repeatability run to run is: no atomics in the smooth stages, so two runs must be byte-equal.
- **One binary per body.** Sizes are `-D` constants (NBODY, NJNT, NQ, NGEOM, FLIES), so loop bounds and offsets fold
  at compile time. A new species is a new MJCF plus an export, never new engine code.

## Performance

- **Layout `[row][animal]`, animals fastest**: a wave reads 32 neighbouring animals in one transaction. Model arrays
  are shared by every animal and stay in cache.
- **Bytes are the budget.** Quaternions, not 3x3 matrices (4 floats vs 9). Write to global memory only what a later
  launch reads; inside a fused step, keep intermediates in registers or LDS. Measure bytes per launch (rocprofv3).
- **Read-after-write chains are latency.** A body that reads its parent's pose right after writing it serialises
  the launch. Keep such chains in registers or LDS when fusing.
- **Registers, not stack arrays.** Per-thread arrays that are not comptime-indexed become scratch memory and run
  several times slower. Check spill bytes before calling a kernel fast.
- **The mass matrix (74 x 74 per fly) does not fit one thread.** One block per body; LDS for M and its factor.
  Check VGPRs and occupancy from the trace.
- **Timing.** Warm the card for at least 1 s before each variant, time at least 100 launches, and report per-launch
  time and body-steps per second with the body count. Time batched runs batched (1k / 16k / 65k), never one body
  from a Python loop. Compare to MuJoCo CPU as steps per second to steps per second at the same dt.
- **Correctness before speed.** A speed number from a build that has not passed `check.sh` is not reported.

## GPU and gates

- Run GPU stages one at a time. `check.sh` takes `.work/check.lock` per worktree and fails a second stage in the same
  tree, because the mutant edits sources in place. A killed run can leave a `.mutant-orig.*` backup; check `git diff`.
- CPU first: `mojo build` and the export run on CPU. The GPU is needed only for the run.
- Gate scripts use `set -euo pipefail`, print `FAIL <step>` on each failure, and never swallow a failure with
  `|| true` when a later step reads its output.

## Float32 Newton

- MuJoCo's line-search tolerances sit below float32 noise, so every line search can run to its evaluation cap.
  Stop on a slope floor relative to the start, and stop when a step moves qacc by less than a small fraction of its
  size. Do not stop on "a full step kept the active set": with stiff contacts one float32 step is not exact.
- `phases.sh` (`-DSTOP=k`) gives time per phase. Read the Newton iteration and line-search counts before any solver change.

## Contacts

- **Check states come from actuated rollouts** (`model.standing`), not random poses: random poses put the body through
  the floor or the feet in the air. MuJoCo's `ncon` also lists excluded contacts; count only those with constraint rows.
- **float32 floor.** Stiff contacts amplify rounding of the input state. Measure each contact field's floor (MuJoCo on
  the rounded state) and bar it at max(1e-5, 3 x floor). Before blaming the solver, rerun with tolerance 0: if the
  error does not move, the solver is converged.
- **Active-set flips.** A constraint row sitting within the tolerance of its switch can land on either side in float32.
  Diagnose one bad state with per-state errors and the nearest-switch row before touching the solver.
- **Merged bodies** (`model.combine`): one floating origin per model. A second body far from the first raises the
  float32 input floor; per-root origins fix it.

## Mojo traps met here

| error | fix |
|---|---|
| `expected argument name` at a parameter called `out` | `out` is a keyword: rename it |
| `Could not infer capture convention of the captured value` | no nested closures over tensors in a kernel: top-level `@always_inline` helpers that take the tensor |
| `ambiguous merge: FloatLiteral and IntLiteral` in `a if c else 0` | write `0.0` |
| `lld: error: undefined symbol: atan2f` at run time | amdgpu has no `atan2f`: use the engine's own `atan2p`. GPU-only link errors appear at launch-time compile, not in the CPU build; a 64-body GPU run is the real build check |

## Adding a species

Export the MJCF with `model.py`, run `check.sh step` on the new body, and read the refusal list the export prints.
Do not add engine code for a new body shape; if the export refuses it, that is the port queue.
