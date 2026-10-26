# Body engine: maximum optimisation before S2 continues

The engine is correct through the S2 probe and runs 2.03 us per animal-step at 65,536 animals, but it
is a straight port: one thread per animal, every intermediate in global memory, and contacts plus
Newton take 89% of the step (118 of 133 ms). The floor is two to three orders lower. A fused step
that keeps each animal in LDS moves about 600 B of state per step: 39 MB per launch at 65,536, about
40 us at 960 GB/s. Its arithmetic, a few factorisations of a 74-dof sparse matrix, is under 1 ms at
61 TFLOPS. Small batches are the worst case today: 1,024 animals take 32 ms, slower than MuJoCo on the
CPU, because 1,024 threads leave most of the 96 compute units idle.

## Baseline (2026-10-03, Ultimate-AI, `01c4c57`)

| batch | per launch | per animal-step | steps/s | vs MuJoCo CPU (70,582) |
|---|---:|---:|---:|---:|
| 1,024 | 32.1 ms | 31.4 us | 31,880 | 0.45x |
| 16,384 | 83.4 ms | 5.09 us | 196,412 | 2.8x |
| 65,536 | 133 ms | 2.03 us | 492,210 | 7.0x |

rocprofv3: one kernel, 96 VGPRs, 0 scratch, 0 LDS, 64-thread blocks. The contact-free step (S1) is
15.0 ms of the 133.

## Targets

- 65,536 animals: at most 13 ms per launch (10x), stretch 1.3 ms (100x).
- 1,024 animals: faster than MuJoCo on the CPU, so at most 14.5 ms per launch; stretch 1.5 ms.
- Every field still passes `check.sh` for every stage (kinematics through contact), the mutants still
  fail, and two runs are byte-equal.

## Rules for every step

1. Measure before and after: `bench.sh run` at 1k / 16k / 65k on stages 5 and 6, plus rocprof-kernels
   (VGPRs, scratch, LDS). Put the numbers in the commit body.
2. `check.sh` for kinematics, crb, factorM, rne, step and contact passes before a number counts.
3. One change per commit. A step that does not move the time is reverted and recorded here.
4. Same LACT profile for every comparison, named next to the number.

## Steps

```
0 split ──► 1 cheap work cuts ──► 2 wave per animal ──► 3 K steps per launch ──► 4 tune ──► 5 report
   │            (warm start,          (LDS state,           (state stays          (block size,
   │             hull climb,           lanes over dofs,      resident)             occupancy,
   │             skip dead rows)       rows, vertices)                              layouts)
   └─ which phase costs what
```

**0. Phase split (S, measurement only).** Time kinematics, comPos/crb, factorM, rne/passive, collision,
constraint build, Newton, Euler separately: comptime switches that stop after each phase, the same
bench. Add rocprofv3 counters for bytes fetched and written per launch. Output: a table of ms and
bytes per phase at 65,536 and 1,024. Every later step is ranked by this table.

**1. Cheap work cuts, layout unchanged (S each).**
- 1a. Warm start: the bench keeps the last step's qacc per animal, and Newton starts from the cheaper
  of it and qacc_smooth (MuJoCo's warmstart rule). Fewer Newton iterations and line-search evaluations.
- 1b. Hull support by hill climbing over the mesh graph from the last support vertex, as MuJoCo does,
  instead of scanning all 1,244 hull vertices.
- 1c. Skip dead work: no Jacobian, impedance or Hessian rows for empty contact slots; Hessian rebuilt
  only for the rows whose state changed (MuJoCo's incremental update).
- Check: phase times drop; check.sh unchanged.

**2. One wave per animal, state in LDS (L, the main lever).** One 32-lane wave owns one animal. M,
the Hessian, J, the efc rows and the solver vectors live in LDS (about 18 KB per animal). Loops over
dofs, constraint rows and hull vertices split across lanes, and reductions use wave shuffles. Global
memory sees only model data (cached) and the state in and out.
- 2a. Newton plus constraint build (89% of the step) first, behind the same check.
- 2b. Factorisation and solve by tree level: the six legs are independent subtrees, so their rows
  eliminate in parallel; the thorax chain stays serial.
- 2c. Kinematics through rne in the same wave (bodies by tree level), so the whole step is one fused
  pass.
- This step also fixes small batches: 1,024 animals become 1,024 waves instead of 16 blocks.
- Check: LDS per wave and occupancy from the trace, plus the full check.sh.

**3. K steps per launch (M).** The state stays in LDS or registers across K steps (as in S0), and only
qpos and qvel go out. This needs the step to feed itself, so the bench gains a K-step mode, and the
check compares K = 1 against MuJoCo and K = 10 against 10 MuJoCo steps on the same states (each field
within its float32 floor).

**4. Tune (S).** Waves per block, LDS layout (bank conflicts), model arrays as float4 or in constant
memory, VGPR budget against occupancy, hull vertices packed. Each knob is one bench run.

**5. Report (XS).** The final 1k / 16k / 65k table against the baseline above and MuJoCo CPU, in
`ENGINE.md` and the board. S2 resumes (noslip, the other 496 contact pairs, S3 actuators) only
after this.

## Effort

| step | size | risk |
|---|---|---|
| 0 phase split | S | none |
| 1 work cuts | 3 x S | low |
| 2 wave per animal | L | the main risk: LDS budget against occupancy, and a cooperative sparse factor |
| 3 K steps per launch | M | the check must chain steps |
| 4 tune | S per knob | low |

## Results

**Step 0 (`32d27c2`).** Phase split at 65,536 animals, ms per launch: kinematics 1.4, comPos + crb 4.0,
factorM 2.9, rne/passive 2.1, collision + constraints 0.8, Newton 117.4, Euler 5.0. Newton took
MuJoCo's iteration count (5.4) but MuJoCo's tolerances sit below float32 noise: every line search ran
to its 50-evaluation cap (178 per animal), and a few animals ran all 100 Newton iterations, holding
their wave. Two float32 stopping rules (a slope floor of 1e-6 of the start; stop when a full step keeps
the active set) cut that to 16 evaluations and at most 9 iterations, with check.sh unchanged:

| batch | before | after |
|---|---:|---:|
| 1,024 | 32.1 ms | 18.6 ms |
| 16,384 | 83.4 ms | 31.4 ms |
| 65,536 | 133 ms | 61.9 ms (0.94 us per animal-step, 15x MuJoCo CPU) |

After it, Newton is still 46 of the 62 ms; the rest of the step is 15 ms.

**Step 1.** 1a (warm start) is in: K > 0 builds start Newton from the last step's qacc (`OWARM`). 1b (hull climbing) is dropped:
collision costs 0.8 ms. 1c (dead rows, Hessian only on a state change) was already in place.

**Step 2 probe: wave per animal loses at large batches.** `probes/proto_factor.mojo` runs
mj_factorI on the fly's 74-dof M (64 MuJoCo samples, every variant within 2.7e-06 of MuJoCo's qLD):

| us per factorisation | 1,024 animals | 65,536 animals |
|---|---:|---:|
| thread per animal, global memory (the engine) | 790 | 2,120 |
| wave per animal, L in LDS | 79 | 4,130 |
| 8 / 4 / 2 lanes per animal, L in LDS | 181 / 315 / 601 | 8,519 / 14,419 / 27,815 |
| wave per animal, pattern also in LDS | 110 | 5,033 |

A wave per animal is 10 times faster at 1,024 animals, where one thread per animal leaves the card idle,
and 2 times slower at 65,536: each row waits on two barriers and keeps few lanes busy, and packing
several animals into a wave multiplies the LDS per wave until occupancy collapses. The whole step needs
about 34 KB of LDS per animal, ten times this probe, so a fully fused LDS step would do worse still at
large batches. Step 2 as written is dropped for large batches; the two batch regimes need different
kernels.

**Revision with S3 (`dd2f42c`).** The second stopping rule, stop when a full step keeps the active set,
was wrong: with stiff contacts the Newton matrix is ill-conditioned (condition ~4e4), so one float32
step is not exact, and the actuated check caught one state at 5.4e-05. Newton now stops when a step
moves qacc by under 1e-6 of its size. Contact bench unchanged (65,536: 62 ms, at most 9 iterations).
With servos and adhesion the standing fly needs 2.4 iterations: 65,536 animals 42.2 ms, 1,024 13.7 ms.

**Next here (option 1, chosen):** big-batch traffic cuts in the one-thread-per-animal kernel:
incremental Hessian updates, warm start from the previous step (K-step loop), Jacobian recomputed
instead of stored, K steps per launch.

**Newton breakdown (2026-10-05, stage 6, fly, WAVE=0).** Counters (`8f47bc5`): per animal-step 4.45 Newton
iterations, 15.2 line-search evaluations, 4.33 Hessian rebuilds, 6.3 of 172 contact slots active (65,536 animals).
Timing probes (patch `.work/newton-hprobe.patch`, not kept): rebuilding the same H twice adds 27.7 ms for 3.33 extra
rebuilds, so one rebuild = 8.3 ms and rebuilds = ~36 ms of the 88.8 ms step (40 %); assembling H twice, factoring
once adds 15.4 ms, so per rebuild assembly (zero, copy M, J'DJ) = 3.55 ms and `factorI` = 4.75 ms. At 1,024: one
rebuild ~2.4 ms. Looping only the active slots in update/search/hessian/start (candidate A, `.work/newton-candA.patch`)
measured 89.74 vs 89.11 ms: lost, reverted. Next: cut factorI (20.6 ms/step) and assembly (15.4 ms/step).

**Hessian cuts tried (2026-10-05, 65,536 animals, stage 6, baseline 88.9 ms).**
- Incremental assembly (keep assembled H in the free noslip scratch, add/subtract D J'J only for rows whose state
  flipped, copy + factor; `.work/newton-incH.patch`): 86.25 ms, -2.9 % (1,024: -3.2 %). All check.sh stages and stand.sh
  PASS except terrain spider qfrc_constraint 1.24e-05 (recorded gap 1.21e-05). Under the 3 % bar: reverted. The J'DJ
  sums were never the cost; moving H's 905 entries per animal through global memory is, and the copy into the factor
  buffer still does it.
- Stale H (refactor only every Nth active-set change): never 389.6 ms (75.6 iterations, max 100), every 2nd 303.4 ms
  (17.8 iterations). Stiff contacts make an old H a bad Newton direction. Dead.
Rank-1 update of the factor in place (`8a097d6`): 88.9 -> 70.6 ms at 65,536 (-20.6 %), rebuilds 4.33 -> 1.0. Smaller H pattern not worth it: all 77 fill entries come from the 7 grooming pairs (feet only: H = M pattern, 828), < 1 % after the rank-1 change.

**Byte budget (2026-10-09, fly body-v2, stage 7, standing actuated states, WAVE=0).** Footprint (`-DLAYOUT=1`): 35,309
float32 rows = 141 KB per animal, 9.3 GB at 65,536: noslip scratch 12,490 rows (35 %, OWC alone 9,356 = a second
J-shaped array), contact Jacobian 9,356 (26 %), stage outputs 7,576 (21 %), contact rows 3,612, Hessian 979, solver
vectors 444, body scratch 852. Both big regions are sized for all 172 slots; 6.1 are active.
DRAM traffic per launch (`traffic.sh`, L2->memory request counters, iGPU unbound + profile_standard):

| animals | read | written (L2 evictions) | per animal read | ms/launch (auto) |
|---|---|---|---|---|
| 1,024 | 136 MB | 64 MB | 139 KB | 22.0 |
| 65,536 | 38.2 GB | 14.4 GB | 611 KB | 66.7 |

At 65,536 that is 52.6 GB in 66.7 ms = 790 GB/s, about 82 % of the card's 960 GB/s: **the step is DRAM-bandwidth
bound.** Each animal reads 4.3x its own footprint per step (the Newton loop re-reads J, D and H rows every iteration);
at 1,024 the working set mostly sits in the 96 MB Infinity Cache and reads drop to 1x. The lever is reuse, not size:
rows that are never touched (the ~166 empty slots) cost VRAM, not bandwidth. Ranked: (1) per-animal solver state in
LDS/registers across Newton iterations (removes the 4.3x), (2) narrower J/OWC storage (halves each re-read), (3) slot
pooling (VRAM only: ~2x animals per card, no speed).

**Newton bandwidth lane, step 1 (2026-10-26, CPU only).** The "Newton re-reads" attribution above is not measured per
phase: a static count of the Newton loop plus one Hessian rebuild plus noslip gives about 200 KB of the 611 KB. The
bench is K = 0, so it never warm-starts (production builds K > 0). Reads are 99.95 % full 128 B lines. Lane divergence is
a measured candidate: the per-contact loops walk each animal's own pair list, and on the bench's states a wave visits
1.74 distinct pairs per step (3.21 in random order, 1.03 sorted; `tools/wave_coherence.py`). A wave-uniform union walk
would cut those lines to 0.60x (0.42x random) with the same per-animal order. fp16/bf16 storage is out (gate 1e-5);
LDS state is out at 65,536 (step 2 probe). Prereg with zero-code arms: `bench/LANE_UNION_PREREG.md`.
