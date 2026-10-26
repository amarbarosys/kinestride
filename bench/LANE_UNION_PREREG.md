# Prereg: wave-uniform contact walk (Newton bandwidth lane, 2026-10-26)

Frozen before any GPU run. Context and the CPU evidence: `.work/bandwidth-ideas/REPORT.md`.

## Hypothesis

At 65,536 animals the step is DRAM-bound. Part of its 611 KB of reads per animal is lane divergence, not reuse: the
per-contact loops walk each animal's own active-pair list, so the lanes of one wave sit at different pairs and each
distinct pair costs its own 128 B line for every per-contact row. The CPU line model (`tools/wave_coherence.py`, bench
states) gives 1.74 lines per wave per active-pair step in bench order, 3.21 in random order, 1.03 sorted by contact
set. A union walk (every pair any lane holds, ascending, others masked) gives 0.60x the current lines in bench order
and 0.42x in random order. The sorted input gives 0.59x in bench order. So the sorted arm measures the union walk's
payoff with no engine change.

## Arms (no engine code; fly body-v2, stage 7, WAVE=0, same binary per batch size)

| arm | input | what it measures |
|---|---|---|
| C | bench states as stored (`.work/bench/body-7/65536`), K = 0 | control |
| R | same states, columns in a seeded random order (seed 0) | a crowd's divergence cost |
| S | same states, columns sorted by MuJoCo contact pair set | the union walk's payoff (proxy) |
| W | bench order, build `-DK=1` (warm start from OWARM, states evolve over the timed launches) | idea 1, separate verdict |
| P | bench order, STOP = 5 / 9 / 8 / 6 binaries, and STOP 6 again with noslip_iterations 0 in a copy of mi.bin (`mi[CITER + 2]`, index 91853 here) | 9 - 5: Newton start + first H; 8 - 9: first gradient; 6(nit 0) - 8: the Newton loop; 6 - 6(nit 0): noslip |

Harness (XS, written after review): `tools/permute_bench.py DIR OUT ORDER` permutes the columns of qpos.bin, qvel.bin
and ctrl.bin (mi/mf unchanged) and writes the permutation; a check that un-permutes `smooth.bin` columns.

## Metrics

- DRAM read and write bytes per launch: `traffic.sh` (NS=65536), medians of 101 launches, iGPU unbound and
  profile_standard (`~/Brain/OS/rocprofiler-install.md`), restored to `auto` after.
- ms per launch: the bench binary under `auto`, same LACT profile for every arm, order C R S C R S, first run of the
  session discarded (cold clocks), each arm reported as the median of its runs.
- Newton iterations per animal (mean, max) and line-search evaluations: the bench's own print.

## Predictions (frozen)

| quantity | band |
|---|---|
| R / C reads | +5 % to +50 % |
| S / C reads | -3 % to -24 % |
| S / C ms | 0 % to -20 % |
| R / C ms | 0 % to +40 % |
| C, R, S Newton iterations mean | equal (same states; per-animal work does not depend on lane position) |
| S and R `smooth.bin`, un-permuted, vs C | byte-identical |
| W / C Newton iterations | -30 % to -60 % |
| W / C reads | -3 % to -12 % |
| P: (STOP 6 nit 0 - STOP 8) / C reads, the Newton loop alone | 10 % to 40 % |

## Falsifiers

- **Idea dead:** S / C reads above -3 % **and** R / C reads below +5 %. Divergence is then not a byte lever: no union
  walk, and the lane moves to the largest phase in P.
- **Run void:** iteration means differ between C, R and S, or un-permuted outputs are not byte-identical. That would
  mean a cross-lane dependency (a bug), which comes first.
- **Line model wrong:** S / C reads below -24 % (more than the line model's bound): something other than per-contact
  rows also gains from coherence (for example the iteration spread inside a wave). Then the hypothesis is re-derived
  before any implementation.

## Adoption rule (for the implementation lane after review)

Implement the union walk only if S / C ms <= -5 % or R / C ms >= +10 %. It ships only with every check.sh stage PASS,
bench outputs byte-identical to the per-lane walk in C, R and S order, and its own traffic.sh run on C and R inside
the S and R bands above. Pooled convex pairs (NCV > 0) stay on the per-lane walk.

## Run rules

CPU builds first (`mojo-cached`, all arms in parallel), then one `gpu-wait run --priority 90` job per mode: traffic
(profile_standard) and timing (auto). No lock waits or builds inside the hold. About 15 GPU minutes in total. Results
are appended below with the commit that produced them, misses included.

## Results

Run 2026-10-26, kinestride `a36d683` engine (no engine change), harness `tools/permute_bench.py`; GPU job
`.work/bandwidth-ideas/run/time.sh` via gpu-wait, card1 `auto`, LACT Ultimate-AI (290 W). Warmup C 79.13 ms discarded.

**Byte arms not run.** traffic.sh needs the iGPU (0000:18:00.0) unbound, and it drives the DP-3 monitor; Mario chose
to skip. Every reads/writes band below is UNVERIFIED, so the falsifiers stated in bytes have no verdict.

| quantity | band | measured | verdict |
|---|---|---|---|
| C ms (C1, C2) | | 79.54, 79.32 -> 79.43 | |
| R / C ms | 0 % to +40 % | 97.10, 97.42 -> +22.4 % | hit |
| S / C ms | 0 % to -20 % | 72.14, 71.92 -> -9.3 % | hit |
| C, R, S Newton iterations | equal | 3.980 (max 9, 10.23 line-search evaluations) in all six runs | hit |
| S, R `smooth.bin` un-permuted vs C | byte-identical | PASS, 7,576 rows x 65,536 animals each | hit |
| W / C Newton iterations | -30 % to -60 % | 3.980 -> 2.930 = -26.4 % | **miss** (smaller than predicted) |
| W ms (no band) | | 77.90 = -1.9 % (one run) | |
| R / C, S / C, W / C reads | as frozen | not measured | UNVERIFIED |
| P Newton loop share of reads | 10 % to 40 % | not measured; time share 36.2 % | UNVERIFIED |

Phase split by time (one run each, same job; P6n0 = STOP 6 with noslip_iterations 0):

| phase | ms | share of C |
|---|---:|---:|
| kinematics through constraints (P5) | 14.67 | 18.5 % |
| Newton start + first Hessian (P9 - P5) | 15.38 | 19.4 % |
| first gradient (P8 - P9) | 1.87 | 2.4 % |
| Newton loop (P6n0 - P8) | 28.73 | 36.2 % |
| noslip, 5 sweeps (P6 - P6n0) | 14.24 | 17.9 % |
| Euler (C - P6) | 4.54 | 5.7 % |

**Verdict.** Lane divergence is real and costs time: the same states, with the same per-animal work and byte-identical
results, take 22.4 % longer in crowd order and 9.3 % less sorted. The adoption rule on time is met on both counts
(S / C <= -5 %, R / C >= +10 %), so the union walk qualifies for an implementation lane. Its ship rule still asks for a
traffic.sh run on C and R, which needs a counter pass with the session off DP-3, or a ms-only ship rule decided at
review. Warm start missed its iteration band (-26 %) and moves time by -1.9 %: not a lever. Noslip (14.2 ms) costs as
much as the Newton start, far above the static estimate.
