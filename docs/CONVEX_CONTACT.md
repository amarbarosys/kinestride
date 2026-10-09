# Convex contact (box-mesh, ellipsoid-mesh) and exact heightfield, lane-contact

## What MuJoCo 3.9 does (read from the source, receipts in the commits)
- Box-mesh and ellipsoid-mesh pairs go through `mjc_Convex` (engine_collision_convex.c:857) with the NATIVE GJK + EPA (`mjc_ccd`,
  engine_collision_gjk.c), not libccd/MPR. The fly pairs have margin 0.001, so `mjc_Convex`'s own static `maxContacts` is 1 (:833):
  one GJK/EPA contact, then, not for ellipsoids, the perturbation search of :874-927 (both geoms tilted -+1e-3 rad about the two
  tangent axes of the first contact, a new contact kept when farther than 1e-3 min(rbound) from the earlier ones, depth of the
  first): 0, 1 or 5 contacts. Native polygon clipping (multicontact) is not reached for margin > 0 and is not ported.
  (I first read the driver's `mj_maxContact`, a different function, and wrongly called the perturbation path dead; ASTRA-R1 caught it.)
- Heightfield pairs go through `mjc_ConvexHField` (:1120): one GJK + EPA per cell-triangle prism under the mesh's AABB, prism top
  raised by the margin, mesh inflated by it; up to 50 contacts per pair.

## Code
| file | what |
|---|---|
| `convex.mojo` | gjk, subdistance, gjkIntersect, polytope2/3/4, EPA (horizon recursion as an explicit stack), `collide` (perturbation), `collide_hfield` (prisms); float64 on the card (`-DCCD64=1`, set by the export), float32 builds for the CPU gate |
| `smooth.mojo` | `convex_pair`, `hfield_pair` (the contacts take the shared `emit_contact` tail of `contacts()`), slot pool, active-pair list |
| `model.py` | `pairs(..., convex)`, `convex_block` (kinds, sizes, hfield size/origin, `NCV`, `CCD64`, `HFEXACT`, `NPOOL`, `JPOOL`), `arena` states, opt-in `CONVEX=1` / `HFEXACT=1`; default exports unchanged (`cmp_export.sh` PASS) |
| gates | `check.sh convex` `hfield` (narrowphase vs MuJoCo, CPU), `arena` `terrain_exact` (step vs MuJoCo, GPU), each with mutants |
| tools | `convex_gpu.mojo` + `convex_bench.sh` (ns per pair class, VGPR/scratch), `row_diag.py` (engine rows vs MuJoCo efc rows, `DUMP=1`) |

## Results (2026-10-05, fly body-v2 and spider, MuJoCo 3.9.0)
| gate | result |
|---|---|
| `convex` ellipsoid-mesh, 6000 random poses | float64 count equal, worst dist 1.4e-15, pos 8.9e-12, normal 5e-13; float32 within the stated bars (dist 9.4e-7, pos p99 3.6e-4) |
| `convex` box-mesh, 6000 poses, 0/1/5 contacts as MuJoCo | float64 count equal, 0.018 % of values above 1e-9, max 7.8e-6 (EPA tolerance flips); float32 1.5 % contact-count mismatches (float32 cannot resolve EPA's 1e-6 at the 21 mm walls, so a repeated support vertex ends EPA) |
| `hfield`, bumpy seeded terrain vs MuJoCo's own hfield geom, 6000 poses | 21,286 contacts, float64 worst 2.5e-10, up to 34 contacts per pair; float32 passes its bars |
| `arena` (original arena, states placed by contact depth -0.03..0.0008 mm in walls and stones) | ncon equal; contact and actuated within the measured bar REL 3e-3 (qacc 7.1e-4, qacc_euler 2.0e-3); `row_diag`: convex contacts' aref within 6e-5 of the state's largest row, plane contacts 1.3e-4 |
| `convexstep` (unit 4): 20 steps in one launch vs 20 chained `mj_step`, 64 states each; fly in the arena (walls and stones, contact depth -0.03..0.0008 mm), spider on box and ellipsoid stones under its feet (`model.stones`, 3 stones x 50 leg meshes = 150 pairs) | fly: qpos 2.0e-6, qvel 3.0e-4 (17/64 chaotic states dropped); spider: qpos 5.2e-6, qvel 3.2e-5 (1/64 dropped); bar REL 1e-3 (qvel lifted to 1.6e-3 by the float32 floor) |
| `convexstep` forces: full one-step fields on states the ENGINE reached (fly after 3 steps, spider after 1), vs MuJoCo's forward on the same states | fly: qfrc_constraint 1.1e-5, qacc 2.4e-5, qacc_euler 2.9e-5, ncon equal (55/64 states with a convex contact); spider: qfrc_constraint 1.1e-5, qacc 4.9e-6 (4/64 states keep a stone contact: the feet hover at the 1 mm margin and the contact chatters); bar REL 1e-4 (the spider's qfrc_constraint is above the default 1e-5 on plane contacts too) |
| `terrain_exact` (fly and spider, bumpy terrain, step vs MuJoCo's hfield) | ncon equal, contact and actuated within the measured bar REL 4e-4 (fly qacc 2.1e-4, others <= 3.4e-5) |
Mutants caught: contact normal flipped (convex, hfield, arena, terrain_exact, convexstep on fly and spider), one tilt sign dropped (box), prism height test inverted (hfield).

Why the step bars are measured, not 1e-5: arena: the engine's float32 solver on states with a 0.03 mm deep wall or stone contact
(7e-4 qacc, not from the convex contacts, see row_diag). terrain_exact: the engine samples the float32 terrain function, MuJoCo holds a
float32 elevation table, 1e-7 mm apart, 1e-4 of a contact depth (`hfield_check ... func` reproduces it on the CPU; rounding to the table's float32 does not help).

Chained runs on contact are chaotic, so `convexstep` filters, and says so: a state whose own MuJoCo trajectory moves by more than the bar when its start is
perturbed by 1e-6 relative (12 random draws, `CHAOS_SEEDS`) is dropped, at most half, counted in the output. Without it 4 fly states (wall contact
1 vs 5 contacts flips with the perturbation search, then qvel differs by 10-40 % of its range in MuJoCo itself) failed qvel at 0.37 while the median state agreed
to 2e-5; MuJoCo rounded to float32 (the old floor) misses those. The engine's own error on the kept states is 3e-4 (fly) and 3e-5 (spider).

Flight model (`flight.xml`): no `<pair>` elements, its 7 resting contacts are automatic body-body (capsule-cylinder 6, ellipsoid-ellipsoid 1) among about 2,200 candidate
geom pairs of capsule, ellipsoid, cylinder and sphere. Unit 4 does not unblock it: convex.mojo has ellipsoid and mesh supports only, MuJoCo takes capsule-capsule
analytically, and the export reads explicit pairs. That needs capsule, cylinder and sphere supports plus automatic pair generation.

## Cost
- Narrowphase on the card, 6000 random poses, one thread per case: ellipsoid-mesh 510 ns float32 / 670 ns float64; box-mesh 725 / 994 ns. Kernel 192 VGPR, 46 KB scratch per thread, no LDS. Float64 costs 1.35x, so the engine keeps it.
- Memory: a static slot per possible contact (440 convex pairs x up to 5) made a convex-on animal 15x larger and 1.9x slower with no convex contact active
  (1,024 animals 36.3 ms vs 18.8 ms off). Fix: convex and exact-heightfield pairs take slots and Jacobian entries from a per-animal pool each
  step (`NPOOL`/`JPOOL`: 48 and 4,608 entries for the arena, 192 and 18,432 for exact terrain) and the solver loops visit only pairs with a valid slot.
  Arena export: NSLOT 1712 -> 220, NJE 92,236 -> 13,964. A pool overflow drops contacts (the gates' ncon shows it).
- Convex off is unchanged: default exports are byte-identical to main (`cmp_export.sh`) and the walking fly stage 7 at 65,536 is 66.5 ms vs main's 65.3 (65.8 in an earlier run, other lanes share the card, so this is noise).

Bench rows (`bench.sh run body-v2/body.xml 7`, actuated walking fly standing on the floor, exclusive card, per launch):

| animals | convex off | `CONVEX=1` (440 pairs, pooled) | ratio |
|---|---|---|---|
| 1,024 | 19.29 ms | 20.92 ms | 1.08x |
| 16,384 | 35.27 ms | 37.63 ms | 1.07x |
| 65,536 | 66.51 ms | 92.41 ms | 1.39x |

Newton iterations, line-search evaluations and active slots are identical in both columns (these states touch no wall, so no convex pair is active):
the extra time is the pair loop's broad phase and the narrowphase of pairs whose bounding spheres overlap, not contact work. Before the pool the
convex-on column was 36.3 / 67.3 ms at 1,024 / 16,384 animals and 65,536 did not fit in memory.

Step kernel resources (rocprofv3 `--kernel-trace`, 1,024 animals, 64 threads per workgroup, one animal per thread): convex off 128 VGPR, 0 B scratch, 0 LDS;
`CONVEX=1` 192 VGPR, 55,072 B scratch per thread, 0 LDS. The scratch is GJK/EPA's simplex, polytope and mesh hill-climb state, not a register spill of the
convex-off path. Other lanes' shared jobs ran on the card during these runs (the traced convex-on launch came out 7% faster than convex-off, 20.4 vs 21.9 ms),
so the 1.07-1.08x at 1k and 16k is inside the load noise; the 65,536 gap (1.39x) is the only difference that exceeds it and is the figure to quote.

## Not done / limits
- float32 EPA on the 21 mm walls loses 1.5 % of box-mesh contact counts; the engine therefore runs the narrowphase in float64.
- The pool caps per-animal convex contacts (48) and exact-heightfield contacts (192, 16 per pair); MuJoCo keeps up to 50 per hfield pair.
