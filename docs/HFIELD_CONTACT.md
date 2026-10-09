# Heightfield terrain contact (tangent-plane), body engine STAGE 6 extension

## Context
Procedural (Minecraft-style, seeded noise, chunked) worlds need non-flat ground. Engine contacts are plane-mesh only
(`docs/CAPABILITIES.md:11`, "non-plane contacts missing"). Exact MuJoCo parity (`mjc_ConvexHField`) needs a GJK+EPA
port (`mjc_ccd`, XL, divergent waves). Chosen: **tangent plane**. Per mesh-ground pair, sample the heightfield under
the mesh, build the local plane, and run the EXISTING `contacts()` plane-mesh path unchanged. Minimum LOC, near-zero
per-step cost. Known limit: wrong where terrain bends inside one foot (cell size << foot); generator keeps terrain
smooth at foot scale.

## Design
```
model.py export ─► mf: hfield block (nrow, ncol, size x,y,z, base; data float32 row-major)  shared by all animals
                ─► per pair int: kind 0 = plane, 1 = hfield            (append to the 8-int pair record -> 9)
smooth.mojo contacts():
   kind 1 ─► w = pos2 + origin o (floating origin, already passed to step)
           ─► bilinear h(x,y) + gradient (4 taps, same cell)  ─► pos1 = (x, y, h) - o, normal = unit(-hx, -hy, 1)
   kind 0 ─► unchanged (q1 basis z)
   then the whole existing body: support vertex, hull neighbours, frame, Jacobian, Newton. No other edits.
```
Only `pos1` and `normal` change source; ~25 Mojo LOC in `contacts()` + ~30 in `model.py` export/oracle.

## Memory / performance (memory-optimization)
- Hot path: `contacts()` per pair per step. Adds 4 float32 reads (one cell) + ~20 flops, all in registers.
- Zero new per-animal global writes. Hfield is shared, read-only: 256 x 256 float32 = 256 KB, sits in 6 MB L2.
- Measure before/after with `bench.sh` at 1k / 16k / 65,536 animals: us per animal-step, `rocprof-kernels` VGPRs +
  spill bytes for the step kernel. Gate: step <= +5 % vs current 0.64 us, zero new spill bytes. Numbers in commit body.
- float16 heights rejected unless L2 misses show in the trace (256 KB already fits).

## Steps (body-engine 5-step order each)
1. **Export** `model.py`: `terrain(mjcf, heights)` writes an MJCF with an `<hfield>` + ground hfield geom + pairs
   (mirror of `cube()`); `feet()` emits kind + hfield block; sizes `-DNHF_R -DNHF_C`. → verify: export runs on fly and
   spider, `sizes.txt` shows them.
2. **Port** kind-1 branch in `contacts()` (`smooth.mojo:436`), cite `engine_collision_convex.c` mjc_ConvexHField +
   `mjc_PlaneConvex` in docstring and state the tangent-plane substitution. → verify: CPU `mojo build` clean.
3. **Exact gate (1e-5)**: heightfield = a tilted plane (linear heights). Tangent plane is then exact; oracle = MuJoCo
   with the SAME tilted plane as a plane geom. New stage `terrain_tilt` in `STAGES`, `check.sh terrain_tilt`, fly +
   spider, standing states (`model.standing`), float32 floor bar as `contact` stage.
4. **Measured gate vs real MuJoCo hfield**: smooth noise terrain (seeded), oracle = MuJoCo `mjc_ConvexHField`.
   Report per-field error (ncon, qacc, qfrc_constraint); bar = measured value stated, plus behavioural gate: fly and
   spider stand 300 frames on it, max foot penetration < 2 x flat-floor value, tilt < flat + 5 deg.
5. **Mutant**: flip gradient sign (normal tilts wrong way) via `mutant-check`; step 3 gate must FAIL.
6. **Bench** per Memory section; commit with numbers. Update `docs/CAPABILITIES.md` row + `body-engine` skill
   "Contacts" section (tangent-plane limit stated).

Out of scope (next plan): seeded noise generator + chunk streaming around the floating origin (fills the same hfield
buffer; engine needs no change for it).

## Critical files
- `kinestride/smooth.mojo` (`contacts()` :436, pair record `CPH` :68, `step()` origin `o` :919)
- `kinestride/model.py` (`feet()` :125, `cube()` :58 pattern, `STAGES` :41-49)
- `kinestride/check.sh`, `bench.sh`; tools `~/iTools/bin/mj-oracle`, `mutant-check`, `rocprof-kernels`

## Verification
`gpu-wait run --priority 90 --timeout 3600 -- kinestride/check.sh terrain_tilt` (PASS 1e-5, fly + spider),
then `check.sh terrain_noise` (measured bar + stand gate), mutant FAIL, `bench.sh` numbers, and
`live watch 10 terrain` looked at (§14) to see the fly standing on bumps.
