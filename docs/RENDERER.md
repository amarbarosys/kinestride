# Renderer for the body engine

The body engine already holds every animal's pose on the GPU, so its renderer only has to draw: a Mojo
compute rasteriser that reads the poses in place, writes one 64-bit visibility buffer and shades it into
the frame a viewer shows. The fly is 69 mesh parts, 115,158 triangles and 54,905 vertices; sixty-four
flies are 7.4 million triangles a frame. The design keeps the model data under 1 MB on the card for any
number of flies, allocates nothing per frame, and copies no pose: the bytes per frame are the
visibility buffer and the finished image.

## Status

Steps 1 to 5 PASS (`0ad21a9`, `905d2bc`, `86fb4d2`): mesh tables (1.1 MB), 64-bit atomic min (10M writes
0.54 ms), raster and shade, mpv/ffmpeg streaming, silhouette IoU against mujoco.Renderer 0.995-0.998 in
three views from the engine's own pose, 10,000-frame leak gate (fds 7/7, VRAM flat). One fly at 960 x 540:
0.57-1.0 ms per frame including the copy to host. VRAM: 152 MB with the MAX pool off (it was 22.7 GB
with MAX's default pool). Steps 6 (cull, LOD) and 7 (tune) open.

Spectator follow-camera unit 1 is implemented in `kinestride/follow_camera.mojo`. `spectator_check.py`
checks a merged two-root camera in landscape and portrait, and confirms that a one-unit target offset fails.
The checked sphere bound must fit within the narrower horizontal or vertical half-angle, with 5% distance
margin. This gate validates sphere framing math and target centering only; it does not yet exercise recorded
moving engine poses or the renderer integration hook. Camera input bytes are caller-owned root xyz triples plus
one radius per root; camera output is seven float32 values in the existing eye/target/fovy order.

Spectator HUD unit 2 is implemented in `kinestride/hud.mojo`. It takes game-bar values as normalized
scalars and source descriptor XY/kind, and returns pixel fill extents and mini-map marker positions. It does not
own text rendering or resource-state transitions. The standalone fixture checks energy/health/stamina fills of
90/30/60 px and a decoy marker at the independently expected pixel coordinate and kind.

## Data flow

```
model.py export (once)                       engine step (GPU, every physics step)
  meshes: int16 vertices per part, uint16      xpos, xquat per body  ──┐  read in place, no copy
  triangles, part -> body + local pose ───┐                            │
                                          ▼                            ▼
                 cull pass: part bounding sphere vs frustum, one thread per (fly, part)
                                          │ visible (fly, part) list + LOD level
                                          ▼
                 raster pass: one thread per triangle, 64-bit atomicMin
                 into the visibility buffer = depth 32 | fly 12 | triangle 20
                                          ▼
                 (SHADOW=2) the same raster pass from the sun's camera into the shadow map
                                          ▼
                 shade pass: one thread per pixel, ids -> smooth normal (vertex normals,
                 barycentric), sun + sky/ground ambient + highlight in linear light, shadow
                 (map 3x3 PCF | blob | terrain self-shadow), SSAO from vis depths, fog,
                 sky gradient; silhouette pixels: 4 analytic sub-rays (AA=1)
                                          ▼
                 RGBA8 into a pinned host ring (2 frames) ──▶ mpv (live) or ffmpeg (mp4), one pipe
```

## Memory budget

| what | layout | bytes | grows with |
|---|---|---:|---|
| vertices | int16 x 3, quantised to each part's bounding box (scale + offset per part) | 330 KB | nothing |
| vertex normals | int16 x 3 unit x 32767 after the vertices in rv.bin (area-weighted, mesh frame) | 330 KB | nothing |
| triangles | uint16 x 3 per triangle, local to the part (largest part: 1,002 vertices) | 690 KB | nothing |
| part table | body id, local pose (3 + 4), quantisation (6), bounding sphere (4), colour (1) | 6 KB | nothing |
| LOD meshes (step 6) | the same format at 10k and 1k triangles | ~110 KB | nothing |
| poses | the engine's own xpos / xquat rows | 0 extra | animals, already paid |
| visibility buffer | uint64 per pixel, 1920 x 1080 | 16.6 MB | pixels |
| aux layers (shadow map, glass, mirror) | uint64 per pixel each, the view's size (raster is compiled for W x H); `NSM` = their sum | 16.6 MB each at 1080p | pixels |
| image | RGBA8 per pixel, device | 8.3 MB | pixels |
| host ring | 2 pinned RGBA8 frames, reused | 16.6 MB | pixels |

All mesh data is stored once and every fly draws from it; a fly costs no renderer memory beyond the
pose the engine already keeps. int16 vertices over a part's box (at most 2.4 mm) resolve 0.04 um, far
finer than a pixel; uint16 triangles halve the index bytes. The visibility buffer stores ids, not
colours, so the raster pass writes 8 bytes per covered pixel and shading runs once per pixel instead
of once per overlapping triangle.

## No leak, by construction

- Every device buffer, the pinned ring and the output pipe are created once, before the first frame,
  sized by the resolution and the model; a frame allocates nothing on host or device.
- One process owns the pipe; the viewer reads the other end, and the renderer exits on EOF or a broken
  pipe. No `input()`: the stdin fd leak of 2026-10-03 (`1a07a08`) is the scar.
- Gate (`kinestride/verify_render_leak.py`, like `verify_worker_fds.py`): render 10,000 frames,
  sample host RSS, open fds and the process's VRAM (sysfs) at frames 100 and 10,000. They must be equal.

## Performance targets (1920 x 1080, Ultimate-AI)

| scene | target per frame | why |
|---|---:|---|
| 1 fly, close-up | under 0.5 ms | 115k triangles, most a few pixels |
| 64 flies, grid | under 2 ms | 7.4M triangles before culling and LOD |
| 4,096 flies, wide | under 4 ms | only possible with LOD: a fly 10 px tall gets the 1k mesh |

At 60 frames per second the renderer gets one frame every 167 physics steps at dt 0.1 ms, so these
costs stay under 2% of a 65,536-animal step loop.

Spectator live hook (lane-live unit 1): `live.mojo -DFOLLOW=1 -DHUD=1 -DPROFILE=1` with `spectate.mojo`. `follow` runs on the card
after the K-step launch and writes the camera eye and target from the engine's own root xpos rows (no pose copied to the host) with
`follow_camera.fit_distance`, the formula the host `fit_camera` uses. `hud` draws the three bars of animal 0 over the shaded image
from four float32 rows (energy, health, stamina, dead; `smooth.mojo` OGAME layout once the game hook is on); HUD pixels carry alpha
254, alpha 255 stays "animal" for the stand and framing gates. Flags off compile none of it: the frame of the starting state is
byte-equal to the pre-hook `live.mojo` (5faa668). Gate `check.sh spectate` (`spectate_check.py`): hooks off byte-equal; 300-frame clip of
the merged spider walker (`BRAIN=2`, 640 x 360, K=10) framed without clipping and centred in all 300 frames, root travel 6.5 mm;
HUD bar fills 90/30/60 px of 120 for 0.75/0.25/0.5 and grey when dead; frame cost split and leak (fds 8/8, VRAM 147,792 to 145,744 KiB,
RSS +64 KiB over frames 100 and 10,000) in `spectator_integration_check.py`. Mutants: camera target +30 mm, HUD fill one pixel
short, follow compiled in by default. Frame cost (ms, one spider, 640 x 360, K=10, phases synced, so a measurement of the split,
not of fps): physics 40.51, neural_control 0.147, render 0.797, copy 0.149, display 0.277 (run co-running under `--shared`; an exclusive run read physics 37.22, render 0.274); physics is 97 to 99 % of a frame. Animals are
bodies in a shared physical scene (one rendered). Bug fixed on the way: `fit_camera` mixed radians and degrees and framed the
animal about 100 x too far (its fixture repeated the mix); the fixture and a short-distance mutant now catch it.

## Steps

```
1 export ──▶ 2 probe ──▶ 3 raster + shade ──▶ 4 display + leak gate ──▶ 5 check vs MuJoCo ──▶ 6 cull + LOD ──▶ 7 tune
```

1. **Export (S).** model.py writes the mesh tables above for any MJCF; the export refuses meshes over
   65,535 vertices per part. Check: vertices decoded on the CPU are within 1e-4 of the part size of
   MuJoCo's mesh_vert in the geom frame.
2. **Probe the one risky primitive (S).** A 64-bit atomicMin on global memory from Mojo on gfx1100,
   timed on a synthetic 10M-triangle load. If Mojo does not expose it, fall back to 32-bit depth
   atomicMin plus a second pass that writes ids where depth matches.
3. **Raster and shade (M).** Vertices transformed by the engine's pose of the part's body times the
   part's local pose; edge-function raster with a bounding-box loop for small triangles; a separate
   tile path for triangles over 32 x 32 pixels so one large triangle cannot hold a wave. Shading:
   flat normal from the triangle, part colour from the MJCF material, one directional light, a floor.
4. **Display and leak gate (S).** Pinned ring, `mpv` for a live window, `ffmpeg` for mp4;
   verify_render_leak.py as above.
5. **Check against MuJoCo (S).** The same pose rendered by `mujoco.Renderer` and by ours, same camera:
   silhouette IoU at least 0.98 per view (3 views). Then the images are opened and looked at.
6. **Cull and LOD (M).** Per (fly, part) frustum test; LOD by projected size; decimated meshes made
   once at export. Check: the 64-fly frame unchanged in IoU, time down.
7. **Tune (S per knob).** Threads per block, raster bin size, vertex fetch as packed int16 x 4.
   rocprof-kernels before and after: VGPRs, scratch, device time per pass.

Every step reports ms per frame (median of 1,000 frames after a 1 s warm-up), bytes allocated, VRAM
and RSS, with the LACT profile named.

## Look (2026-10-09)

All in `shade`, comptime-selected, no per-frame allocation:

| flag | default | what |
|---|---|---|
| (always) | | gamma 2 (albedo squared in, sqrt out), sky/ground hemisphere ambient, sun at 57 degrees, Blinn highlight on animals, distance fog on the ground only (fogged animals turned grey at world-camera distance), sky gradient |
| (always) | | smooth normals from `rv.bin` vertex normals; where the interpolated normal disagrees with the face by > 60 degrees (two-sided thin parts, wing normals cancel) the face normal is kept |
| (always) | | SSAO: 8 neighbours 0.12 mm away in the visibility buffer, a neighbour 0.004..0.5 mm nearer occludes (darkens leg-body creases and the ground at the feet) |
| `SHADOW` | 2 | 2: shadow map, raster from a near-orthographic sun camera over the view target (50 x the view distance up the sun ray), 3x3 PCF, slope-scaled bias, a texel of the pixel's own triangle never shades it. 1: a Gaussian blob of radius `BLOB_UM` per animal under its root's sun shadow point (no extra pass; loops over animals per ground pixel, so small F). 0: none |
| `TERRAIN` | 0 | adds terrain self-shadow: 16 steps up the sun ray, soft penumbra |
| `CITY` | 0 | the ground is city.mojo's endless city, 1 unit = 1 m; cam[16:20] = seed, origin cell x y, far m (render args 13-16, default 1 0 0 1500). Per pixel: city_hit (unit ray), facades in 4 palettes with windows on 3.5 m floors and 2.5 m bays above the ground floor, roofs, sidewalk tiles, asphalt with a dashed centre line between blocks; sun shadow from one city_hit ray (plus the shadow map for animals), fog exp(-3 d / far). REFLECT off. `CITY_DEPTH=1` (gate) writes each pixel's hit distance in m as float32 bits. `kinestride/city_look.sh` renders a frame (`look`) or checks the depth against city_hit on the CPU (`depth`, city_check frame) |
| (always) | | colour per vertex baked at export (`meshes`): MuJoCo cube textures sampled in each vertex's direction at a 5x5 mip level, times the material rgba (the fly's materials are white; its colour is in textures); rv.bin = position, normal, rgb (RVI = 9 int16 per vertex) |
| `GLASS` | 1 | parts with alpha < 1 (wings, alpha 0.3) rastered into their own layer and blended over the scene at their alpha |
| `REFLECT` | 1 on a flat floor | the opaque scene rastered mirrored in z = 0 into a layer; the floor mixes it at 0.2 (MuJoCo's grid reflectance) |
| `AA` | 1 | silhouette pixels (a 4-neighbour changes animal / ground or jumps 5 % in depth) average 4 rotated-grid sub-rays, each tested against the triangles of the 3x3 neighbourhood, then the ground |

`kinestride/render_look.py OUT N` renders N standing flies (1 close-up, or a grid) at 1920 x 1080 and prints ms
per frame. Measured under `--shared` (a mojo-baro gate job on the card), before -> after:
1 fly 1.36 -> 1.67 ms, 64 flies 1.23 -> 2.76 ms (SHADOW=2) / 2.45 ms (SHADOW=1). Live world in fly-connectome
(fly + spider on terrain, 960 x 540, K=10 physics + render): 78.2 -> 79.0 ms per frame. `render_check` stays
IoU 0.9948 / 0.9981 / 0.9952 (alpha is the centre sample, unchanged by AA).

## Window and live brain view (2026-10-09)

The engine window and its panels come from imkit (`~/Projects/mojo/mojo-imkit`, package `imkit`: SDL3 window, FreeType
atlas, an immediate-mode UI composed on the GPU); fly-connectome's `live.mojo -DGUI=1` draws the scene and the UI layer
in one process. Kinestride itself adds one module for it:

`kinestride/brain_view.mojo`: a brain as points (any dataset: `xyz.bin` float32 N x 3 unit-scaled, `class.bin` uint8,
255 = not drawn), coloured by class, lit by an activity byte per neuron, through an orbit camera, into a
`BRAIN_W x BRAIN_H` RGBA image (default 360 x 270). Two kernels: `splat` (one thread per neuron, additive RGB atomics;
a lit neuron is a small cross) and `tone` (1 - exp, clears the accumulators). Capacity `BRAIN_N` (262,144).
FlyWire, 138,639 neurons: 95.7 us per frame (upload + splat + tone). Check: `brain_view_test` (a lit group's pixels
brighter than at rest, background unchanged; writes rest | lit | all-lit as PPM):

    mojo build -I kinestride kinestride/brain_view_test.mojo -o .work/brain/bv_test
    gpu-wait run --priority 90 --vram 1 --timeout 300 -- .work/brain/bv_test LAYOUT_DIR/ OUT.ppm

The method for a new animal (layout export, the brain worker's activity tap, groups, checks): skill `brain-view`.

## Not in scope

Textures and the fly's own compound-eye view. Walking needs S3 (actuators and adhesion); until then the renderer shows the passive
body, which is still a useful look at contacts.

Keys: depth bits << 32 | animal << TB | triangle, with TB the bits `NTRI` needs (17 for the fly, 18 with the arena's 249k
triangles), so the animal field shrinks as the model grows (`comptime assert F <= 2^(32 - TB)`). `render_check` draws the
arena's box and ellipsoid geoms (`meshes(world=[])`); alpha marks animal parts only. Compared side by side with MuJoCo
(2026-10-09, render_check views): textured golden-brown body, see-through wings, reflections, arena walls; the floor stays
our grey checker (MuJoCo's is its dark grid material). 960 x 540 frame 0.72 -> 1.15 ms with glass, mirror and arena.

## Living World graphics tier (2026-10-17)

With `-DLIVING=1`, terrain and sky shade at half resolution, then bilinear-upscale to the window size. Mesh-hit
pixels stay full resolution. This is spatial upscale, not TAA. At 1280 x 720 it adds 640 x 360 uint32 scratch
(921,600 bytes); the 600-frame walk gate measured 6.26 ms after upscale, down from 6.54 ms with full-resolution
terrain shading. Near tree-slot shadows add sun-ray occlusion and soft ground AO. The soft-crown revision is
still a candidate: visual review accepts it, but isolated timing pairs vary from 7.19 to 9.85 ms; the latest
baseline/patch pair was 6.696/9.847 ms, above the 1.1x baseline cap.

State after review (2026-10-17): half-res upscale committed (d507312). Tree shadows reverted (493cdd3): the hard version drew flat discs and a wrong-scale wedge, the soft version costs ~3 ms. Soft patch parked in living-world `.work/graphics/item4-soft-only.patch`; next try is to shade it inside the half-res pass. Bloom and god rays: two attempts changed 0 pixels in noon/dusk/night; a sun-facing review frame now exists (living-world bb73699). Gate 6.46 ms.

State after gfx2 (2026-10-22): one march per shaded pixel (e322441). `shade()` skips open-sky odd pixels before any march, marches once, and passes the hit to `sample()` as `h0` (also to the low-buffer sample). Walk gate 6.06 to 3.72 ms, images byte-identical, 0 B. Soft tree shadows dropped on the look (Mario, 2026-10-22; +12% over S1). N 256 needs no max-height mip at present: 2.17 ms vs 3.76 ms at N 128 on the same gate. Report: `.work/gfx2/REPORT.md` in the Kinestride checkout.

State after realism pass R (2026-10-22, branch `realism`): living-world shading gained a display-space `post()` (grade,
vignette, grain) at the end of `upscale`, bounce from the surface's albedo in `light()`, large-scale tint in `meadow()`,
`cloud()` (sky layer in `living_sky` and sun-ray shadow in `meadow`), grass sheen, scree specular and water ripples.
`TAA` (default 1 when LIVING): `shade()` shades pixel `phase(sm)` of each 2x2 quad and writes that quad's previous-frame
position to the `OMV` layer; `upscale` calls `resolve()` (history in `OTA`, two RGBA8 frames per uint64, parity
ping-pong; clamp to the low quad; weight 1/4 on the fresh pixel, 1/10 on the rest); `clear_sm` (now takes `cam`) rolls
the camera tail at `OTL` once per frame and zeroes the frame count on a cut. Walk gate 3.79 -> 3.83 ms, sm +8.29 MB.

Perf split (2026-10-22): `cell` and `slot` wrap grid indices with a mask (N and TN are powers of two), not two integer modulos per read: the GPU has no integer divide, and the walk gate went 3.79 -> 1.58 ms with images unchanged (ppm_gate 0 of 57600 in noon, dusk, night; 0 bytes differ in the 7 review frames).

Step D1 (2026-10-22, lane-detail): smooth macro surface. The 20 m two-triangle cells became one uniform cubic B-spline patch per cell over N x N controls built on the host (`lw_ctrl`, appended to the world table: the highest control, the controls, the per-cell bounds; +131,076 B at N 128, world.bin unchanged). C1 across cells, analytic normals, one function for walker and renderer. Why a B-spline: the patch is a convex combination of its 16 controls, so their maximum bounds a cell's ground exactly (Catmull-Rom would overshoot the nodes by about 7 % of every cliff). It does not interpolate: the controls are the nodes deconvolved twice on dry nodes, so the surface stays within 12.1 m of a node at the worst cliff top (mean 0.18 m), and pushed down on wet nodes until the surface there is at or under the sea (the water test is unchanged: height <= sea + 0.02 cells). On the 5 m grid wet nodes go 34,756 -> 35,388 (317 wet -> dry, 949 dry -> wet), 0 wet height nodes turn dry, and the 5 wet regions merge into 2: no river is cut. The march finds the first sign change of the ray over the patch on 4 samples then 8 steps of false position.

Step D4 (2026-10-22, lane-detail): clump crowns. Within `LW_CLUMP` (80 m) of the ray origin a broadleaf crown is a core ellipsoid (0.72 R) plus 5-8 small ellipsoids on a shell at 0.58 R (size 0.30-0.40 R, hashed per tree); a spruce is 3-5 stacked cone skirts, a pine 3-4 stacked discs. The old single envelope stays as the cheap reject test and as the whole crown beyond 80 m, so the far view and the world table are unchanged (0 B). Cost: gate 2.09 -> 2.12 ms (3 interleaved rounds, +0.03 ms).

Step D2 (2026-10-22, lane-detail): ground props. A hash of each 5 m tree-grid cell and the habitat of its middle node picks 0-1 prop (`prop_at`, nothing stored, 0 B): boulders on scree and riparian ground, bushes on meadow and heather, fallen logs under spruce and beech; 0.3-2 m, placed on the D1 surface, never in a tree's slot or within 0.5 m of the water. The ray tests one prop per near-phase step (`prop`, an analytic ellipsoid `blob`, materials 10-12, shaded in render.mojo); beyond NEAR_M the habitat ground colour carries the look (no extra tint). At N 128: 17,017 props (4,256 boulders, 2,910 bushes, 9,851 logs). They do not block the walker. Cost: gate 2.13 -> 2.35 ms (median of 3 rounds, +0.2 ms). Trails are not yet a world layer (D5), so no prop avoids one yet.

Step D6 (2026-10-22, lane-detail): shorelines, from height and `T[1]` alone (0 B). Ground 0-3 m over the sea is wet sand: darker (up to 40 percent) and a touch cooler, wettest at the line. Water is tinted by its depth (sea minus bed): shallows lighter and greener, deep water darker. A foam line sits where the bed is within about one metre along the slope of the waterline (breathing 0.5-1.3 m with a travelling sine and noise), only where the bed is under 0.3 m deep, so a flat lake bed gets none; it fades out by 300 m. Cost: gate 2.33 -> 2.37 ms (3 rounds, +0.05 ms).
