"""Renderer for the body engine (docs/RENDERER.md): a compute rasteriser over the engine's own poses.

Poses are read in the engine's layout: rows [row][animal], xpos of bodies 1.. in rows 0..3(NB-1), xquat from
row 3(NB-1); the buffer is the engine's own (here: its smooth.bin), so drawing copies no pose. Mesh tables come
from model.py meshes (int16 vertices per part box, uint16 triangles, uint8 part per triangle).

Per frame, five passes and no allocation:
  suncam  one thread: the sun's camera (near-orthographic, over the view target) from the view camera
  clear   visibility buffer and shadow map = all ones
  raster  one thread per (animal, triangle): world vertices from body pose x part pose, perspective, edge
          functions over the triangle's pixel box, 64-bit atomic min of (depth bits << 32 | animal << TB | tri);
          run twice, from the view camera into vis and (SHADOW=2) from the sun into the shadow map sm
  shade   one thread per pixel, in linear light (gamma 2): smooth normal and colour (vertex normals and texture-baked
          vertex colours from rv.bin, barycentric),
          sun + sky/ground ambient + highlight, sun shadow (map with 3x3 PCF, or a blob per animal, plus terrain
          self-shadow), screen-space ambient occlusion from vis depths, distance fog, sky gradient; silhouette pixels
          (AA=1) average 4 analytic sub-rays against the triangles of their 3x3 neighbourhood
then the RGBA8 image goes to a pinned host buffer allocated once.

usage: render RENDER_DIR/ POSES.bin OUT FRAMES ex ey ez tx ty tz fovy_deg spacing [seed ox oy far: -DCITY=1]
       OUT: .ppm image, .rgba raw (alpha 255 on animals), or - to stream raw RGBA frames to stdout while the
       camera orbits the target 1 degree a frame (FRAMES 0 = until the reader closes the pipe), e.g.
       render ... - 0 ... | mpv --demuxer=rawvideo --demuxer-rawvideo-w=W --demuxer-rawvideo-h=H
                                --demuxer-rawvideo-mp-format=rgba -
       spacing 0 draws animals where they are; spacing s > 0 lays them on a grid of s, root over the cell centre
"""
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.math import cos, exp, floor, log, pow, sin, sqrt, tan
from std.memory import bitcast
from std.os import getenv
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from world import NP, PW, height_grad
from city import city_hit
from living import atan2_gpu, cell, lw_albedo, lw_aurora, lw_grass, lw_hit, lw_lit, lw_trail, rnd, sky_cam, trail_d

comptime W = get_defined_int["W", 1920]()
comptime H = get_defined_int["H", 1080]()
comptime NB = get_defined_int["NBODY"]()
comptime F = get_defined_int["FLIES"]()  # animals in the pose buffer, all drawn
comptime PROWS = get_defined_int["PROWS"]()  # rows of the pose buffer
comptime NPART = get_defined_int["NPART"]()
comptime NRV = get_defined_int["NRV"]()
comptime NRT = get_defined_int["NRT"]()  # triangle indices (3 per triangle)
comptime NTRI = NRT // 3


def _bits(n: Int) -> Int:
    var b = 1
    while (1 << b) < n:
        b += 1
    return b


comptime TB = _bits(NTRI)  # key = depth bits << 32 | animal << TB | triangle: the triangle field sized to the model
comptime TB64 = UInt64(TB)
comptime TMASK = UInt64((1 << TB) - 1)
comptime PFS = 21  # part floats: pos 3, quat 4, box offset 3, box scale 3, sphere centre 3, radius 1, rgba 4
comptime GX = get_defined_int["GX", 1]()
comptime BOX = Float32(get_defined_int["BOX_MM", 0]())  # > 0: the camera is inside a closed cube of this side (model.cube)  # grid columns when spacing > 0
comptime OXQ = 3 * (NB - 1)
comptime EMPTY = UInt64.MAX
comptime NEAR = Float32(1e-3)

comptime LVIS = row_major[W * H]()
comptime LPI = row_major[5 * NPART]()
comptime NBIRD = 9  # gulls (the page's flock)
comptime BGP = 48  # outline points of a gull; the outline is BGP - 2 = NTB triangles (the host's strip between its two edges)
comptime NTB = BGP - 2
comptime BLK = get_defined_int["LIVING", 0]() * (2 * BGP + 3 * NTB)  # LIVING: the gull's outline after the parts, written once: x, y (m) of its BGP points, then the point indices of its NTB triangles
comptime LWO = PFS * NPART + BLK  # where the world table starts in pf
comptime LPF = row_major[LWO + get_defined_int["LIVING", 0]() * 8]()  # parts, the flock, then the world header; the table runs on past it (LWF)
comptime RVI = 9  # int16 per vertex in rv.bin: position 3, unit normal x 32767 3, rgb 0..255 3 (callers load 2 * RVI * NRV bytes)
comptime LRV = row_major[NRV * RVI]()  # all positions, then all normals, then all colours
comptime LRT = row_major[NRT]()
comptime LRP = row_major[NTRI]()
comptime LPOSE = row_major[PROWS * F]()
comptime CAMB = 16 + NP  # LIVING: per gull, from cam[CAMB]: centre x y z (m), heading x y, flap, 2 spare
comptime NCAM = CAMB + get_defined_int["LIVING", 0]() * 8 * NBIRD  # eye 3, target 3, fovy, focal, ..., [10] grid spacing; [16:] world.mojo terrain params
comptime LCAM = row_major[NCAM]()
comptime SOIL = get_defined_int["SOIL", 0]() == 1  # play: soil and moss on the terrain instead of the grey checker (gates keep the checker)
comptime TERRAIN = get_defined_int["TERRAIN", 0]()  # 1: the ground is world.mojo's terrain (cam[16:24]), not z = 0
comptime CITY = get_defined_int["CITY", 0]()  # 1: the ground is city.mojo's endless city, 1 unit = 1 m (cam[16:20]: seed, origin cell x y, far m)
comptime LIVING = get_defined_int["LIVING", 0]()  # 1: the ground is living.mojo's baked torus (world.bin after the parts in pf; cam[16:32] the sky)
comptime LWF = 8 if LIVING else 0  # world floats the layout covers: the header only. The table is as long as world.bin, read through T = pf + LWO
comptime BIRD0 = PFS * NPART  # where the gull outline starts in pf
comptime LIVING_FAR = get_defined_int["LIVING_FAR", 125]()  # view distance in cells per 128 of the world's N (125 x 20 m = 2.5 km at N 128, private note SCALE)
comptime CITY_DEPTH = get_defined_int["CITY_DEPTH", 0]()  # 1 (gate): each pixel's city hit distance in m as float32 bits, 1e30 on none
comptime SHADOW = get_defined_int["SHADOW", 2]()  # 0 none, 1 a soft blob under each animal (crowds: no extra pass), 2 shadow map
comptime AA = get_defined_int["AA", 1]()  # 1: silhouette pixels average 4 analytic sub-rays
comptime GLASS = get_defined_int["GLASS", 1]()  # 1: see-through parts (alpha < 1: wings) in their own layer, blended over the scene
comptime REFLECT = get_defined_int["REFLECT", 1]() if TERRAIN == 0 and CITY == 0 and LIVING == 0 and get_defined_int["BOX_MM", 0]() == 0 else 0  # flat floor mirror
comptime REFL = Float32(0.2)  # floor reflectance (MuJoCo's grid material)
comptime OGL = W * H if SHADOW == 2 else 0  # aux layers in sm, each W x H (raster is compiled for W x H): shadow map at 0,
comptime OMR = OGL + (W * H if GLASS else 0)  # then the glass layer at OGL, the mirror layer at OMR
comptime NAUX = OMR + (W * H if REFLECT else 0)
comptime TAA = get_defined_int["TAA", 1]() if LIVING else 0  # 1: temporal AA; the half-res background shades one pixel per 2x2 quad per frame, in turn
comptime OTA = NAUX  # TAA layers after the cleared ones: history (two RGBA8 frames per uint64, frame-parity ping-pong),
comptime OMV = OTA + (W * H if TAA else 0)  # each quad's background one frame ago (x, y in 1/8 px, uint16 each, as uint32),
comptime OTL = OMV + ((HW * HH + 1) // 2 if TAA else 0)  # float32 tail: last frame's camera, this camera, frame count, history ok
comptime NTA = OTL + (NCAM + 2 if TAA else 0)
comptime NSM = NTA if NTA > 0 else 1  # sm buffer size (uint64), what callers allocate
comptime BLOB = Float32(get_defined_int["BLOB_UM", 1200]()) * 1e-3  # SHADOW=1 blob radius, mm
comptime TVIS = TileTensor[DType.uint64, type_of(LVIS), MutAnyOrigin]
comptime TIMG = TileTensor[DType.uint32, type_of(LVIS), MutAnyOrigin]
comptime HW = (W + 1) // 2
comptime HH = (H + 1) // 2
comptime LLOW = row_major[HW * HH]()
comptime TLOW = TileTensor[DType.uint32, type_of(LLOW), MutAnyOrigin]
comptime TPI = TileTensor[DType.int32, type_of(LPI), MutAnyOrigin]
comptime TPF = TileTensor[DType.float32, type_of(LPF), MutAnyOrigin]
comptime TRV = TileTensor[DType.int16, type_of(LRV), MutAnyOrigin]
comptime TRT = TileTensor[DType.uint16, type_of(LRT), MutAnyOrigin]
comptime TRP = TileTensor[DType.uint8, type_of(LRP), MutAnyOrigin]
comptime TPOSE = TileTensor[DType.float32, type_of(LPOSE), MutAnyOrigin]
comptime TCAM = TileTensor[DType.float32, type_of(LCAM), MutAnyOrigin]
comptime V3 = SIMD[DType.float32, 4]
comptime Q4 = SIMD[DType.float32, 4]



@always_inline
def rot(q: Q4, v: V3) -> V3:
    """MuJoCo mju_rotVecQuat, quaternion w, x, y, z."""
    var t = V3(q[0] * v[0] + q[2] * v[2] - q[3] * v[1], q[0] * v[1] + q[3] * v[0] - q[1] * v[2], q[0] * v[2] + q[1] * v[1] - q[2] * v[0], 0)
    return V3(v[0] + 2 * (q[2] * t[2] - q[3] * t[1]), v[1] + 2 * (q[3] * t[0] - q[1] * t[2]), v[2] + 2 * (q[1] * t[1] - q[2] * t[0]), 0)


@always_inline
def cross(a: V3, b: V3) -> V3:
    return V3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0)


@always_inline
def dot(a: V3, b: V3) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@always_inline
def vertex(pi: TPI, pf: TPF, rv: TRV, rt: TRT, pose: TPOSE, cam: TCAM, f: Int, p: Int, k: Int) -> V3:
    """World position of triangle index k (absolute) of part p for animal f, plus the grid offset."""
    var vi = Int(pi[5 * p + 1]) + Int(rt[k])
    var o = PFS * p
    var local = V3(Float32(rv[3 * vi]) * pf[o + 10] + pf[o + 7], Float32(rv[3 * vi + 1]) * pf[o + 11] + pf[o + 8],
                   Float32(rv[3 * vi + 2]) * pf[o + 12] + pf[o + 9], 0)
    var inbody = rot(Q4(pf[o + 3], pf[o + 4], pf[o + 5], pf[o + 6]), local) + V3(pf[o], pf[o + 1], pf[o + 2], 0)
    var b = Int(pi[5 * p]) - 1
    if b < 0:  # a world-body part (stone, wall, plant, food marker): fixed, its part pose is already in world mm
        return inbody
    var bq = Q4(pose[(OXQ + 4 * b) * F + f], pose[(OXQ + 4 * b + 1) * F + f], pose[(OXQ + 4 * b + 2) * F + f], pose[(OXQ + 4 * b + 3) * F + f])
    var bp = V3(pose[(3 * b) * F + f], pose[(3 * b + 1) * F + f], pose[(3 * b + 2) * F + f], 0)
    var w = rot(bq, inbody) + bp
    var spacing = cam[10]
    if spacing > 0:  # grid: root body 1 over the centre of cell f
        w += V3(Float32(f % GX) * spacing - pose[f], Float32(f // GX) * spacing - pose[F + f], 0, 0)
    return w


@always_inline
def camera(cam: TCAM) -> Tuple[V3, V3, V3, V3, Float32]:
    """Eye, right, up, forward and focal length in pixels from cam = eye 3, target 3, fovy, ..."""
    return basis(V3(cam[0], cam[1], cam[2], 0), V3(cam[3], cam[4], cam[5], 0), cam[7])


@always_inline
def basis(eye: V3, target: V3, focal: Float32) -> Tuple[V3, V3, V3, V3, Float32]:
    var fw = target - eye
    fw = fw / sqrt(dot(fw, fw))
    var right = cross(fw, V3(0, 0, 1, 0))
    right = right / sqrt(dot(right, right))
    var up = cross(right, fw)
    return (eye, right, up, fw, focal)


@always_inline
def phase(sm: TVIS) -> Tuple[Int, Int, Int]:
    """TAA: the frame count since the last cut and which pixel of each 2x2 quad the half-res background shades this
    frame, cycling (0,0) (1,1) (0,1) (1,0); without TAA always (0,0)."""
    comptime if TAA:
        var f = Int(sm.ptr.unsafe_offset(OTL).bitcast[Float32]()[2 * NCAM])
        return (f, f & 1, ((f >> 1) ^ f) & 1)
    return (0, 0, 0)


def clear(vis: TVIS):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < W * H:
        vis[i] = EMPTY


@always_inline
def fill(vis: TVIS, base: Int32, id: UInt64, sx: V3, sy: V3, sz: V3):
    """The triangle (screen x, y, 1 / view z per corner) into vis at offset base: edge functions over its pixel box, 64-bit atomic min of
    (depth bits << 32 | id)."""
    var area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0])
    if abs(area) < 1e-12:
        return
    var x0 = max(Int(min(sx[0], min(sx[1], sx[2]))), 0)
    var x1 = min(Int(max(sx[0], max(sx[1], sx[2]))) + 1, W - 1)
    var y0 = max(Int(min(sy[0], min(sy[1], sy[2]))), 0)
    var y1 = min(Int(max(sy[0], max(sy[1], sy[2]))) + 1, H - 1)
    for y in range(y0, y1 + 1):
        var py = Float32(y) + 0.5
        for x in range(x0, x1 + 1):
            var px = Float32(x) + 0.5
            var w0 = ((sx[2] - sx[1]) * (py - sy[1]) - (sy[2] - sy[1]) * (px - sx[1])) / area
            var w1 = ((sx[0] - sx[2]) * (py - sy[2]) - (sy[0] - sy[2]) * (px - sx[2])) / area
            var w2 = 1 - w0 - w1
            if w0 < 0 or w1 < 0 or w2 < 0:
                continue
            var depth = 1 / (w0 * sz[0] + w1 * sz[1] + w2 * sz[2])
            var key = UInt64(depth.to_bits()) << 32 | id
            _ = Atomic.min[ordering=Ordering.RELAXED](vis.ptr.unsafe_offset(Int(base) + y * W + x), key)


comptime BANK_S = Float32(-0.479)  # a gull rolls 0.5 rad, the outer wing (+x, the flock circles counter-clockwise) down: a walker under the circle looks up along the wing axis, and the roll turns the wings' underside to it instead of their edge
comptime BANK_C = Float32(0.878)


@always_inline
def bird_vertex(pf: TPF, cam: TCAM, b: Int, k: Int) -> V3:
    """World position (m) of outline point k of gull b. The host writes each gull's state (centre, heading,
    flap) into cam every frame and the outline into pf once; the wings lift with |x| times the flap, as the page's gulls, and the body rolls by BANK."""
    var s = CAMB + 8 * b
    var o = BIRD0 + 2 * k
    var x = pf[o]
    var y = pf[o + 1]
    var fx = cam[s + 3]
    var fy = cam[s + 4]
    return V3(cam[s] + x * BANK_C * fy + y * fx, cam[s + 1] - x * BANK_C * fx + y * fy, cam[s + 2] + x * BANK_S + abs(x) * 0.55 * cam[s + 5], 0)


comptime BIRDF = UInt64((1 << (32 - TB)) - 2)  # the animal field of a gull's key: no animal and not EMPTY's all ones; the triangle field holds gull * NTB + triangle


@always_inline
def isbird(key: UInt64) -> Bool:
    comptime if LIVING:
        return key != EMPTY and ((key >> TB64) & ((UInt64(1) << UInt64(32 - TB)) - 1)) == BIRDF
    return False


@always_inline
def birdcol(cam: TCAM) -> V3:
    """Gull ink, linear: dark by day, pale by night (the page's colours)."""
    var n = min(max((cam[30] - 0.4) / 0.6, Float32(0)), Float32(1))
    var c = V3(0.08, 0.08, 0.1, 0) * (1 - n) + V3(0.75, 0.8, 0.95, 0) * n
    return c * c


def raster(vis: TVIS, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM, base: Int32, mode: Int32, birds: Int32):
    """Into vis at offset base: mode 0 the opaque parts, 1 the see-through parts (alpha < 1) only, 2 the opaque parts mirrored in z = 0.
    birds 1 (LIVING, the view camera): the threads past the animals' triangles draw the flock, NBIRD x NTB triangles."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx >= F * NTRI:
        comptime if LIVING:
            var k = idx - F * NTRI
            if birds == 1 and k < NBIRD * NTB:
                var c = camera(cam)
                var sx = V3(0)
                var sy = V3(0)
                var sz = V3(0)
                for j in range(3):
                    var w = bird_vertex(pf, cam, k // NTB, Int(pf[BIRD0 + 2 * BGP + 3 * (k % NTB) + j]))
                    var d = w - c[0]
                    var z = dot(d, c[3])
                    if z < NEAR:
                        return
                    sx[j] = Float32(W) * 0.5 + c[4] * dot(d, c[1]) / z
                    sy[j] = Float32(H) * 0.5 - c[4] * dot(d, c[2]) / z
                    sz[j] = 1 / z
                fill(vis, base, BIRDF << TB64 | UInt64(k), sx, sy, sz)
        return
    var f = idx // NTRI
    var t = idx % NTRI
    var p = Int(rp[t])
    if f > 0 and pi[5 * p] == 0:  # a world part (body 0) is the same for every animal: animal 0 draws it
        return
    if (pf[PFS * p + 20] < 0.999) != (mode == 1):
        return
    var c = camera(cam)
    var sx = SIMD[DType.float32, 4](0)
    var sy = SIMD[DType.float32, 4](0)
    var sz = SIMD[DType.float32, 4](0)
    for k in range(3):
        var w = vertex(pi, pf, rv, rt, pose, cam, f, p, 3 * t + k)
        if mode == 2:
            w[2] = -w[2]
        var d = w - c[0]
        var z = dot(d, c[3])
        if z < NEAR:
            return
        sx[k] = Float32(W) * 0.5 + c[4] * dot(d, c[1]) / z
        sy[k] = Float32(H) * 0.5 - c[4] * dot(d, c[2]) / z
        sz[k] = 1 / z
    fill(vis, base, UInt64(f << TB | t), sx, sy, sz)


@always_inline
def terrain(cam: TCAM) -> PW:
    var p = PW()
    for k in range(NP):
        p[k] = cam[16 + k]
    return p


@always_inline
def hash2(x: Int, y: Int) -> Float32:
    var h = UInt32(x * 374761393 + y * 668265263)
    h = (h ^ (h >> 13)) * 1274126177
    return Float32(h ^ (h >> 16)) / 4294967295.0


@always_inline
def vnoise(x: Float32, y: Float32) -> Float32:
    """value noise in 0..1, one cell per unit."""
    var ix, iy = Int(x + 1e4) - 10000, Int(y + 1e4) - 10000
    var fx, fy = x - Float32(ix), y - Float32(iy)
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    var a = hash2(ix, iy) + (hash2(ix + 1, iy) - hash2(ix, iy)) * fx
    var b = hash2(ix, iy + 1) + (hash2(ix + 1, iy + 1) - hash2(ix, iy + 1)) * fx
    return a + (b - a) * fy


@always_inline
def soil(x: Float32, y: Float32) -> V3:
    """soil (brown, grain at 0.3 mm) with moss patches (green, 6 mm blobs)."""
    var grain = 0.55 * vnoise(x * 3.3, y * 3.3) + 0.3 * vnoise(x * 11, y * 11) + 0.15 * vnoise(x * 37, y * 37)
    var moss = vnoise(x * 0.17 + 31, y * 0.17 + 7) * 0.7 + vnoise(x * 0.6, y * 0.6) * 0.3
    var m = min(max((moss - 0.5) * 5, Float32(0)), Float32(1))
    var dirt = V3(0.36, 0.26, 0.17, 0) * (0.7 + 0.6 * grain)
    var green = V3(0.24, 0.40, 0.15, 0) * (0.75 + 0.5 * vnoise(x * 6, y * 6))
    return dirt * (1 - m) + green * m


def ground(cam: TCAM, eye: V3, ray: V3) -> Float32:
    """Ray parameter of the first terrain hit (3.4e38: none within FAR mm). Octaves 0 is a plane: solved exactly
    (so a flat world renders as the old z = 0 floor); so is amplitude 0. Else 96 steps, denser near the eye, then 10 bisections on
    eye z - height."""
    var p = terrain(cam)
    if (Int(p[4]) == 0 or p[2] == 0) and p[8] == 0:  # no noise, no landmarks: a plane
        var den = ray[2] - p[0] * ray[0] - p[1] * ray[1]
        var t = (p[0] * eye[0] + p[1] * eye[1] - eye[2]) / den if den != 0 else Float32(-1)
        return t if t > 0 else Float32(3.4e38)
    var far = Float32(300) / sqrt(dot(ray, ray))
    var lo = Float32(0)
    for k in range(1, 97):
        var s = Float32(k) / 96
        var t = far * s * s
        var x = eye + ray * t
        if x[2] < height_grad(x[0], x[1], p)[0]:
            var hi = t
            for _ in range(10):
                var m = 0.5 * (lo + hi)
                var y = eye + ray * m
                if y[2] < height_grad(y[0], y[1], p)[0]:
                    hi = m
                else:
                    lo = m
            return hi
        lo = t
    return Float32(3.4e38)


@always_inline
def sun(cam: TCAM) -> V3:
    """Unit direction to the light: the sun, 57 degrees up; LIVING: cam[17:20], the sun by day and the moon by night."""
    comptime if LIVING:
        return V3(cam[17], cam[18], cam[19], 0)
    return V3(0.4504, 0.3003, 0.8408, 0)


@always_inline
def unit(v: V3) -> V3:
    return v / max(sqrt(dot(v, v)), Float32(1e-30))


@always_inline
def sun_view(cam: TCAM) -> Tuple[V3, V3, V3, V3, Float32]:
    """The sun's camera as camera() returns it: 50 x the view distance up the sun ray from the view target,
    focal so the map spans 2 x 1.5 view distances vertically (near-orthographic)."""
    var t = V3(cam[3], cam[4], cam[5], 0)
    var e = V3(cam[0], cam[1], cam[2], 0)
    var r = max(sqrt(dot(e - t, e - t)), Float32(1e-3))
    var fw = -sun(cam)
    var right = unit(cross(fw, V3(0, 0, 1, 0)))
    return (t + sun(cam) * (50 * r), right, cross(right, fw), fw, Float32(H) * 0.5 * 50 / 1.5)


def suncam(cam: TCAM, scam: TCAM):
    """Scam = cam with the sun's eye and focal, so raster() draws the shadow map with the view's grid and terrain."""
    var s = sun_view(cam)
    for k in range(NCAM):
        scam[k] = cam[k]
    for a in range(3):
        scam[a] = s[0][a]
    scam[7] = s[4]


@always_inline
def vnormal(pi: TPI, pf: TPF, rv: TRV, rt: TRT, pose: TPOSE, f: Int, p: Int, k: Int) -> V3:
    """World vertex normal (x 32767, not unit) at triangle index k of part p, animal f."""
    var vi = 3 * (NRV + Int(pi[5 * p + 1]) + Int(rt[k]))
    var o = PFS * p
    var n = rot(Q4(pf[o + 3], pf[o + 4], pf[o + 5], pf[o + 6]), V3(Float32(rv[vi]), Float32(rv[vi + 1]), Float32(rv[vi + 2]), 0))
    var b = Int(pi[5 * p]) - 1
    if b >= 0:
        n = rot(Q4(pose[(OXQ + 4 * b) * F + f], pose[(OXQ + 4 * b + 1) * F + f], pose[(OXQ + 4 * b + 2) * F + f], pose[(OXQ + 4 * b + 3) * F + f]), n)
    return n


@always_inline
def vcolour(pi: TPI, rv: TRV, rt: TRT, p: Int, k: Int) -> V3:
    """Display rgb 0..1 at triangle index k of part p (meshes(): material x texture, per vertex)."""
    var vi = 3 * (2 * NRV + Int(pi[5 * p + 1]) + Int(rt[k]))
    return V3(Float32(rv[vi]), Float32(rv[vi + 1]), Float32(rv[vi + 2]), 0) / 255


@always_inline
def tri_hit(o: V3, d: V3, a: V3, b: V3, c: V3) -> V3:
    """(t, u, v) of the ray o + t d through triangle abc (Moller-Trumbore); t = 3.4e38 on a miss."""
    var miss = V3(3.4e38, 0, 0, 0)
    var e1 = b - a
    var e2 = c - a
    var pv = cross(d, e2)
    var det = dot(e1, pv)
    if det == 0:
        return miss
    var inv = 1 / det
    var tv = o - a
    var u = dot(tv, pv) * inv
    var qv = cross(tv, e1)
    var v = dot(d, qv) * inv
    var t = dot(e2, qv) * inv
    if u < 0 or v < 0 or u + v > 1 or t <= 0:
        return miss
    return V3(t, u, v, 0)


@always_inline
def depth(key: UInt64) -> Float32:
    return bitcast[DType.float32](UInt32(key >> 32))


@always_inline
def city(cam: TCAM, eye: V3, ray: V3) -> SIMD[DType.float32, 8]:
    """city.mojo's first surface along ray (any length): t as this ray's parameter (3.4e38: none), normal, material."""
    var l = sqrt(dot(ray, ray))
    var h = city_hit(UInt32(Int(cam[16])), Int(cam[17]), Int(cam[18]), eye, ray / l, cam[19])
    h[0] = h[0] / l if h[0] < 1e30 else Float32(3.4e38)
    return h


@always_inline
def street(sm: TVIS, cam: TCAM, eye: V3, ray: V3, h: SIMD[DType.float32, 8], ao: Float32) -> V3:
    """Linear colour of a city surface: facades (4 palettes) with windows on 3.5 m floors and 2.5 m bays above the
    ground floor, roofs, sidewalk tiles, asphalt with a dashed centre line between blocks; the sun shadow from one
    ray through city_hit (and the shadow map for the animals); fog to cam[19]."""
    var hit = eye + ray * h[0]
    var n = V3(h[1], h[2], h[3], 0)
    var mat = Int(h[4])
    var base = V3(0.22, 0.22, 0.23, 0)  # asphalt
    var spec = Float32(0)
    if mat >= 3:
        base = V3(0.62, 0.58, 0.52, 0) if mat == 3 else (V3(0.45, 0.47, 0.52, 0) if mat == 4 else (V3(0.66, 0.42, 0.32, 0) if mat == 5 else V3(0.78, 0.76, 0.70, 0)))
        if n[2] > 0.5:
            base = V3(0.42, 0.42, 0.40, 0)
        elif abs(n[2]) < 0.5:
            var u = hit[1] * n[0] - hit[0] * n[1]  # along the wall
            var fu = u / 2.5 - floor(u / 2.5)
            var fz = hit[2] / 3.5 - floor(hit[2] / 3.5)
            if hit[2] > 3.5 and fu > 0.18 and fu < 0.82 and fz > 0.25 and fz < 0.85:
                base = V3(0.16, 0.20, 0.26, 0)
                spec = 0.6
    elif mat == 2:
        var tile = (Int(floor(hit[0] * 0.5)) + Int(floor(hit[1] * 0.5))) % 2
        base = V3(0.66, 0.65, 0.62, 0) * (Float32(0.92) if tile == 0 else Float32(1))
    else:  # a street between blocks (not a crossing) gets its centre line: the cell edge, + 4 m on an avenue
        var ex = floor(hit[0] / 64 + 0.5)
        var ey = floor(hit[1] / 64 + 0.5)
        var u = hit[0] - (ex * 64 + (Float32(4) if (Int(ex) + Int(cam[17])) % 8 == 0 else Float32(0)))
        var v = hit[1] - (ey * 64 + (Float32(4) if (Int(ey) + Int(cam[18])) % 8 == 0 else Float32(0)))
        var dash = hit[0] / 6 - floor(hit[0] / 6) < 0.5 if abs(v) < abs(u) else hit[1] / 6 - floor(hit[1] / 6) < 0.5
        if dash and min(abs(u), abs(v)) < 0.12 and max(abs(u), abs(v)) > 12:
            base = V3(0.85, 0.75, 0.30, 0)
    var lit = Float32(1)
    if city(cam, hit + n * 0.05, sun(cam))[0] < 3.4e38:
        lit = 0
    comptime if SHADOW == 2:
        lit = min(lit, shadow_map(sm, cam, hit, EMPTY, dot(n, sun(cam))))
    var col = light(cam, base * base, n, unit(ray), lit, ao, spec)
    var fade = exp(-3 * h[0] * sqrt(dot(ray, ray)) / cam[19])
    return col * fade + V3(0.36, 0.4225, 0.5184, 0) * (1 - fade)


@always_inline
def living(cam: TCAM, pf: TPF, eye: V3, ray: V3) -> SIMD[DType.float32, 8]:
    """living.mojo's first surface along ray (any length), in metres (cam[16] per world unit): t as this ray's parameter."""
    var l = sqrt(dot(ray, ray))
    var h = lw_hit(pf.ptr.unsafe_offset(LWO), eye / cam[16], ray / l, pf[LWO] * Float32(LIVING_FAR) / 128)  # scales with the world's N
    h[0] = h[0] * cam[16] / l if h[0] < 1e30 else Float32(3.4e38)
    return h


@always_inline
def atmosphere(cam: TCAM, view: V3) -> V3:
    """Rayleigh blue scatter and forward Mie glow over the time-of-day sky gradient."""
    var f = min(max(view[2] / 0.8, Float32(0)), Float32(1))
    var c = V3(cam[23], cam[24], cam[25], 0) * (1 - f) + V3(cam[20], cam[21], cam[22], 0) * f
    c = c * c
    var day = 1 - cam[30]
    var mu = dot(view, sun(cam))
    var rayleigh = 0.08 * (1 + mu * mu) * day
    var mie = 0.002 * (1 - 0.8 * 0.8) / pow(max(1 + 0.8 * 0.8 - 1.6 * mu, Float32(0.02)), Float32(1.5)) * day
    var air = 0.2 + 0.8 * exp(-max(view[2], Float32(0)) * 2)
    return c + V3(0.08, 0.22, 0.58, 0) * (rayleigh * air) + V3(1, 0.46, 0.2, 0) * mie


@always_inline
def cloud_at(x: Float32, y: Float32, t: Float32) -> Float32:
    """Cover 0..1 of the cloud layer over (x, y) m at time t (s): three value-noise octaves from 600 m to 80 m (ragged
    edges), drifting with the wind. Plain floats, so the host (living.mojo lw_temp's caller) reads the same sky."""
    var u = (x + t * 8) / 600
    var v = (y + t * 3) / 600
    var c = 0.55 * vnoise(u, v) + 0.3 * vnoise(u * 2.7 + 5, v * 2.7) + 0.15 * vnoise(u * 7.3 + 11, v * 7.3)
    return min(max((c - 0.5) * 5, Float32(0)), Float32(1))


@always_inline
def cloud(cam: TCAM, x: Float32, y: Float32) -> Float32:
    return cloud_at(x, y, cam[27])


@always_inline
def living_sky(cam: TCAM, pf: TPF, view: V3) -> V3:
    """Analytic atmosphere, sun disc and halo, stars and the aurora at night, the cloud layer over them (linear)."""
    var c = atmosphere(cam, view)
    if cam[30] < 0.5:
        var r = (1 - dot(view, sun(cam))) / 1.6e-4  # ~1 deg disc
        c += V3(1, 0.85, 0.6, 0) * (Float32(1.4) if r < 1 else 0.25 * exp(-(sqrt(r) - 1) * 0.7))
    elif view[2] > 0:
        var q = hash2(Int(floor(view[0] * 400)) * 7919 + Int(floor(view[1] * 400)), Int(floor(view[2] * 400)))
        if q > 0.9985:
            c += V3(0.8, 0.8, 0.9, 0) * cam[30]
    if cam[30] > 0.4 and view[2] > 0.05 and view[2] < 0.78:  # the loops hang between 0.08 and 0.68 rad up
        var az = atan2_gpu(view[1], view[0])
        var a = lw_aurora(pf.ptr.unsafe_offset(LWO), az + (Float32(6.2831853) if az < 0 else Float32(0)),
                          atan2_gpu(view[2], sqrt(view[0] * view[0] + view[1] * view[1])), cam[27])
        c += V3(a[0], a[1], a[2], 0) * (min((cam[30] - 0.4) / 0.6, Float32(1)) * 0.35)
    if view[2] > 0.02:  # 1.5 km up: the same field that shades the ground, thinning into the haze
        var t = (1500 - cam[2]) / view[2]
        var cc = cloud(cam, cam[0] + view[0] * t, cam[1] + view[1] * t) * exp(-t / 40000)
        var hz = V3(cam[23], cam[24], cam[25], 0)
        var lit = hz * hz * 0.5 + V3(1, 0.95, 0.88, 0) * (0.9 * max(sun(cam)[2], Float32(0)) * (1 - cam[30]))
        c = c * (1 - cc) + lit * ((1.2 - 0.6 * cc) * cc)  # thick cover has a darker base
    return c


@always_inline
def meadow(sm: TVIS, pf: TPF, cam: TCAM, eye: V3, ray: V3, h: SIMD[DType.float32, 8], ao: Float32) -> V3:
    """Linear colour of a Living World surface: table albedo and AO on the ground, Fresnel water, cone foliage (glow-fern
    lights at night); baked horizon shadow, the shadow map for the walker, exponential height fog to the horizon colour."""
    var T = pf.ptr.unsafe_offset(LWO)
    var hit = eye + ray * h[0]
    var u = hit / cam[16]
    var n = V3(h[1], h[2], h[3], 0)
    var view = unit(ray)
    var mat = Int(h[4])
    var sh = lw_lit(T, u[0], u[1], cam[28], cam[29])
    comptime if SHADOW == 2:
        sh = min(sh, shadow_map(sm, cam, hit, EMPTY, dot(n, sun(cam))))
    var s = sun(cam)
    var up = (1500 - hit[2]) / max(s[2], Float32(0.05))  # along the sun ray to the cloud layer
    sh *= 1 - 0.65 * cloud(cam, hit[0] + s[0] * up, hit[1] + s[1] * up)
    var col: V3
    var dm = h[0] * sqrt(dot(ray, ray))  # metres to the hit
    if mat == 2:  # water: two travelling swells plus 1.6-3.7 m ripples (faded with distance) tilt the normal; sky by Fresnel, sun glint
        var t = cam[27]
        var rp = 0.03 * exp(-dm / 80)
        var bz = n[2]  # the bed's normal, before the waves replace n
        n = unit(V3(0.05 * cos(t * 1.7 + u[0] * 2.1 + u[1] * 1.3) + rp * cos(t * 3.1 + hit[0] * 1.7 - hit[1] * 0.9) + 0.5 * rp * cos(t * 4.7 - hit[0] * 1.3 + hit[1] * 3.9),
                    0.05 * cos(t * 2.3 - u[0] * 0.9 + u[1] * 2.6) + rp * cos(t * 2.9 + hit[0] * 0.8 + hit[1] * 1.9) + 0.5 * rp * cos(t * 5.3 + hit[0] * 3.6 + hit[1] * 1.1), 1, 0))
        var r = view - n * (2 * dot(view, n))
        var fr = 0.02 + 0.98 * pow(1 - max(-dot(n, view), Float32(0)), Float32(5))
        var a = lw_albedo(T, u[0], u[1])
        col = light(cam, V3(a[0], a[1], a[2], 0) * V3(a[0], a[1], a[2], 0), n, view, sh, ao, 0) * (1 - fr) + living_sky(cam, pf, r) * fr
        var e = 200 / (1 + dm * 0.004)  # far water is rougher per pixel: a wider, dimmer glint instead of sparkle
        col += V3(1, 0.9, 0.7, 0) * (sh * pow(max(dot(r, sun(cam)), Float32(0)), e) * 3 * sqrt(e / 200) * (1 - cam[30]))
        var dep = (T[1] - u[2]) * cam[16]  # metres of water over the bed: shallows lighter and greener, deep water darker, a thin foam line that breathes at the shore
        col = col * (0.75 + 0.25 * exp(-dep / 8)) * (1 - 0.35 * exp(-dep / 2)) + V3(0.08, 0.3, 0.26, 0) * (0.35 * exp(-dep / 2) * sh)
        var hd = dep * bz / max(sqrt(max(1 - bz * bz, Float32(0))), Float32(0.02))  # metres from the waterline along the bed: a thin line on any slope, none over a flat lake bed
        var foam = min(max((0.9 + 0.4 * sin(cam[27] * 0.9 + hit[0] * 0.05 + hit[1] * 0.04) - hd) / 0.3, Float32(0)), Float32(1)) * min(max((0.3 - dep) / 0.1, Float32(0)), Float32(1)) * (0.55 + 0.45 * vnoise(hit[0] * 2, hit[1] * 2 + cam[27] * 0.6)) * (1 - min(dm / 300, Float32(1)))
        col += V3(0.9, 0.92, 0.9, 0) * (foam * (0.25 + 0.75 * sh) * (1 - 0.8 * cam[30]))
    elif mat >= 10:  # a ground prop: 10 boulder (grey, speckled), 11 bush (leaf green, dry-tipped), 12 fallen log (bark)
        var q = vnoise(hit[0] * 4, hit[1] * 4) * (0.6 + 0.4 * vnoise(hit[2] * 6, hit[0] * 3))
        var base = V3(0.42, 0.4, 0.37, 0) * (0.7 + 0.5 * q) if mat == 10 else (V3(0.17, 0.27, 0.09, 0) * (0.8 + 0.5 * q) * V3(0.92 + 0.16 * h[6], 1, 1.08 - 0.16 * h[6], 0) if mat == 11 else V3(0.26, 0.19, 0.13, 0) * (0.75 + 0.5 * q))
        col = light(cam, base * base, n, view, sh, ao, 0) + base * base * (0.08 * pow(1 - abs(dot(n, view)), Float32(2)) * (1 - cam[30]) if mat == 11 else 0)
    elif mat >= 3:  # a tree: leaf colour by species (web/world.mjs SPECIES), glow-fern lit at night (hue per tree)
        var g = h[5] * cam[30]
        var sp = mat - 3
        var leaf = V3(0.08, 0.24, 0.12, 0) if sp == 0 else (V3(0.16, 0.3, 0.08, 0) if sp == 1 else (V3(0.06, 0.2, 0.08, 0) if sp == 2 else (V3(0.3, 0.42, 0.12, 0) if sp == 3 else (V3(0.2, 0.28, 0.1, 0) if sp == 4 else V3(0.17, 0.24, 0.16, 0)))))
        var base = leaf * V3(0.92 + 0.16 * h[6], 1, 1.08 - 0.16 * h[6], 0) * (1 - g) + V3(0.3 + 0.7 * h[6], 1 - 0.5 * h[6], 0.9, 0) * g  # hue per tree
        var leaf_n = n
        if mat == 9:  # bark: birch white-grey, the rest brown, a little variation per tree
            base = (V3(0.72, 0.7, 0.64, 0) if Int(h[7]) == 3 else V3(0.3, 0.21, 0.14, 0)) * (0.85 + 0.3 * h[6])
        else:
            var q = vnoise(hit[0] * 2, hit[1] * 2)
            var qx = vnoise(hit[0] * 2 + 0.13, hit[1] * 2) - vnoise(hit[0] * 2 - 0.13, hit[1] * 2)
            var qy = vnoise(hit[0] * 2, hit[1] * 2 + 0.13) - vnoise(hit[0] * 2, hit[1] * 2 - 0.13)
            base *= 0.82 + 0.36 * q
            leaf_n = unit(n + V3(qx, qy, 0, 0) * 0.2)
        col = light(cam, base * base, leaf_n, view, sh, ao, 0.1) + base * base * (1.6 * g)
        if mat != 9:  # backlight through leaf edges and crowns
            var rim = pow(1 - abs(dot(leaf_n, view)), Float32(2))
            var back = max(dot(-leaf_n, sun(cam)), Float32(0))
            col += base * base * ((0.2 * back + 0.08 * rim) * (1 - cam[30]))
    else:
        var a = lw_albedo(T, u[0], u[1])
        # ground per habitat: noise scale (1/m) and contrast. 0 scree stones, 1 riparian, 2 meadow grass streaks,
        # 3 spruce moss, 4 heather clumps, 5 beech leaf litter
        var hb = Int(cell(T, Int(floor(u[0] + 0.5)), Int(floor(u[1] + 0.5)), 4))
        var fq = Float32(1.25) if hb == 0 else (Float32(2) if hb == 1 else (Float32(3.3) if hb == 2 else (Float32(1.5) if hb == 3 else (Float32(2) if hb == 4 else Float32(1)))))
        var amp = Float32(0.4) if hb == 0 else (Float32(0.3) if hb == 1 else (Float32(0.35) if hb == 2 else (Float32(0.15) if hb == 3 else (Float32(0.5) if hb == 4 else Float32(0.4)))))
        var nx = hit[0] * fq * (Float32(0.35) if hb == 2 else Float32(1))  # grass: streaks along y
        var grain = 1 - amp * 0.5 + amp * vnoise(nx, hit[1] * fq)
        var far = min(max((dm - 100) / 100, Float32(0)), Float32(1))  # past 100-200 m
        var canopy = far * min(max((0.32 - a[1]) * 6, Float32(0)), Float32(1)) * Float32(1 - get_defined_int["LW_NOCANOPY", 0]())  # dark green: a stand's crowns
        grain = grain * (1 - canopy) + canopy * (0.55 + 0.9 * vnoise(hit[0] * 0.17, hit[1] * 0.17) * vnoise(hit[0] * 0.5 + 7, hit[1] * 0.5))  # 6 m crowns, gaps
        var base = V3(a[0], a[1], a[2], 0) * grain
        # 50-300 m patches: dry yellow-green to lush green, drier on slopes facing -y, bare rock where steep
        var v = 0.6 * vnoise(hit[0] / 90, hit[1] / 90) + 0.4 * vnoise(hit[0] / 310 + 17, hit[1] / 310)
        var dry = min(max((v - 0.5) * 2.5 - 2 * n[1] + 0.5, Float32(0)), Float32(1))
        var rock = min(max((0.8 - n[2]) * 4, Float32(0)), Float32(1)) * 0.6
        var tint = V3(1, 1, 1, 0) if hb == 0 else V3(0.88, 1.06, 0.9, 0) * (1 - dry) + V3(1.18, 1.04, 0.72, 0) * dry  # scree stays grey
        base = base * tint * (1 - rock) + V3(0.4, 0.39, 0.36, 0) * (grain * rock)
        var wetg = min(max(1 - (u[2] - T[1]) * cam[16] / 3, Float32(0)), Float32(1))  # 0-3 m over the sea: wet sand, darker and a touch cooler, wettest at the line
        base = base * (1 - 0.4 * wetg * wetg) * V3(1 - 0.08 * wetg, 1 - 0.02 * wetg, 1 + 0.04 * wetg, 0)
        var tr = lw_trail(T, u[0], u[1]) * 0.85 * min(max((dm - 40) / 60, Float32(0)), Float32(1))  # trails: the 20 m cell tint from 40 m out, packed earth ribbon (1.2 m, soft edge) within 100 m
        var rb = Float32(0)
        if dm < 100:
            rb = min(max((0.85 - trail_d(T, u[0], u[1]) * cam[16]) / 0.4, Float32(0)), Float32(1))
            tr = max(tr, rb * rb * (3 - 2 * rb) * 0.95 * min(max((100 - dm) / 40, Float32(0)), Float32(1)))
        base = base * (1 - tr) + V3(0.80, 0.70, 0.52, 0) * (grain * tr)  # pale dry earth: the old (0.5, 0.42, 0.3) was the heath's brown, and a trail read as heath from the air
        var gw = lw_grass(T, u[0], u[1]) if dm < 30 else V3(0, 0, 0, 0)  # meadow and heather share here, bilinear over the nodes: the grass fades across habitat edges
        if gw[0] + gw[1] > 0.01 and n[2] > 0.5:  # blades: 7 shells over the ground within 30 m, thinned out by 18-30 m, each shell on its own turned, offset hash grid (about 6 cm)
            var cover = min((30 - dm) / 12, Float32(1)) * (1 - rb) * (1 - wetg) * min((gw[0] + gw[1]) * 1.3, Float32(1))
            var back = max(-dot(view, n), Float32(0.15))
            var pw = max(dm * 0.005, Float32(0.008))  # metres one shaded pixel covers (half-res quad): a blade's edge is a ramp of that width, so a pixel that sees half a blade is half lit and does not flicker
            var fc = min(max(1 - (dm - 4) / 14, Float32(0.12)), Float32(0.75))  # contrast of the blades: full within 6 m, down to 12 percent by 20 m, where a pixel holds many blades (less flicker under the temporal AA)
            var acc = V3(0, 0, 0, 0)  # shells composited from the top down
            var trans = Float32(1)
            for k in range(7):
                var ca = cos(Float32(k) * 2.4 + 0.3)  # a different turn of the grid for every shell: no shared axes
                var sa = sin(Float32(k) * 2.4 + 0.3)
                var hg = Float32(0.38)
                var hh = hg * Float32(7 - k) / 7
                var sk = min(hh / back, Float32(2))
                var qx = hit[0] - view[0] * sk
                var qy = hit[1] - view[1] * sk
                var rx = ca * qx - sa * qy + Float32(k) * 0.0173
                var ry = sa * qx + ca * qy + Float32(k) * 0.0291
                var ix = Int(floor(rx * 16.667))
                var iy = Int(floor(ry * 16.667))
                var hs = (ix * 73856093) ^ (iy * 19349663) ^ (k * 83492791)
                var isheath = rnd(hs, 5) * (gw[0] + gw[1]) < gw[1]  # this blade is heather with the heather's share
                hg = Float32(0.26) if isheath else Float32(0.38)
                var hb2 = hg * (0.3 + 0.7 * rnd(hs, 0))  # this blade's height
                if hb2 > hh and rnd(hs, 1) < cover:
                    var ld = 6.2831853 * rnd(hs, 7)  # a lean of its own, then the wind's
                    var lm = (0.04 + 0.07 * rnd(hs, 8)) * (hh / hb2) * (hh / hb2)
                    var wnd = 0.05 * (hh / hb2) * (hh / hb2) * sin(cam[27] * 1.4 + qx * 0.35 + qy * 0.2) * Float32(1 - get_defined_int["LW_NOWIND", 0]())
                    var lx = lm * cos(ld) + wnd
                    var ly = lm * sin(ld) + 0.5 * wnd
                    var dx = rx - (Float32(ix) + 0.15 + 0.7 * rnd(hs, 2)) * 0.06 - (ca * lx - sa * ly)
                    var dy = ry - (Float32(iy) + 0.15 + 0.7 * rnd(hs, 3)) * 0.06 - (sa * lx + ca * ly)
                    var w = (0.035 * (1 - 0.6 * hh / hb2) + 0.004) * (0.7 + 0.6 * rnd(hs, 6))
                    var al = min(max((w - sqrt(dx * dx + dy * dy)) / pw + 0.5, Float32(0)), Float32(1))
                    if al > 0:
                        var tb = hh / hb2
                        var hue = rnd(hs, 4)
                        var hv = Float32(1) if isheath else Float32(0)
                        acc += base * ((0.95 + (0.9 * tb - 0.45) * fc) * trans * al) * (V3(1.0 + (0.2 * hue - 0.1) * fc, 1.2 + (0.15 * hue - 0.07) * fc, 0.7, 0) * (1 - hv) + V3(1.0 + (0.15 * tb - 0.07) * fc, 0.92 + (0.05 * tb - 0.02) * fc, 1.0 + (0.2 * tb - 0.1) * fc, 0) * hv)  # a blade: dark at the root, lit at the tip, green on the meadow, purple-tipped on the heather
                        trans *= 1 - al
            base = acc + base * (trans * (1 - 0.3 * cover))  # what no blade covers is the ground, in their shade
        col = light(cam, base * base, n, view, sh, ao * a[3], Float32(0.05) if hb == 0 else 0)  # wet-looking scree catches the sun
        if hb == 2:  # meadow grass: back-lit sheen when looking toward the sun
            col += base * base * (0.35 * pow(1 - abs(dot(n, view)), Float32(4)) * max(dot(view, s), Float32(0)) * sh * (1 - cam[30]))
    var d = dm / cam[16]  # world units, as the web's fog
    var k = 0.006 * (0.6 + 0.5 * exp(-(u[2] - pf[LWO + 1]) / 3))  # per 20 m cell: haze visibility about 10 km, a little thicker in valleys
    var far = pf[LWO] * Float32(LIVING_FAR) / 128
    var fog = max(1 - exp(-max(d - 2.5, Float32(0)) * k), min(max((d - far * 0.7) / (far * 0.3), Float32(0)), Float32(1)))
    fog *= Float32(1 - get_defined_int["LW_NOFOG", 0]())  # debug: LW_NOFOG=1 shows the bare surface
    return (col * (1 - fog) + atmosphere(cam, unit(ray)) * fog) * cam[26]


@always_inline
def ground_hit(cam: TCAM, pf: TPF, eye: V3, ray: V3) -> Tuple[Float32, Int]:
    """Ray parameter of the floor, terrain or box wall (3.4e38: none) and the box face (BOX: 0, 1 x walls, 2, 3 y, 4 floor, 5 ceiling)."""
    var tmin = Float32(3.4e38)
    var face = -1
    comptime if BOX > 0:  # exit point of the cube [-B/2, B/2]^2 x [0, B] along the ray
        for a in range(3):
            if ray[a] != 0:
                var lo = Float32(0) if a == 2 else -BOX * 0.5
                var t = ((lo + BOX if ray[a] > 0 else lo) - eye[a]) / ray[a]
                if t > 0 and t < tmin:
                    tmin = t
                    face = 2 * a + (1 if ray[a] > 0 else 0)
    elif TERRAIN:
        tmin = ground(cam, eye, ray)
    elif CITY:
        tmin = city(cam, eye, ray)[0]
    elif LIVING:
        tmin = living(cam, pf, eye, ray)[0]
    else:
        if ray[2] < 0:
            tmin = -eye[2] / ray[2]
    return (tmin, face)


@always_inline
def terrain_shadow(cam: TCAM, x0: V3, p: PW) -> Float32:
    """Soft terrain self-shadow: 16 steps up the sun ray, penumbra from the closest clearance over distance."""
    if (Int(p[4]) == 0 or p[2] == 0) and p[8] == 0:  # a plane never shades itself
        return 1
    var res = Float32(1)
    var t = Float32(0.05)
    for _ in range(16):
        var x = x0 + sun(cam) * t
        var d = x[2] - height_grad(x[0], x[1], p)[0]
        if d < 0:
            return 0
        res = min(res, 6 * d / t)
        t = t * 1.45 + 0.05
    return res


@always_inline
def shadow_map(sm: TVIS, cam: TCAM, x0: V3, id: UInt64, ndl: Float32) -> Float32:
    """Lit fraction of x0 from the shadow map, 3x3 PCF; a texel of x0's own triangle never shades it."""
    var s = sun_view(cam)
    var d = x0 - s[0]
    var z = dot(d, s[3])
    var x = Float32(W) * 0.5 + s[4] * dot(d, s[1]) / z
    var y = Float32(H) * 0.5 - s[4] * dot(d, s[2]) / z
    var bias = z / s[4] * (1.5 + 2 * sqrt(max(1 - ndl * ndl, Float32(0))) / max(ndl, Float32(0.2)))  # texels, slope-scaled
    var lit = Float32(0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var ix = Int(x) + dx
            var iy = Int(y) + dy
            if ix < 0 or iy < 0 or ix >= W or iy >= H or x < 0 or y < 0:
                lit += 1
                continue
            var k = sm[iy * W + ix]
            if k == EMPTY or (k & 0xFFFFFFFF) == id or depth(k) >= z - bias:
                lit += 1
    return lit / 9


@always_inline
def blob(pose: TPOSE, cam: TCAM, x0: V3) -> Float32:
    """SHADOW=1: each animal darkens the ground around its root's shadow point, Gaussian of radius BLOB."""
    var s = Float32(1)
    var spacing = cam[10]
    for f in range(F):
        var r = V3(pose[f], pose[F + f], pose[2 * F + f], 0)
        if spacing > 0:
            r = V3(Float32(f % GX) * spacing, Float32(f // GX) * spacing, r[2], 0)
        var c = r - sun(cam) * ((r[2] - x0[2]) / sun(cam)[2])
        var d2 = ((x0[0] - c[0]) * (x0[0] - c[0]) + (x0[1] - c[1]) * (x0[1] - c[1])) / (BLOB * BLOB)
        if d2 < 9:
            s *= 1 - 0.6 * exp(-d2)
    return s


@always_inline
def occlusion(vis: TVIS, i: Int, z0: Float32, focal: Float32) -> Float32:
    """Screen-space ambient occlusion: 8 neighbours 0.12 mm away; a neighbour 0.004..0.5 mm nearer occludes."""
    var r = min(max(focal * 0.12 / z0, Float32(2)), Float32(16))
    var x = i % W
    var y = i // W
    var occ = Float32(0)
    for k in range(8):
        var ang = Float32(k) * 0.7853982 + 0.39
        var j = Int(Float32(x) + r * cos(ang))
        var l = Int(Float32(y) + r * sin(ang))
        if j < 0 or l < 0 or j >= W or l >= H:
            continue
        var kk = vis[l * W + j]
        if kk != EMPTY:
            var dz = z0 - depth(kk)
            if dz > 0.004 and dz < 0.5:
                occ += 1 - dz * 2
    return 1 - 0.6 * occ / 8


@always_inline
def light(cam: TCAM, albedo: V3, n: V3, view: V3, sh: Float32, ao: Float32, spec: Float32) -> V3:
    """Linear colour: sky/ground hemisphere ambient x ao, sun x shadow, Blinn highlight of strength spec."""
    var ndl = max(dot(n, sun(cam)), Float32(0))
    var amb = (V3(0.36, 0.40, 0.48, 0) * (0.5 + 0.5 * n[2]) + V3(0.20, 0.17, 0.13, 0) * (0.5 - 0.5 * n[2])) * ao
    var lc = V3(0.80, 0.76, 0.70, 0)
    comptime if LIVING:  # the web's palette: ambient from the sky by elevation, warm sun by day, blue moon by night
        var day = 1 - cam[30]
        lc = V3(1, 0.95 - 0.25 * cam[31], 0.85 - 0.4 * cam[31], 0) * day + V3(0.05, 0.07, 0.16, 0) * cam[30]
        var bounce = sqrt(albedo) * lc * (0.7 * max(sun(cam)[2], Float32(0)))  # sunlit ground of this surface's colour lights it from below
        amb = (V3(cam[20], cam[21], cam[22], 0) * (0.5 + 0.5 * n[2]) + bounce * (0.5 - 0.5 * n[2])) * ((0.25 + 0.75 * day) * ao)
    var col = albedo * (amb + lc * (ndl * sh))
    if spec > 0 and ndl > 0:
        var h = max(dot(n, unit(sun(cam) - view)), Float32(0))
        var h2 = h * h
        var h8 = (h2 * h2) * (h2 * h2)
        var h16 = h8 * h8
        col += V3(1, 1, 1, 0) * (spec * h16 * h16 * h16 * sh)
    return col


@always_inline
def sky(view: V3) -> V3:
    """Linear sky: horizon (the old flat sky colour) to zenith."""
    var s = sqrt(max(view[2], Float32(0)))
    var c = V3(0.6, 0.65, 0.72, 0) * (1 - s) + V3(0.36, 0.52, 0.80, 0) * s
    return c * c


@always_inline
def agx(x: Float32) -> Float32:
    """AgX fitted sigmoid on exposure-scaled linear light."""
    var y = min(max((log(max(x, Float32(1e-10))) / 0.6931472 + 10) / 16, Float32(0)), Float32(1))
    var y2 = y * y
    var y3 = y2 * y
    var y4 = y3 * y
    var y5 = y4 * y
    var y6 = y5 * y
    return min(max(15.5 * y6 - 40.14 * y5 + 31.96 * y4 - 6.868 * y3 + 0.4298 * y2 + 0.1191 * y - 0.00232, Float32(0)), Float32(1))


@always_inline
def animal(sm: TVIS, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM, eye: V3, ray: V3,
           key: UInt64, tt: Float32, ao: Float32, mirror: Bool) -> V3:
    """Linear colour of animal triangle `key` along ray (at ray parameter tt): smooth normal and vertex colour, sun with its
    shadow, sky/ground ambient x ao, highlight. mirror: the triangle reflected in z = 0, lit as the real one, no shadow."""
    var f = Int(key >> TB64) & ((1 << (32 - TB)) - 1)
    var t = Int(key & TMASK)
    var p = Int(rp[t])
    var base = 3 * t
    var m = V3(1, 1, -1, 0) if mirror else V3(1, 1, 1, 0)
    var a = vertex(pi, pf, rv, rt, pose, cam, f, p, base) * m
    var b = vertex(pi, pf, rv, rt, pose, cam, f, p, base + 1) * m
    var c = vertex(pi, pf, rv, rt, pose, cam, f, p, base + 2) * m
    var h = tri_hit(eye, ray, a, b, c)
    var n = unit(cross(b - a, c - a)) * (Float32(-1) if mirror else Float32(1))  # a reflection flips the winding
    if h[0] < 3.4e38:  # smooth normal, unless it disagrees with the face (two-sided thin parts: wing normals cancel)
        var ns = unit(vnormal(pi, pf, rv, rt, pose, f, p, base) * (1 - h[1] - h[2]) + vnormal(pi, pf, rv, rt, pose, f, p, base + 1) * h[1]
                      + vnormal(pi, pf, rv, rt, pose, f, p, base + 2) * h[2]) * m
        if dot(ns, n) > 0.5:
            n = ns
    if dot(n, ray) > 0:
        n = -n
    var sh = Float32(1)
    if not mirror:
        var x0 = eye + ray * tt
        comptime if SHADOW == 2:
            sh = shadow_map(sm, cam, x0, key & 0xFFFFFFFF, dot(n, sun(cam)))
        comptime if TERRAIN:
            sh = min(sh, terrain_shadow(cam, x0, terrain(cam)))
        comptime if LIVING:
            sh = min(sh, lw_lit(pf.ptr.unsafe_offset(LWO), x0[0] / cam[16], x0[1] / cam[16], cam[28], cam[29]))
    var rgb = V3(pf[PFS * p + 17], pf[PFS * p + 18], pf[PFS * p + 19], 0)
    if h[0] < 3.4e38:  # the vertex colours (texture baked per vertex), interpolated
        rgb = (vcolour(pi, rv, rt, p, base) * (1 - h[1] - h[2]) + vcolour(pi, rv, rt, p, base + 1) * h[1]
               + vcolour(pi, rv, rt, p, base + 2) * h[2])
    return light(cam, rgb * rgb, n * m, unit(ray) * m, sh, ao, 0.35)  # no fog on animals: at world-camera distances it greys them out


@always_inline
def key_hit(pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM, eye: V3, ray: V3, key: UInt64) -> V3:
    """tri_hit of the triangle a vis key names (an animal's or a gull's) against ray."""
    if isbird(key):
        var k = Int(key & TMASK)
        var o = BIRD0 + 2 * BGP + 3 * (k % NTB)
        return tri_hit(eye, ray, bird_vertex(pf, cam, k // NTB, Int(pf[o])), bird_vertex(pf, cam, k // NTB, Int(pf[o + 1])), bird_vertex(pf, cam, k // NTB, Int(pf[o + 2])))
    var f = Int(key >> TB64) & ((1 << (32 - TB)) - 1)
    var t = Int(key & TMASK)
    var p = Int(rp[t])
    return tri_hit(eye, ray, vertex(pi, pf, rv, rt, pose, cam, f, p, 3 * t), vertex(pi, pf, rv, rt, pose, cam, f, p, 3 * t + 1),
                   vertex(pi, pf, rv, rt, pose, cam, f, p, 3 * t + 2))


@always_inline
def sample(sm: TVIS, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM, eye: V3, ray: V3,
           key: UInt64, tt: Float32, ao: Float32, mk: UInt64, h0: SIMD[DType.float32, 8] = SIMD[DType.float32, 8](-1)) -> V3:
    """Linear colour along ray: the animal triangle key at ray parameter tt when it is nearer than the ground,
    else the ground or box (the flat floor mixes in the mirror layer's key mk at REFL), else the sky. Fog: ground only.
    LIVING: h0 is this ray's living() hit when the caller has marched it already (h0[0] < 0: march here)."""
    comptime if LIVING:
        var h = h0 if h0[0] >= 0 else living(cam, pf, eye, ray)
        if key != EMPTY and tt < h[0]:
            if isbird(key):
                return birdcol(cam) * cam[26]
            return animal(sm, pi, pf, rv, rt, rp, pose, cam, eye, ray, key, tt, ao, False) * cam[26]
        return meadow(sm, pf, cam, eye, ray, h, ao) if h[0] < 3.4e38 else living_sky(cam, pf, unit(ray)) * cam[26]
    comptime if CITY:
        var h = city(cam, eye, ray)
        if key != EMPTY and tt < h[0]:
            return animal(sm, pi, pf, rv, rt, rp, pose, cam, eye, ray, key, tt, ao, False)
        return street(sm, cam, eye, ray, h, ao) if h[0] < 3.4e38 else sky(unit(ray))
    var g = ground_hit(cam, pf, eye, ray)
    var view = unit(ray)
    var fogc = V3(0.36, 0.4225, 0.5184, 0)  # horizon colour, linear
    if key != EMPTY and tt < g[0]:
        return animal(sm, pi, pf, rv, rt, rp, pose, cam, eye, ray, key, tt, ao, False)
    if BOX > 0 and g[1] >= 0:
        var hit = eye + ray * g[0]
        var face = g[1]
        var u = hit[1] if face < 2 else hit[0]
        var v = hit[1] if face >= 4 else hit[2]
        var check = (Int(u + 1e4) + Int(v + 1e4)) % 2
        var tint = V3(0.55, 0.57, 0.6, 0) if face == 4 else (V3(0.62, 0.6, 0.55, 0) if face == 5 else V3(0.5, 0.56, 0.62, 0))
        tint = tint * (Float32(0.8) if check == 0 else Float32(1.0))
        return tint * tint
    if g[0] < 3.4e38:
        var hit = eye + ray * g[0]
        var n = V3(0, 0, 1, 0)
        comptime if TERRAIN:
            var hg = height_grad(hit[0], hit[1], terrain(cam))
            n = V3(-hg[1], -hg[2], 1, 0) / sqrt(hg[1] * hg[1] + hg[2] * hg[2] + 1)
        var check = (Int(hit[0] + 1e4) + Int(hit[1] + 1e4)) % 2
        var base = soil(hit[0], hit[1]) if SOIL else V3(0.55, 0.57, 0.6, 0) * (Float32(0.8) if check == 0 else Float32(1.0))
        var sh = Float32(1)
        comptime if SHADOW == 2:
            sh = shadow_map(sm, cam, hit, EMPTY, dot(n, sun(cam)))
        comptime if SHADOW == 1:
            sh = blob(pose, cam, hit)
        comptime if TERRAIN:
            sh = min(sh, terrain_shadow(cam, hit, terrain(cam)))
        var lit = light(cam, base * base, n, view, sh, ao, 0)
        comptime if REFLECT:
            if mk != EMPTY:
                lit = lit * (1 - REFL) + animal(sm, pi, pf, rv, rt, rp, pose, cam, eye, ray, mk, 0, 1, True) * REFL
        var fade = Float32(1) / (1 + 0.02 * sqrt(dot(hit - eye, hit - eye)))
        return lit * fade + fogc * (1 - fade)
    return sky(view)


comptime SHT = ((HW + 15) // 16) * ((HH + 15) // 16) * 256  # LIVING shade: threads for the quads' marching pixels, whole 16 x 16 tiles


def shade_px(i: Int, vis: TVIS, sm: TVIS, img: TIMG, low: TLOW, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM):
    var c = camera(cam)
    var px = Float32(i % W) + 0.5 - Float32(W) * 0.5
    var py = Float32(i // W) + 0.5 - Float32(H) * 0.5
    var ray = c[3] * c[4] + c[1] * px - c[2] * py
    var key = vis[i]
    var gk = sm.ptr.unsafe_offset(OGL + i)[] if GLASS else EMPTY
    var ph = phase(sm)
    comptime if LIVING:  # half-res: background pixels off this frame's quad pixel come from upscale, so they skip the march
        if key == EMPTY and gk == EMPTY and ((i % W) & 1 != ph[1] or (i // W) & 1 != ph[2]):
            img[i] = 0
            return
    var h = SIMD[DType.float32, 8](-1)  # LIVING: this pixel's one march, shared by every sample() below
    comptime if LIVING:
        h = living(cam, pf, c[0], ray)
    var g = (h[0], -1) if LIVING else ground_hit(cam, pf, c[0], ray)
    comptime if CITY_DEPTH:
        img[i] = bitcast[DType.uint32](g[0] * sqrt(dot(ray, ray)) if g[0] < 3.4e38 else Float32(1e30))
        return
    var front = key != EMPTY and depth(key) < g[0] * c[4]  # raster depth is view z = ray parameter x focal
    var z0 = depth(key) if front else g[0] * c[4]
    var mk = sm.ptr.unsafe_offset(OMR + i)[] if REFLECT else EMPTY
    var glass = gk != EMPTY and depth(gk) < z0
    var fly = (front and (isbird(key) or pi[5 * Int(rp[Int(key & TMASK)])] > 0)) or glass  # alpha: animal parts only (not world parts)
    var ao = occlusion(vis, i, z0, c[4]) if z0 < 3.4e38 else Float32(1)
    var col = sample(sm, pi, pf, rv, rt, rp, pose, cam, c[0], ray, key, depth(key) / c[4], ao, mk, h)
    comptime if AA:
        var x = i % W
        var y = i // W
        var edge = False
        for k in range(4):
            var j = x + (1 if k == 0 else (-1 if k == 1 else 0))
            var l = y + (1 if k == 2 else (-1 if k == 3 else 0))
            if j >= 0 and l >= 0 and j < W and l < H:
                var kk = vis[l * W + j]
                if (kk == EMPTY) != (key == EMPTY) or (kk != EMPTY and key != EMPTY and abs(depth(kk) - depth(key)) > 0.05 * z0):
                    edge = True
        if edge:  # 4 rotated-grid sub-rays, each against the triangles of the 3x3 neighbourhood, then the ground
            col = V3(0)
            for s in range(4):
                var ox = Float32(0.125) if s == 0 else (Float32(0.375) if s == 1 else (Float32(-0.125) if s == 2 else Float32(-0.375)))
                var oy = Float32(0.375) if s == 0 else (Float32(-0.125) if s == 1 else (Float32(-0.375) if s == 2 else Float32(0.125)))
                var rs = c[3] * c[4] + c[1] * (px + ox) - c[2] * (py + oy)
                var best = Float32(3.4e38)
                var bkey = EMPTY
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        var j = x + dx
                        var l = y + dy
                        if j < 0 or l < 0 or j >= W or l >= H:
                            continue
                        var kk = vis[l * W + j]
                        if kk == EMPTY or kk == bkey:
                            continue
                        var h = key_hit(pi, pf, rv, rt, rp, pose, cam, c[0], rs, kk)
                        if h[0] < best:
                            best = h[0]
                            bkey = kk
                col += sample(sm, pi, pf, rv, rt, rp, pose, cam, c[0], rs, bkey, best, ao, mk) * 0.25
    if glass:  # a see-through part in front: its colour over the scene at its alpha
        var a = pf[PFS * Int(rp[Int(gk & TMASK)]) + 20]
        col = col * (1 - a) + animal(sm, pi, pf, rv, rt, rp, pose, cam, c[0], ray, gk, depth(gk) / c[4], 1, False) * a
    comptime if LIVING:
        col = V3(agx(col[0]), agx(col[1]), agx(col[2]), 0)
    var r = UInt32(min(sqrt(max(col[0], 0)), 1) * 255)
    var gg = UInt32(min(sqrt(max(col[1], 0)), 1) * 255)
    var b = UInt32(min(sqrt(max(col[2], 0)), 1) * 255)
    var pixel = r | gg << 8 | b << 16 | (UInt32(255) if fly else UInt32(0)) << 24
    img[i] = pixel
    comptime if LIVING:
        if ((i % W) & 1) == ph[1] and ((i // W) & 1) == ph[2]:
            comptime if TAA:  # where this quad's background was one frame ago, for upscale's history fetch
                var t = sm.ptr.unsafe_offset(OTL).bitcast[Float32]()
                var mv = UInt32(0xFFFFFFFF)
                if t[2 * NCAM + 1] > 0:
                    var pc = basis(V3(t[0], t[1], t[2], 0), V3(t[3], t[4], t[5], 0), t[7])
                    var dv = ray if h[0] >= 3.4e38 else c[0] + ray * h[0] - pc[0]  # sky: the direction only
                    var z = dot(dv, pc[3])
                    var sx = Float32(W) * 0.5 - 0.5 + pc[4] * dot(dv, pc[1]) / z
                    var sy = Float32(H) * 0.5 - 0.5 - pc[4] * dot(dv, pc[2]) / z
                    if z > 0 and sx >= 0 and sy >= 0 and sx < Float32(W - 1) and sy < Float32(H - 1):
                        mv = UInt32(sx * 8) | UInt32(sy * 8) << 16
                sm.ptr.unsafe_offset(OMV).bitcast[UInt32]()[(i // W // 2) * HW + (i % W // 2)] = mv
            if key == EMPTY and not glass:  # alpha 255 marks open sky for glow's shaft march
                low[(i // W // 2) * HW + (i % W // 2)] = (pixel & 0xFFFFFF) | (UInt32(255) << 24 if g[0] >= 3.4e38 else UInt32(0))
            else:
                var bg = sample(sm, pi, pf, rv, rt, rp, pose, cam, c[0], ray, EMPTY, Float32(3.4e38), 1, EMPTY, h)
                low[(i // W // 2) * HW + (i % W // 2)] = UInt32(min(sqrt(max(agx(bg[0]), 0)), 1) * 255) | UInt32(min(sqrt(max(agx(bg[1]), 0)), 1) * 255) << 8 | UInt32(min(sqrt(max(agx(bg[2]), 0)), 1) * 255) << 16


def shade(vis: TVIS, sm: TVIS, img: TIMG, low: TLOW, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM):
    """LIVING: the first HW * HH threads are the quads' marching pixels, in 8 x 4 tiles per wave so neighbouring rays march alike
    (a thread per pixel left every other lane of a marching row, and the whole odd rows, idle); the rest are one per pixel for
    everything else, and background there returns at once. Otherwise one thread per pixel."""
    var t = Int(block_idx.x * block_dim.x + thread_idx.x)
    comptime if LIVING:
        if t < SHT:
            var ph = phase(sm)
            var tile = t // 256  # 16 x 16 quads, a wave of 32 threads 8 x 4 of them
            var l = t % 256
            var nx = (HW + 15) // 16
            var qx = tile % nx * 16 + (l // 32 % 2) * 8 + l % 8
            var qy = tile // nx * 16 + (l // 64) * 4 + l % 32 // 8
            if qx < HW and qy < HH and qx * 2 + ph[1] < W and qy * 2 + ph[2] < H:
                shade_px((qy * 2 + ph[2]) * W + qx * 2 + ph[1], vis, sm, img, low, pi, pf, rv, rt, rp, pose, cam)
            return
        t -= SHT
        if t < W * H and ((t % W) & 1 != phase(sm)[1] or (t // W) & 1 != phase(sm)[2]):
            shade_px(t, vis, sm, img, low, pi, pf, rv, rt, rp, pose, cam)
    else:
        if t < W * H:
            shade_px(t, vis, sm, img, low, pi, pf, rv, rt, rp, pose, cam)


@always_inline
def glow(img: TIMG, low: TLOW, cam: TCAM, i: Int, x: Int, y: Int):
    """Bloom + god rays (LIVING, day): 16 taps from the pixel toward the sun's screen point over low's open-sky
    mask; the open fraction times a radial falloff, added in display space. A blocked sun loses its halo."""
    if cam[30] >= 0.5:
        return
    var c = camera(cam)
    var s = sun(cam)
    var fz = dot(s, c[3])
    if fz <= 0.05:
        return
    var sx = c[4] * dot(s, c[1]) / fz + Float32(W) * 0.5
    var sy = -c[4] * dot(s, c[2]) / fz + Float32(H) * 0.5
    var dx = sx - Float32(x)
    var dy = sy - Float32(y)
    var r = sqrt(dx * dx + dy * dy) / Float32(H)
    var fall = 0.45 * exp(-r * 3.5) + 0.35 * exp(-r * 18)
    if fall < 0.01:
        return
    var open = Float32(0)
    for k in range(16):
        var t = (Float32(k) + 0.5) / 16
        var lx = min(max(Int((Float32(x) + dx * t) * 0.5), 0), HW - 1)
        var ly = min(max(Int((Float32(y) + dy * t) * 0.5), 0), HH - 1)
        open += Float32(low[ly * HW + lx] >> 24) / 255
    var a = fall * open / 16 * min(fz * 4, Float32(1)) * (1 - 2 * cam[30])
    var p = img[i]
    var out = p & 0xFF000000
    var tint = V3(1, 0.85, 0.6, 0)
    for k in range(3):
        var shift = UInt32(8 * k)
        out |= UInt32(min(Float32((p >> shift) & 255) + a * tint[k] * 255, Float32(255))) << shift
    img[i] = out


@always_inline
def post(p: UInt32, cam: TCAM, x: Int, y: Int) -> UInt32:
    """Display-space finish (LIVING): mild S-curve, split tone (cool shadows, warm highlights), +10% saturation, a soft
    vignette, monochrome film grain that changes every frame (triangular, +-2.5/255: it also dithers away banding)."""
    var v = V3(Float32(p & 255), Float32((p >> 8) & 255), Float32((p >> 16) & 255), 0) / 255
    v = v * 0.75 + v * v * (3 - 2 * v) * 0.25
    v += (1 - v) * (1 - v) * V3(-0.012, 0, 0.018, 0) + v * v * V3(0.02, 0.005, -0.025, 0)
    var l = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    var dx = (Float32(x) - Float32(W) * 0.5) / Float32(W)
    var dy = (Float32(y) - Float32(H) * 0.5) / Float32(W)
    var fr = Int(cam[27] * 60)
    var g = (hash2(x + fr * 7919, y) + hash2(x, y + fr * 104729) - 1) * Float32(2.5 / 255)
    v = (V3(l, l, l, 0) + (v - l) * 1.1) * (1 - 0.45 * (dx * dx + dy * dy)) + g
    var out = p & 0xFF000000
    for k in range(3):
        out |= UInt32(min(max(v[k] * 255 + 0.5, Float32(0)), Float32(255))) << UInt32(8 * k)
    return out


@always_inline
def resolve(sm: TVIS, a: UInt32, b: UInt32, c: UInt32, d: UInt32, cur: UInt32, x: Int, y: Int, ph: Tuple[Int, Int, Int]) -> UInt32:
    """TAA: last frame's colour where this pixel was (bilinear in the history), clamped to the 2x2 low quad around it,
    blended in: this frame weighs 1/4 on the freshly shaded quad pixel, 1/10 on the three interpolated ones. No history
    after a cut or off screen."""
    var m = sm.ptr.unsafe_offset(OMV).bitcast[UInt32]()[(y // 2) * HW + x // 2]
    if sm.ptr.unsafe_offset(OTL).bitcast[Float32]()[2 * NCAM + 1] == 0 or m == 0xFFFFFFFF:
        return cur
    var hx = Float32(m & 0xFFFF) / 8 + Float32(x - (x & ~1) - ph[1])  # the quad pixel's spot plus this pixel's offset from it
    var hy = Float32(m >> 16) / 8 + Float32(y - (y & ~1) - ph[2])
    if hx < 0 or hy < 0 or hx > Float32(W - 1) or hy > Float32(H - 1):
        return cur
    var hp = sm.ptr.unsafe_offset(OTA).bitcast[UInt32]()
    var par = 1 - (ph[0] & 1)
    var ix = min(Int(hx), W - 2)
    var iy = min(Int(hy), H - 2)
    var tx = hx - Float32(ix)
    var ty = hy - Float32(iy)
    var w = Float32(0.25) if (x & 1) == ph[1] and (y & 1) == ph[2] else Float32(0.1)
    var out = UInt32(0)
    for k in range(3):
        var sh = UInt32(8 * k)
        var h0 = Float32((hp[2 * (iy * W + ix) + par] >> sh) & 255) * (1 - tx) + Float32((hp[2 * (iy * W + ix + 1) + par] >> sh) & 255) * tx
        var h1 = Float32((hp[2 * (iy * W + W + ix) + par] >> sh) & 255) * (1 - tx) + Float32((hp[2 * (iy * W + W + ix + 1) + par] >> sh) & 255) * tx
        var lo = Float32(min(min((a >> sh) & 255, (b >> sh) & 255), min((c >> sh) & 255, (d >> sh) & 255)))
        var hi = Float32(max(max((a >> sh) & 255, (b >> sh) & 255), max((c >> sh) & 255, (d >> sh) & 255)))
        var hv = min(max(h0 * (1 - ty) + h1 * ty, lo), hi)
        out |= UInt32(hv + (Float32((cur >> sh) & 255) - hv) * w + 0.5) << sh
    return out


def upscale(vis: TVIS, sm: TVIS, img: TIMG, low: TLOW, cam: TCAM):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= W * H:
        return
    comptime if LIVING:
        var x = i % W
        var y = i // W
        var bgpx = vis[i] == EMPTY
        comptime if GLASS:
            if sm.ptr.unsafe_offset(OGL + i)[] != EMPTY:
                bgpx = False
        var ph = phase(sm)
        if bgpx:  # bilinear over this frame's quad pixels (low), then (TAA) the history blend
            var fx = Float32(x - ph[1]) * 0.5
            var fy = Float32(y - ph[2]) * 0.5
            var x0 = min(max(Int(floor(fx)), 0), HW - 1)
            var y0 = min(max(Int(floor(fy)), 0), HH - 1)
            var x1 = min(x0 + 1, HW - 1)
            var y1 = min(y0 + 1, HH - 1)
            var tx = min(max(fx - Float32(x0), Float32(0)), Float32(1))
            var ty = min(max(fy - Float32(y0), Float32(0)), Float32(1))
            var a = low[y0 * HW + x0]
            var b = low[y0 * HW + x1]
            var c = low[y1 * HW + x0]
            var d = low[y1 * HW + x1]
            var out = UInt32(0)
            for k in range(3):
                var shift = UInt32(8 * k)
                var av = Float32((a >> shift) & 255) * (1 - tx) + Float32((b >> shift) & 255) * tx
                var cv = Float32((c >> shift) & 255) * (1 - tx) + Float32((d >> shift) & 255) * tx
                var value = UInt32(av * (1 - ty) + cv * ty)
                out |= value << shift
            comptime if TAA:
                out = resolve(sm, a, b, c, d, out, x, y, ph)
            img[i] = out
        comptime if TAA:  # this frame before glow and grain is next frame's history
            sm.ptr.unsafe_offset(OTA).bitcast[UInt32]()[2 * i + (ph[0] & 1)] = img[i]
        glow(img, low, cam, i, x, y)
        img[i] = post(img[i], cam, x, y)


def clear_sm(sm: TVIS, cam: TCAM):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < NAUX:
        sm[i] = EMPTY
    comptime if TAA:
        if i == 0:  # once per frame, before any reader: roll the camera, count frames, no history after a cut (eye > 5 m, sun moved)
            var t = sm.ptr.unsafe_offset(OTL).bitcast[Float32]()
            var d = V3(cam[0] - t[NCAM], cam[1] - t[NCAM + 1], cam[2] - t[NCAM + 2], 0)
            var s = V3(cam[17] - t[NCAM + 17], cam[18] - t[NCAM + 18], cam[19] - t[NCAM + 19], 0)
            var ok = dot(d, d) < 25 and dot(s, s) < 1e-4
            for k in range(NCAM):
                t[k] = t[NCAM + k]
                t[NCAM + k] = cam[k]
            t[2 * NCAM] = t[2 * NCAM] + 1 if ok else Float32(0)
            t[2 * NCAM + 1] = Float32(1) if ok else Float32(0)


comptime P_CLEAR_SM = 0  # slots of the caller's profile table (frame's `pr`): milliseconds per pass, then the counters
comptime P_SUNCAM = 1
comptime P_SHADOW = 2  # raster from the sun into the shadow map
comptime P_CLEAR = 3
comptime P_RASTER = 4  # raster from the view camera
comptime P_GLASS = 5  # glass and mirror layers
comptime P_SHADE = 6  # the half-res march + shading
comptime P_UPSCALE = 7  # the quad resolve + TAA
comptime LAUNCH = 8  # GPU launches counted (kernels, and the caller's copies)
comptime LAST = 9  # time of the previous lap, ns
comptime P_COPY = 10  # the walk's own steps: device-to-host copy and its wait, window upload + present, fifo write
comptime P_PRESENT = 11
comptime P_FIFO = 12
comptime NPROF = 16
comptime PROF = get_defined_int["PROF", 0]()  # 1: frame() waits after every pass and adds its milliseconds to the caller's table


def lap(ctx: DeviceContext, pr: Int, k: Int) raises:
    """Count one launch in slot LAUNCH of the caller's Float64 table (address pr, 0 = none); with PROF add the pass's
    milliseconds since the previous lap to slot k (the wait serialises the passes: the split, not the overlapped total)."""
    if pr == 0:
        return
    var t = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=pr)
    t[LAUNCH] += 1
    comptime if PROF > 0:
        ctx.synchronize()
        var now = Float64(perf_counter_ns())
        t[k] += (now - t[LAST]) / 1e6
        t[LAST] = now


def frame(ctx: DeviceContext, kc: type_of(DeviceContext(0).compile_function[clear]()), kr: type_of(DeviceContext(0).compile_function[raster]()),
          ks: type_of(DeviceContext(0).compile_function[shade]()), ku: type_of(DeviceContext(0).compile_function[upscale]()), ksc: type_of(DeviceContext(0).compile_function[suncam]()),
          kcs: type_of(DeviceContext(0).compile_function[clear_sm]()), vis: TVIS, sm: TVIS, img: TIMG, low: TLOW, tpi: TPI, tpf: TPF, trv: TRV, trt: TRT,
          trp: TRP, tpose: TPOSE, tcam: TCAM, tscam: TCAM, pr: Int = 0) raises:
    """One frame's passes on the card: aux layers cleared, (SHADOW=2) sun camera + shadow map, view raster, (GLASS) the
    see-through layer, (REFLECT) the mirror layer; shade."""
    var g = (F * NTRI + NBIRD * NTB + 255) // 256  # + the flock's triangles, drawn by the view raster
    comptime if PROF > 0:
        if pr != 0:
            UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=pr)[LAST] = Float64(perf_counter_ns())
    comptime if NAUX > 0:
        ctx.enqueue_function(kcs, sm, tcam, grid_dim=(NAUX + 255) // 256, block_dim=256)
        lap(ctx, pr, P_CLEAR_SM)
    comptime if SHADOW == 2:
        ctx.enqueue_function(ksc, tcam, tscam, grid_dim=1, block_dim=1)
        lap(ctx, pr, P_SUNCAM)
        ctx.enqueue_function(kr, sm, tpi, tpf, trv, trt, trp, tpose, tscam, Int32(0), Int32(0), Int32(0), grid_dim=g, block_dim=256)
        lap(ctx, pr, P_SHADOW)
    ctx.enqueue_function(kc, vis, grid_dim=(W * H + 255) // 256, block_dim=256)
    lap(ctx, pr, P_CLEAR)
    ctx.enqueue_function(kr, vis, tpi, tpf, trv, trt, trp, tpose, tcam, Int32(0), Int32(0), Int32(1), grid_dim=g, block_dim=256)
    lap(ctx, pr, P_RASTER)
    comptime if GLASS:
        ctx.enqueue_function(kr, sm, tpi, tpf, trv, trt, trp, tpose, tcam, Int32(OGL), Int32(1), Int32(0), grid_dim=g, block_dim=256)
        lap(ctx, pr, P_GLASS)
    comptime if REFLECT:
        ctx.enqueue_function(kr, sm, tpi, tpf, trv, trt, trp, tpose, tcam, Int32(OMR), Int32(2), Int32(0), grid_dim=g, block_dim=256)
        lap(ctx, pr, P_GLASS)
    ctx.enqueue_function(ks, vis, sm, img, low, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(SHT + W * H + 255) // 256 if LIVING else (W * H + 255) // 256, block_dim=256)
    lap(ctx, pr, P_SHADE)
    comptime if LIVING:
        ctx.enqueue_function(ku, vis, sm, img, low, tcam, grid_dim=(W * H + 255) // 256, block_dim=256)
        lap(ctx, pr, P_UPSCALE)


def stream(ctx: DeviceContext, kc: type_of(DeviceContext(0).compile_function[clear]()), kr: type_of(DeviceContext(0).compile_function[raster]()),
           ks: type_of(DeviceContext(0).compile_function[shade]()), ku: type_of(DeviceContext(0).compile_function[upscale]()), ksc: type_of(DeviceContext(0).compile_function[suncam]()),
           kcs: type_of(DeviceContext(0).compile_function[clear_sm]()), vis: TVIS, sm: TVIS, img: TIMG, low: TLOW, tpi: TPI, tpf: TPF, trv: TRV, trt: TRT, trp: TRP,
           tpose: TPOSE, tcam: TCAM, tscam: TCAM, cam_b: type_of(DeviceContext(0).enqueue_create_buffer[DType.float32](NCAM)), mut camh: List[Float32],
           img_b: type_of(DeviceContext(0).enqueue_create_buffer[DType.uint32](W * H)), out_h: type_of(DeviceContext(0).enqueue_create_host_buffer[DType.uint32](W * H)),
           frames: Int) raises:
    """Raw RGBA frames to fd 1 with one write(2) each, the camera orbiting the target; ends after FRAMES frames
    or when the reader closes the pipe. Allocates nothing: the camera goes through the same NCAM-float buffer."""
    var tx = camh[3]; var ty = camh[4]
    var rx = camh[0] - tx; var ry = camh[1] - ty
    var fr = 0
    while frames == 0 or fr < frames:
        var ang = Float32(fr) * 3.14159265 / 180
        camh[0] = tx + rx * cos(ang) - ry * sin(ang)
        camh[1] = ty + rx * sin(ang) + ry * cos(ang)
        ctx.enqueue_copy(cam_b, camh.unsafe_ptr())
        frame(ctx, kc, kr, ks, ku, ksc, kcs, vis, sm, img, low, tpi, tpf, trv, trt, trp, tpose, tcam, tscam)
        ctx.enqueue_copy(out_h, img_b)
        ctx.synchronize()
        var left = W * H * 4
        var p = out_h.unsafe_ptr().unsafe_bitcast[UInt8]()
        while left > 0:
            var n = external_call["__write", Int](Int(1), p.bitcast[NoneType](), left)
            if n <= 0:
                return  # reader gone
            p = p + n
            left -= n
        fr += 1


def load(path: String, nbytes: Int) raises -> List[UInt8]:
    var raw = open(path, "r").read_bytes()
    if len(raw) != nbytes:
        raise Error(path + ": expected " + String(nbytes) + " bytes, got " + String(len(raw)))
    return raw^


def main() raises:
    comptime assert F <= (1 << (32 - TB)), "id = animal << TB | triangle must fit 32 bits"
    var a = argv()
    var rdir = String(a[1])
    var frames = Int(String(a[4]))
    var camh = List[Float32](length=NCAM, fill=0)
    for k in range(6):
        camh[k] = Float32(atof(String(a[5 + k])))
    var fovy = Float32(atof(String(a[11]))) * 3.14159265 / 180
    camh[6] = fovy
    camh[7] = Float32(H) * 0.5 / tan(fovy * 0.5)
    camh[10] = Float32(atof(String(a[12])))
    comptime if LIVING:  # time of day s (default noon), metres per world unit (default 2)
        sky_cam(camh, Float32(atof(String(a[13]))) if len(a) > 13 else Float32(75), Float32(atof(String(a[14]))) if len(a) > 14 else Float32(20))
    comptime if CITY:  # seed, origin cell x y, far m (default 1 0 0 1500)
        for k in range(4):
            camh[16 + k] = Float32(atof(String(a[13 + k]))) if len(a) > 16 else (Float32(1500) if k == 3 else Float32(1 if k == 0 else 0))
    # MAX pre-reserves most of the card per process by default (22.7 GB measured). Every buffer here is made
    # once, so no pool: VRAM is then exactly the buffers plus the HIP runtime (152 MB). A caller's setting wins.
    _ = external_call["setenv", Int32]("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT".unsafe_ptr(), "0".unsafe_ptr(), Int32(0))
    var ctx = DeviceContext(0)  # every buffer below is created once; a frame allocates nothing
    var pi_h = load(rdir + "rmi.bin", 20 * NPART)
    var pf_h = load(rdir + "rmf.bin", 4 * PFS * NPART)
    comptime if LIVING:  # the world table rides behind the parts (LWO); path from LIVING_BIN
        pf_h += List[UInt8](length=4 * BLK, fill=0)  # no flock in this tool
        pf_h += open(String(getenv("LIVING_BIN")), "r").read_bytes()
    var rv_h = load(rdir + "rv.bin", 2 * RVI * NRV)
    var rt_h = load(rdir + "rt.bin", 2 * NRT)
    var rp_h = load(rdir + "rp.bin", NTRI)
    var pose_h = load(String(a[2]), 4 * PROWS * F)
    var pi_b = ctx.enqueue_create_buffer[DType.int32](5 * NPART)
    var pf_b = ctx.enqueue_create_buffer[DType.float32](len(pf_h) // 4)
    var rv_b = ctx.enqueue_create_buffer[DType.int16](RVI * NRV)
    var rt_b = ctx.enqueue_create_buffer[DType.uint16](NRT)
    var rp_b = ctx.enqueue_create_buffer[DType.uint8](NTRI)
    var pose_b = ctx.enqueue_create_buffer[DType.float32](PROWS * F)
    var cam_b = ctx.enqueue_create_buffer[DType.float32](NCAM)
    var vis_b = ctx.enqueue_create_buffer[DType.uint64](W * H)
    var sm_b = ctx.enqueue_create_buffer[DType.uint64](NSM)
    var scam_b = ctx.enqueue_create_buffer[DType.float32](NCAM)
    var img_b = ctx.enqueue_create_buffer[DType.uint32](W * H)
    var low_b = ctx.enqueue_create_buffer[DType.uint32](HW * HH)
    var out_h = ctx.enqueue_create_host_buffer[DType.uint32](W * H)
    ctx.enqueue_copy(pi_b, pi_h.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(pf_b, pf_h.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(rv_b, rv_h.unsafe_ptr().unsafe_bitcast[Int16]())
    ctx.enqueue_copy(rt_b, rt_h.unsafe_ptr().unsafe_bitcast[UInt16]())
    ctx.enqueue_copy(rp_b, rp_h.unsafe_ptr().unsafe_bitcast[UInt8]())
    ctx.enqueue_copy(pose_b, pose_h.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(cam_b, camh.unsafe_ptr())
    var kc = ctx.compile_function[clear]()
    var kr = ctx.compile_function[raster]()
    var ks = ctx.compile_function[shade]()
    var ku = ctx.compile_function[upscale]()
    var ksc = ctx.compile_function[suncam]()
    var kcs = ctx.compile_function[clear_sm]()
    var vis = TileTensor(vis_b, LVIS)
    var sm = TileTensor(sm_b, LVIS)  # NSM texels: read only when SHADOW == 2, then NSM == W * H
    var tscam = TileTensor(scam_b, LCAM)
    var img = TileTensor(img_b, LVIS)
    var low = TileTensor(low_b, LLOW)
    var tpi = TileTensor(pi_b, LPI)
    var tpf = TileTensor(pf_b, LPF)
    var trv = TileTensor(rv_b, LRV)
    var trt = TileTensor(rt_b, LRT)
    var trp = TileTensor(rp_b, LRP)
    var tpose = TileTensor(pose_b, LPOSE)
    var tcam = TileTensor(cam_b, LCAM)
    if String(a[3]) == "-":
        stream(ctx, kc, kr, ks, ku, ksc, kcs, vis, sm, img, low, tpi, tpf, trv, trt, trp, tpose, tcam, tscam, cam_b, camh, img_b, out_h, frames)
        return
    var t0 = perf_counter_ns()
    for fr in range(frames + 1):
        if fr == 1:
            ctx.synchronize()
            t0 = perf_counter_ns()
        frame(ctx, kc, kr, ks, ku, ksc, kcs, vis, sm, img, low, tpi, tpf, trv, trt, trp, tpose, tcam, tscam)
        ctx.enqueue_copy(out_h, img_b)
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / Float64(max(frames, 1)) / 1e6
    var px = out_h.unsafe_ptr()
    if String(a[3]).endswith(".rgba"):  # raw RGBA, alpha 255 where an animal covers the pixel (render_check.py)
        with open(String(a[3]), "w") as file:
            file.write_bytes(Span(unsafe_ptr=px.unsafe_bitcast[UInt8](), length=W * H * 4))
        print(F, "animals,", F * NTRI, "triangles,", W, "x", H, ":", ms, "ms per frame (clear, raster, shade, copy to host)")
        return
    var ppm = List[UInt8](capacity=W * H * 3)
    for i in range(W * H):
        var v = px[i]
        ppm.append(UInt8(v & 255))
        ppm.append(UInt8((v >> 8) & 255))
        ppm.append(UInt8((v >> 16) & 255))
    with open(String(a[3]), "w") as file:
        file.write("P6\n" + String(W) + " " + String(H) + "\n255\n")
        file.write_bytes(Span(ppm))
    print(F, "animals,", F * NTRI, "triangles,", W, "x", H, ":", ms, "ms per frame (clear, raster, shade, copy to host)")
