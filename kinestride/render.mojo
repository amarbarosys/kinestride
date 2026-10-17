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
from living import cell, lw_albedo, lw_hit, lw_lit, sky_cam

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
comptime LPF = row_major[PFS * NPART + get_defined_int["LIVING", 0]() * 8]()  # parts, then the world header; the table runs on past it (LWF)
comptime RVI = 9  # int16 per vertex in rv.bin: position 3, unit normal x 32767 3, rgb 0..255 3 (callers load 2 * RVI * NRV bytes)
comptime LRV = row_major[NRV * RVI]()  # all positions, then all normals, then all colours
comptime LRT = row_major[NRT]()
comptime LRP = row_major[NTRI]()
comptime LPOSE = row_major[PROWS * F]()
comptime NCAM = 16 + NP  # eye 3, target 3, fovy, focal, ..., [10] grid spacing; [16:] world.mojo terrain params
comptime LCAM = row_major[NCAM]()
comptime SOIL = get_defined_int["SOIL", 0]() == 1  # play: soil and moss on the terrain instead of the grey checker (gates keep the checker)
comptime TERRAIN = get_defined_int["TERRAIN", 0]()  # 1: the ground is world.mojo's terrain (cam[16:24]), not z = 0
comptime CITY = get_defined_int["CITY", 0]()  # 1: the ground is city.mojo's endless city, 1 unit = 1 m (cam[16:20]: seed, origin cell x y, far m)
comptime LIVING = get_defined_int["LIVING", 0]()  # 1: the ground is living.mojo's baked torus (world.bin after the parts in pf; cam[16:32] the sky)
comptime LWF = 8 if LIVING else 0  # world floats the layout covers: the header only. The table is as long as world.bin, read through T = pf + LWO
comptime LWO = PFS * NPART  # where the world table starts in pf
comptime LIVING_FAR = get_defined_int["LIVING_FAR", 125]()  # view distance in cells per 128 of the world's N (125 x 20 m = 2.5 km at N 128, SCALE.md)
comptime CITY_DEPTH = get_defined_int["CITY_DEPTH", 0]()  # 1 (gate): each pixel's city hit distance in m as float32 bits, 1e30 on none
comptime SHADOW = get_defined_int["SHADOW", 2]()  # 0 none, 1 a soft blob under each animal (crowds: no extra pass), 2 shadow map
comptime AA = get_defined_int["AA", 1]()  # 1: silhouette pixels average 4 analytic sub-rays
comptime GLASS = get_defined_int["GLASS", 1]()  # 1: see-through parts (alpha < 1: wings) in their own layer, blended over the scene
comptime REFLECT = get_defined_int["REFLECT", 1]() if TERRAIN == 0 and CITY == 0 and LIVING == 0 and get_defined_int["BOX_MM", 0]() == 0 else 0  # flat floor mirror
comptime REFL = Float32(0.2)  # floor reflectance (MuJoCo's grid material)
comptime OGL = W * H if SHADOW == 2 else 0  # aux layers in sm, each W x H (raster is compiled for W x H): shadow map at 0,
comptime OMR = OGL + (W * H if GLASS else 0)  # then the glass layer at OGL, the mirror layer at OMR
comptime NAUX = OMR + (W * H if REFLECT else 0)
comptime NSM = NAUX if NAUX > 0 else 1  # sm buffer size (uint64), what callers allocate
comptime BLOB = Float32(get_defined_int["BLOB_UM", 1200]()) * 1e-3  # SHADOW=1 blob radius, mm
comptime TVIS = TileTensor[DType.uint64, type_of(LVIS), MutAnyOrigin]
comptime TIMG = TileTensor[DType.uint32, type_of(LVIS), MutAnyOrigin]
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
    var eye = V3(cam[0], cam[1], cam[2], 0)
    var fw = V3(cam[3], cam[4], cam[5], 0) - eye
    fw = fw / sqrt(dot(fw, fw))
    var right = cross(fw, V3(0, 0, 1, 0))
    right = right / sqrt(dot(right, right))
    var up = cross(right, fw)
    return (eye, right, up, fw, cam[7])


def clear(vis: TVIS):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < W * H:
        vis[i] = EMPTY


def raster(vis: TVIS, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM, base: Int32, mode: Int32):
    """Into vis at offset base: mode 0 the opaque parts, 1 the see-through parts (alpha < 1) only, 2 the opaque parts mirrored in z = 0."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx >= F * NTRI:
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
    var area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0])
    if abs(area) < 1e-12:
        return
    var x0 = max(Int(min(sx[0], min(sx[1], sx[2]))), 0)
    var x1 = min(Int(max(sx[0], max(sx[1], sx[2]))) + 1, W - 1)
    var y0 = max(Int(min(sy[0], min(sy[1], sy[2]))), 0)
    var y1 = min(Int(max(sy[0], max(sy[1], sy[2]))) + 1, H - 1)
    var id = UInt64(f << TB | t)
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
def living_sky(cam: TCAM, view: V3) -> V3:
    """The web's sky: horizon to zenith by elevation, sun disc and halo, stars at night (linear)."""
    var f = min(max(view[2] / 0.8, Float32(0)), Float32(1))
    var c = V3(cam[23], cam[24], cam[25], 0) * (1 - f) + V3(cam[20], cam[21], cam[22], 0) * f
    c = c * c
    if cam[30] < 0.5:
        var r = (1 - dot(view, sun(cam))) / 1.6e-4  # ~1 deg disc
        c += V3(1, 0.85, 0.6, 0) * (Float32(1.4) if r < 1 else 0.25 * exp(-(sqrt(r) - 1) * 0.7))
    elif view[2] > 0:
        var q = hash2(Int(floor(view[0] * 400)) * 7919 + Int(floor(view[1] * 400)), Int(floor(view[2] * 400)))
        if q > 0.9985:
            c += V3(0.8, 0.8, 0.9, 0) * cam[30]
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
    var col: V3
    if mat == 2:  # water: two travelling swells tilt the normal; the sky by Fresnel, a sun glint
        var t = cam[27]
        n = unit(V3(0.05 * cos(t * 1.7 + u[0] * 2.1 + u[1] * 1.3), 0.05 * cos(t * 2.3 - u[0] * 0.9 + u[1] * 2.6), 1, 0))
        var r = view - n * (2 * dot(view, n))
        var fr = 0.02 + 0.98 * pow(1 - max(-dot(n, view), Float32(0)), Float32(5))
        var a = lw_albedo(T, u[0], u[1])
        col = light(cam, V3(a[0], a[1], a[2], 0) * V3(a[0], a[1], a[2], 0), n, view, sh, ao, 0) * (1 - fr) + living_sky(cam, r) * fr
        col += V3(1, 0.9, 0.7, 0) * (sh * pow(max(dot(r, sun(cam)), Float32(0)), Float32(200)) * 3 * (1 - cam[30]))
    elif mat >= 3:  # a tree: leaf colour by species (web/world.mjs SPECIES), glow-fern lit at night (hue per tree)
        var g = h[5] * cam[30]
        var sp = mat - 3
        var leaf = V3(0.08, 0.24, 0.12, 0) if sp == 0 else (V3(0.16, 0.3, 0.08, 0) if sp == 1 else (V3(0.06, 0.2, 0.08, 0) if sp == 2 else (V3(0.3, 0.42, 0.12, 0) if sp == 3 else (V3(0.2, 0.28, 0.1, 0) if sp == 4 else V3(0.17, 0.24, 0.16, 0)))))
        var base = leaf * (1 - g) + V3(0.3 + 0.7 * h[6], 1 - 0.5 * h[6], 0.9, 0) * g
        if mat == 9:  # bark: birch white-grey, the rest brown, a little variation per tree
            base = (V3(0.72, 0.7, 0.64, 0) if Int(h[7]) == 3 else V3(0.3, 0.21, 0.14, 0)) * (0.85 + 0.3 * h[6])
        col = light(cam, base * base, n, view, sh, ao, 0.1) + base * base * (1.6 * g)
    else:
        var a = lw_albedo(T, u[0], u[1])
        # ground per habitat: noise scale (1/m) and contrast. 0 scree stones, 1 riparian, 2 meadow grass streaks,
        # 3 spruce moss, 4 heather clumps, 5 beech leaf litter
        var hb = Int(cell(T, Int(floor(u[0] + 0.5)), Int(floor(u[1] + 0.5)), 4))
        var fq = Float32(1.25) if hb == 0 else (Float32(2) if hb == 1 else (Float32(3.3) if hb == 2 else (Float32(1.5) if hb == 3 else (Float32(2) if hb == 4 else Float32(1)))))
        var amp = Float32(0.4) if hb == 0 else (Float32(0.3) if hb == 1 else (Float32(0.35) if hb == 2 else (Float32(0.15) if hb == 3 else (Float32(0.5) if hb == 4 else Float32(0.4)))))
        var nx = hit[0] * fq * (Float32(0.35) if hb == 2 else Float32(1))  # grass: streaks along y
        var grain = 1 - amp * 0.5 + amp * vnoise(nx, hit[1] * fq)
        var far = min(max((h[0] * sqrt(dot(ray, ray)) - 100) / 100, Float32(0)), Float32(1))  # past 100-200 m
        var canopy = far * min(max((0.32 - a[1]) * 6, Float32(0)), Float32(1)) * Float32(1 - get_defined_int["LW_NOCANOPY", 0]())  # dark green: a stand's crowns
        grain = grain * (1 - canopy) + canopy * (0.55 + 0.9 * vnoise(hit[0] * 0.17, hit[1] * 0.17) * vnoise(hit[0] * 0.5 + 7, hit[1] * 0.5))  # 6 m crowns, gaps
        var base = V3(a[0], a[1], a[2], 0) * grain
        col = light(cam, base * base, n, view, sh, ao * a[3], 0)
    var d = h[0] * sqrt(dot(ray, ray)) / cam[16]  # world units, as the web's fog
    var k = 0.006 * (0.6 + 0.5 * exp(-(u[2] - pf[LWO + 1]) / 3))  # per 20 m cell: haze visibility about 10 km, a little thicker in valleys
    var far = pf[LWO] * Float32(LIVING_FAR) / 128
    var fog = max(1 - exp(-max(d - 2.5, Float32(0)) * k), min(max((d - far * 0.7) / (far * 0.3), Float32(0)), Float32(1)))
    fog *= Float32(1 - get_defined_int["LW_NOFOG", 0]())  # debug: LW_NOFOG=1 shows the bare surface
    var hor = V3(cam[23], cam[24], cam[25], 0)
    return (col * (1 - fog) + hor * hor * fog) * cam[26]


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
        amb = (V3(cam[20], cam[21], cam[22], 0) * (0.5 + 0.5 * n[2]) + V3(0.20, 0.17, 0.13, 0) * (0.5 - 0.5 * n[2]) * day) * ((0.25 + 0.75 * day) * ao)
        lc = V3(1, 0.95 - 0.25 * cam[31], 0.85 - 0.4 * cam[31], 0) * day + V3(0.05, 0.07, 0.16, 0) * cam[30]
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
def sample(sm: TVIS, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM, eye: V3, ray: V3,
           key: UInt64, tt: Float32, ao: Float32, mk: UInt64) -> V3:
    """Linear colour along ray: the animal triangle key at ray parameter tt when it is nearer than the ground,
    else the ground or box (the flat floor mixes in the mirror layer's key mk at REFL), else the sky. Fog: ground only."""
    comptime if LIVING:
        var h = living(cam, pf, eye, ray)
        if key != EMPTY and tt < h[0]:
            return animal(sm, pi, pf, rv, rt, rp, pose, cam, eye, ray, key, tt, ao, False) * cam[26]
        return meadow(sm, pf, cam, eye, ray, h, ao) if h[0] < 3.4e38 else living_sky(cam, unit(ray)) * cam[26]
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


def shade(vis: TVIS, sm: TVIS, img: TIMG, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= W * H:
        return
    var c = camera(cam)
    var px = Float32(i % W) + 0.5 - Float32(W) * 0.5
    var py = Float32(i // W) + 0.5 - Float32(H) * 0.5
    var ray = c[3] * c[4] + c[1] * px - c[2] * py
    var key = vis[i]
    var g = ground_hit(cam, pf, c[0], ray)
    comptime if CITY_DEPTH:
        img[i] = bitcast[DType.uint32](g[0] * sqrt(dot(ray, ray)) if g[0] < 3.4e38 else Float32(1e30))
        return
    var front = key != EMPTY and depth(key) < g[0] * c[4]  # raster depth is view z = ray parameter x focal
    var z0 = depth(key) if front else g[0] * c[4]
    var mk = sm.ptr.unsafe_offset(OMR + i)[] if REFLECT else EMPTY
    var gk = sm.ptr.unsafe_offset(OGL + i)[] if GLASS else EMPTY
    var glass = gk != EMPTY and depth(gk) < z0
    var fly = (front and pi[5 * Int(rp[Int(key & TMASK)])] > 0) or glass  # alpha: animal parts only (not world parts)
    var ao = occlusion(vis, i, z0, c[4]) if z0 < 3.4e38 else Float32(1)
    var col = sample(sm, pi, pf, rv, rt, rp, pose, cam, c[0], ray, key, depth(key) / c[4], ao, mk)
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
                        var f = Int(kk >> TB64) & ((1 << (32 - TB)) - 1)
                        var t = Int(kk & TMASK)
                        var p = Int(rp[t])
                        var h = tri_hit(c[0], rs, vertex(pi, pf, rv, rt, pose, cam, f, p, 3 * t), vertex(pi, pf, rv, rt, pose, cam, f, p, 3 * t + 1),
                                        vertex(pi, pf, rv, rt, pose, cam, f, p, 3 * t + 2))
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
    img[i] = r | gg << 8 | b << 16 | (UInt32(255) if fly else UInt32(0)) << 24


def clear_sm(sm: TVIS):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < NSM:
        sm[i] = EMPTY


def frame(ctx: DeviceContext, kc: type_of(DeviceContext(0).compile_function[clear]()), kr: type_of(DeviceContext(0).compile_function[raster]()),
          ks: type_of(DeviceContext(0).compile_function[shade]()), ksc: type_of(DeviceContext(0).compile_function[suncam]()),
          kcs: type_of(DeviceContext(0).compile_function[clear_sm]()), vis: TVIS, sm: TVIS, img: TIMG, tpi: TPI, tpf: TPF, trv: TRV, trt: TRT,
          trp: TRP, tpose: TPOSE, tcam: TCAM, tscam: TCAM) raises:
    """One frame's passes on the card: aux layers cleared, (SHADOW=2) sun camera + shadow map, view raster, (GLASS) the
    see-through layer, (REFLECT) the mirror layer; shade."""
    var g = (F * NTRI + 255) // 256
    comptime if NAUX > 0:
        ctx.enqueue_function(kcs, sm, grid_dim=(NSM + 255) // 256, block_dim=256)
    comptime if SHADOW == 2:
        ctx.enqueue_function(ksc, tcam, tscam, grid_dim=1, block_dim=1)
        ctx.enqueue_function(kr, sm, tpi, tpf, trv, trt, trp, tpose, tscam, Int32(0), Int32(0), grid_dim=g, block_dim=256)
    ctx.enqueue_function(kc, vis, grid_dim=(W * H + 255) // 256, block_dim=256)
    ctx.enqueue_function(kr, vis, tpi, tpf, trv, trt, trp, tpose, tcam, Int32(0), Int32(0), grid_dim=g, block_dim=256)
    comptime if GLASS:
        ctx.enqueue_function(kr, sm, tpi, tpf, trv, trt, trp, tpose, tcam, Int32(OGL), Int32(1), grid_dim=g, block_dim=256)
    comptime if REFLECT:
        ctx.enqueue_function(kr, sm, tpi, tpf, trv, trt, trp, tpose, tcam, Int32(OMR), Int32(2), grid_dim=g, block_dim=256)
    ctx.enqueue_function(ks, vis, sm, img, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(W * H + 255) // 256, block_dim=256)


def stream(ctx: DeviceContext, kc: type_of(DeviceContext(0).compile_function[clear]()), kr: type_of(DeviceContext(0).compile_function[raster]()),
           ks: type_of(DeviceContext(0).compile_function[shade]()), ksc: type_of(DeviceContext(0).compile_function[suncam]()),
           kcs: type_of(DeviceContext(0).compile_function[clear_sm]()), vis: TVIS, sm: TVIS, img: TIMG, tpi: TPI, tpf: TPF, trv: TRV, trt: TRT, trp: TRP,
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
        frame(ctx, kc, kr, ks, ksc, kcs, vis, sm, img, tpi, tpf, trv, trt, trp, tpose, tcam, tscam)
        ctx.enqueue_copy(out_h, img_b)
        ctx.synchronize()
        var left = W * H * 4
        var p = out_h.unsafe_ptr().unsafe_bitcast[UInt8]()
        while left > 0:
            var n = external_call["write", Int](Int(1), p.bitcast[NoneType](), left)
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
    var ksc = ctx.compile_function[suncam]()
    var kcs = ctx.compile_function[clear_sm]()
    var vis = TileTensor(vis_b, LVIS)
    var sm = TileTensor(sm_b, LVIS)  # NSM texels: read only when SHADOW == 2, then NSM == W * H
    var tscam = TileTensor(scam_b, LCAM)
    var img = TileTensor(img_b, LVIS)
    var tpi = TileTensor(pi_b, LPI)
    var tpf = TileTensor(pf_b, LPF)
    var trv = TileTensor(rv_b, LRV)
    var trt = TileTensor(rt_b, LRT)
    var trp = TileTensor(rp_b, LRP)
    var tpose = TileTensor(pose_b, LPOSE)
    var tcam = TileTensor(cam_b, LCAM)
    if String(a[3]) == "-":
        stream(ctx, kc, kr, ks, ksc, kcs, vis, sm, img, tpi, tpf, trv, trt, trp, tpose, tcam, tscam, cam_b, camh, img_b, out_h, frames)
        return
    var t0 = perf_counter_ns()
    for fr in range(frames + 1):
        if fr == 1:
            ctx.synchronize()
            t0 = perf_counter_ns()
        frame(ctx, kc, kr, ks, ksc, kcs, vis, sm, img, tpi, tpf, trv, trt, trp, tpose, tcam, tscam)
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
