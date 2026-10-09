"""Renderer for the body engine (docs/RENDERER.md): a compute rasteriser over the engine's own poses.

Poses are read in the engine's layout: rows [row][animal], xpos of bodies 1.. in rows 0..3(NB-1), xquat from
row 3(NB-1); the buffer is the engine's own (here: its smooth.bin), so drawing copies no pose. Mesh tables come
from model.py meshes (int16 vertices per part box, uint16 triangles, uint8 part per triangle).

Per frame, three passes and no allocation:
  clear   visibility buffer = all ones
  raster  one thread per (animal, triangle): world vertices from body pose x part pose, perspective, edge
          functions over the triangle's pixel box, 64-bit atomic min of (depth bits << 32 | animal << 17 | tri)
  shade   one thread per pixel: floor (checker on z = 0) or the triangle's flat normal, Lambert on the part colour
then the RGBA8 image goes to a pinned host buffer allocated once.

usage: render RENDER_DIR/ POSES.bin OUT FRAMES ex ey ez tx ty tz fovy_deg spacing
       OUT: .ppm image, .rgba raw (alpha 255 on animals), or - to stream raw RGBA frames to stdout while the
       camera orbits the target 1 degree a frame (FRAMES 0 = until the reader closes the pipe), e.g.
       render ... - 0 ... | mpv --demuxer=rawvideo --demuxer-rawvideo-w=W --demuxer-rawvideo-h=H
                                --demuxer-rawvideo-mp-format=rgba -
       spacing 0 draws animals where they are; spacing s > 0 lays them on a grid of s, root over the cell centre
"""
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.math import cos, sin, sqrt, tan
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from world import NP, PW, height_grad

comptime W = get_defined_int["W", 1920]()
comptime H = get_defined_int["H", 1080]()
comptime NB = get_defined_int["NBODY"]()
comptime F = get_defined_int["FLIES"]()  # animals in the pose buffer, all drawn
comptime PROWS = get_defined_int["PROWS"]()  # rows of the pose buffer
comptime NPART = get_defined_int["NPART"]()
comptime NRV = get_defined_int["NRV"]()
comptime NRT = get_defined_int["NRT"]()  # triangle indices (3 per triangle)
comptime NTRI = NRT // 3
comptime PFS = 21  # part floats: pos 3, quat 4, box offset 3, box scale 3, sphere centre 3, radius 1, rgba 4
comptime GX = get_defined_int["GX", 1]()
comptime BOX = Float32(get_defined_int["BOX_MM", 0]())  # > 0: the camera is inside a closed cube of this side (model.cube)  # grid columns when spacing > 0
comptime OXQ = 3 * (NB - 1)
comptime EMPTY = UInt64.MAX
comptime NEAR = Float32(1e-3)

comptime LVIS = row_major[W * H]()
comptime LPI = row_major[5 * NPART]()
comptime LPF = row_major[PFS * NPART]()
comptime LRV = row_major[NRV * 3]()
comptime LRT = row_major[NRT]()
comptime LRP = row_major[NTRI]()
comptime LPOSE = row_major[PROWS * F]()
comptime NCAM = 16 + NP  # eye 3, target 3, fovy, focal, ..., [10] grid spacing; [16:] world.mojo terrain params
comptime LCAM = row_major[NCAM]()
comptime SOIL = get_defined_int["SOIL", 0]() == 1  # play: soil and moss on the terrain instead of the grey checker (gates keep the checker)
comptime TERRAIN = get_defined_int["TERRAIN", 0]()  # 1: the ground is world.mojo's terrain (cam[16:24]), not z = 0
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


def raster(vis: TVIS, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx >= F * NTRI:
        return
    var f = idx // NTRI
    var t = idx % NTRI
    var p = Int(rp[t])
    var c = camera(cam)
    var sx = SIMD[DType.float32, 4](0)
    var sy = SIMD[DType.float32, 4](0)
    var sz = SIMD[DType.float32, 4](0)
    for k in range(3):
        var d = vertex(pi, pf, rv, rt, pose, cam, f, p, 3 * t + k) - c[0]
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
    var id = UInt64(f << 17 | t)
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
            _ = Atomic.min[ordering=Ordering.RELAXED](vis.ptr.unsafe_offset(y * W + x), key)


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


def shade(vis: TVIS, img: TIMG, pi: TPI, pf: TPF, rv: TRV, rt: TRT, rp: TRP, pose: TPOSE, cam: TCAM):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= W * H:
        return
    var c = camera(cam)
    var ray = c[3] * c[4] + c[1] * (Float32(i % W) + 0.5 - Float32(W) * 0.5) - c[2] * (Float32(i // W) + 0.5 - Float32(H) * 0.5)
    var key = vis[i]
    var floor_depth = Float32(3.4e38)
    var face = -1  # BOX: 0, 1 x walls, 2, 3 y walls, 4 floor, 5 ceiling
    var tmin = Float32(3.4e38)
    comptime if BOX > 0:  # exit point of the cube [-B/2, B/2]^2 x [0, B] along the ray
        for a in range(3):
            if ray[a] != 0:
                var lo = Float32(0) if a == 2 else -BOX * 0.5
                var t = ((lo + BOX if ray[a] > 0 else lo) - c[0][a]) / ray[a]
                if t > 0 and t < tmin:
                    tmin = t
                    face = 2 * a + (1 if ray[a] > 0 else 0)
        floor_depth = tmin * c[4]
    elif TERRAIN:
        tmin = ground(cam, c[0], ray)
        if tmin < 3.4e38:
            floor_depth = tmin * c[4]
    else:
        if ray[2] < 0:
            floor_depth = -c[0][2] / ray[2] * c[4]  # ray scaled so its forward component is the focal length
    var rgb: V3
    var fly = key != EMPTY and UInt32(key >> 32) < UInt32(floor_depth.to_bits())
    if key != EMPTY and UInt32(key >> 32) < UInt32(floor_depth.to_bits()):  # positive floats order as their bits
        var f = Int((key >> 17) & 0x7FFF)
        var t = Int(key & 0x1FFFF)
        var p = Int(rp[t])
        var base = 3 * t
        var a = vertex(pi, pf, rv, rt, pose, cam, f, p, base)
        var n = cross(vertex(pi, pf, rv, rt, pose, cam, f, p, base + 1) - a, vertex(pi, pf, rv, rt, pose, cam, f, p, base + 2) - a)
        n = n / max(sqrt(dot(n, n)), Float32(1e-30))
        var light = V3(0.3, 0.2, 0.93, 0)
        var lit = Float32(0.35) + Float32(0.65) * abs(dot(n, light))
        rgb = V3(pf[PFS * p + 17], pf[PFS * p + 18], pf[PFS * p + 19], 0) * lit
    elif BOX > 0 and face >= 0:
        var hit = c[0] + ray * tmin
        var u = hit[1] if face < 2 else hit[0]
        var v = hit[1] if face >= 4 else hit[2]
        var check = (Int(u + 1e4) + Int(v + 1e4)) % 2
        var tint = V3(0.55, 0.57, 0.6, 0) if face == 4 else (V3(0.62, 0.6, 0.55, 0) if face == 5 else V3(0.5, 0.56, 0.62, 0))
        rgb = tint * (Float32(0.8) if check == 0 else Float32(1.0))
    elif TERRAIN and floor_depth < 3.4e38:
        var hit = c[0] + ray * tmin
        var hg = height_grad(hit[0], hit[1], terrain(cam))
        var n = V3(-hg[1], -hg[2], 1, 0) / sqrt(hg[1] * hg[1] + hg[2] * hg[2] + 1)
        var lit = (Float32(0.35) + Float32(0.65) * dot(n, V3(0.3, 0.2, 0.93, 0))) / (Float32(0.35) + Float32(0.65) * Float32(0.93))
        var check = (Int(hit[0] + 1e4) + Int(hit[1] + 1e4)) % 2
        var fade = Float32(1) / (1 + 0.02 * sqrt(dot(hit - c[0], hit - c[0])))
        var base = soil(hit[0], hit[1]) if SOIL else V3(0.55, 0.57, 0.6, 0) * (Float32(0.8) if check == 0 else Float32(1.0))
        rgb = base * lit * fade + V3(0.6, 0.65, 0.72, 0) * (1 - fade)
    elif not TERRAIN and ray[2] < 0:
        var hit = c[0] + ray * (-c[0][2] / ray[2])
        var check = (Int(hit[0] + 1e4) + Int(hit[1] + 1e4)) % 2
        var fade = Float32(1) / (1 + 0.02 * sqrt(dot(hit - c[0], hit - c[0])))
        rgb = V3(0.55, 0.57, 0.6, 0) * (Float32(0.8) if check == 0 else Float32(1.0)) * fade + V3(0.6, 0.65, 0.72, 0) * (1 - fade)
    else:
        rgb = V3(0.6, 0.65, 0.72, 0)
    var r = UInt32(min(max(rgb[0], 0), 1) * 255)
    var g = UInt32(min(max(rgb[1], 0), 1) * 255)
    var b = UInt32(min(max(rgb[2], 0), 1) * 255)
    img[i] = r | g << 8 | b << 16 | (UInt32(255) if fly else UInt32(0)) << 24


def stream(ctx: DeviceContext, kc: type_of(DeviceContext(0).compile_function[clear]()), kr: type_of(DeviceContext(0).compile_function[raster]()),
           ks: type_of(DeviceContext(0).compile_function[shade]()), vis: TVIS, img: TIMG, tpi: TPI, tpf: TPF, trv: TRV, trt: TRT, trp: TRP,
           tpose: TPOSE, tcam: TCAM, cam_b: type_of(DeviceContext(0).enqueue_create_buffer[DType.float32](NCAM)), mut camh: List[Float32],
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
        ctx.enqueue_function(kc, vis, grid_dim=(W * H + 255) // 256, block_dim=256)
        ctx.enqueue_function(kr, vis, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(F * NTRI + 255) // 256, block_dim=256)
        ctx.enqueue_function(ks, vis, img, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(W * H + 255) // 256, block_dim=256)
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
    comptime assert NTRI <= (1 << 17) and F <= (1 << 15), "id = animal << 17 | triangle must fit 32 bits"
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
    # MAX pre-reserves most of the card per process by default (22.7 GB measured). Every buffer here is made
    # once, so no pool: VRAM is then exactly the buffers plus the HIP runtime (152 MB). A caller's setting wins.
    _ = external_call["setenv", Int32]("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT".unsafe_ptr(), "0".unsafe_ptr(), Int32(0))
    var ctx = DeviceContext(0)  # every buffer below is created once; a frame allocates nothing
    var pi_h = load(rdir + "rmi.bin", 20 * NPART)
    var pf_h = load(rdir + "rmf.bin", 4 * PFS * NPART)
    var rv_h = load(rdir + "rv.bin", 6 * NRV)
    var rt_h = load(rdir + "rt.bin", 2 * NRT)
    var rp_h = load(rdir + "rp.bin", NTRI)
    var pose_h = load(String(a[2]), 4 * PROWS * F)
    var pi_b = ctx.enqueue_create_buffer[DType.int32](5 * NPART)
    var pf_b = ctx.enqueue_create_buffer[DType.float32](PFS * NPART)
    var rv_b = ctx.enqueue_create_buffer[DType.int16](3 * NRV)
    var rt_b = ctx.enqueue_create_buffer[DType.uint16](NRT)
    var rp_b = ctx.enqueue_create_buffer[DType.uint8](NTRI)
    var pose_b = ctx.enqueue_create_buffer[DType.float32](PROWS * F)
    var cam_b = ctx.enqueue_create_buffer[DType.float32](NCAM)
    var vis_b = ctx.enqueue_create_buffer[DType.uint64](W * H)
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
    var vis = TileTensor(vis_b, LVIS)
    var img = TileTensor(img_b, LVIS)
    var tpi = TileTensor(pi_b, LPI)
    var tpf = TileTensor(pf_b, LPF)
    var trv = TileTensor(rv_b, LRV)
    var trt = TileTensor(rt_b, LRT)
    var trp = TileTensor(rp_b, LRP)
    var tpose = TileTensor(pose_b, LPOSE)
    var tcam = TileTensor(cam_b, LCAM)
    if String(a[3]) == "-":
        stream(ctx, kc, kr, ks, vis, img, tpi, tpf, trv, trt, trp, tpose, tcam, cam_b, camh, img_b, out_h, frames)
        return
    var t0 = perf_counter_ns()
    for fr in range(frames + 1):
        if fr == 1:
            ctx.synchronize()
            t0 = perf_counter_ns()
        ctx.enqueue_function(kc, vis, grid_dim=(W * H + 255) // 256, block_dim=256)
        ctx.enqueue_function(kr, vis, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(F * NTRI + 255) // 256, block_dim=256)
        ctx.enqueue_function(ks, vis, img, tpi, tpf, trv, trt, trp, tpose, tcam, grid_dim=(W * H + 255) // 256, block_dim=256)
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
