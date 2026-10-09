"""Native vision for the spider (docs/SPIDER_BRAIN.md unit 2): one pinhole camera per eye, mounted on the head body,
rendered by render.mojo's own raster pass into a per-animal image, so each animal sees only its own body and its own prey.

Per control step and per eye e: eye_cam builds each animal's camera from the head body's pose (body 1, the prosoma: xpos and
xquat rows the engine already wrote) and the eye table (local position, local axis, fov); the clear, raster and shade passes
below draw it; shade writes luminance (uint8) straight into the retina buffer [animal][eye][H x W]. The scene is the animal's
own parts (flat Lambert on the part colour) over a constant background, no floor texture: a brain that moves its body sees
its own legs, nothing else. Nothing here reads the prey's state: the prey is a body in the model, seen as pixels only.

prey_set is the ENVIRONMENT, not the controller: it writes the prey body's free joint (the last 7 qpos and 6 dof rows) from env.bin (per animal: x0 y0 z0 vx vy t_on, mm, mm/s, s) before each step: still until t_on, then constant velocity.
"""
from max.gpu import block_idx, block_dim, thread_idx
from std.atomic import Atomic, Ordering
from std.math import sqrt, cos, sin
from layout import TileTensor, row_major
import smooth as eng
import render as ren
from render import V3, Q4, rot, cross, dot

comptime F = eng.F
comptime W = ren.W
comptime H = ren.H
comptime NE = 6  # AM_L AM_R AL_L AL_R PL_L PL_R (spider_eyes.py)
comptime LECAM = row_major[F * ren.NCAM]()
comptime LEVIS = row_major[F * W * H]()
comptime LEYE = row_major[NE * 8]()
comptime LRET = row_major[F * NE * W * H]()
comptime LROW = row_major[F * W]()
comptime LENV = row_major[6 * F]()
comptime LGAZE = row_major[F]()
comptime TECAM = TileTensor[DType.float32, type_of(LECAM), MutAnyOrigin]
comptime TEVIS = TileTensor[DType.uint64, type_of(LEVIS), MutAnyOrigin]
comptime TEYE = TileTensor[DType.float32, type_of(LEYE), MutAnyOrigin]
comptime TRET = TileTensor[DType.uint8, type_of(LRET), MutAnyOrigin]
comptime TROW = TileTensor[DType.uint8, type_of(LROW), MutAnyOrigin]
comptime TENV = TileTensor[DType.float32, type_of(LENV), MutAnyOrigin]
comptime TGAZE = TileTensor[DType.float32, type_of(LGAZE), MutAnyOrigin]
comptime NEAR = Float32(1e-3)


def eye_cam(ecam: TECAM, pose: ren.TPOSE, eyes: TEYE, gaze: TGAZE, eye_id: Int32):
    """cam[f] = eye, target, fovy, focal px for eye e of animal f (render.mojo's layout, spacing 0). The two principal (AM)
    eyes turn by gaze[f] degrees about the head's z axis (+ = left): the retina's own muscles (live.mojo -DGAZE=1)."""
    var e = Int(eye_id)
    var f = Int(block_idx.x * block_dim.x + thread_idx.x)
    if f >= F:
        return
    var bq = Q4(pose[ren.OXQ * F + f], pose[(ren.OXQ + 1) * F + f], pose[(ren.OXQ + 2) * F + f], pose[(ren.OXQ + 3) * F + f])  # body 1: the head
    var bp = V3(pose[f], pose[F + f], pose[2 * F + f], 0)
    var lp = V3(eyes[8 * e], eyes[8 * e + 1], eyes[8 * e + 2], 0)
    var ax = V3(eyes[8 * e + 3], eyes[8 * e + 4], eyes[8 * e + 5], 0)
    if e < 2:
        var g = gaze[f] * Float32(0.017453292519943295)
        ax = V3(ax[0] * cos(g) - ax[1] * sin(g), ax[0] * sin(g) + ax[1] * cos(g), ax[2], 0)
    var eye = bp + rot(bq, lp)
    var tgt = eye + rot(bq, ax)
    var o = f * ren.NCAM
    for k in range(ren.NCAM):
        ecam[o + k] = 0
    ecam[o] = eye[0]; ecam[o + 1] = eye[1]; ecam[o + 2] = eye[2]
    ecam[o + 3] = tgt[0]; ecam[o + 4] = tgt[1]; ecam[o + 5] = tgt[2]
    ecam[o + 6] = eyes[8 * e + 6]
    ecam[o + 7] = eyes[8 * e + 7]  # focal length in pixels (tan is host-only)


@always_inline
def basis(ecam: TECAM, f: Int) -> Tuple[V3, V3, V3, V3, Float32]:
    var o = f * ren.NCAM
    var eye = V3(ecam[o], ecam[o + 1], ecam[o + 2], 0)
    var fw = V3(ecam[o + 3], ecam[o + 4], ecam[o + 5], 0) - eye
    fw = fw / sqrt(dot(fw, fw))
    var right = cross(fw, V3(0, 0, 1, 0))
    right = right / sqrt(dot(right, right))
    var up = cross(right, fw)
    return (eye, right, up, fw, ecam[o + 7])


def eye_clear(vis: TEVIS):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < F * W * H:
        vis[i] = ren.EMPTY


def eye_raster(vis: TEVIS, ecam: TECAM, pi: ren.TPI, pf: ren.TPF, rv: ren.TRV, rt: ren.TRT, rp: ren.TRP, pose: ren.TPOSE, cam0: ren.TCAM):
    """render.raster with one camera and one visibility plane per animal."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx >= F * ren.NTRI:
        return
    var f = idx // ren.NTRI
    var t = idx % ren.NTRI
    var p = Int(rp[t])
    var c = basis(ecam, f)
    var sx = SIMD[DType.float32, 4](0)
    var sy = SIMD[DType.float32, 4](0)
    var sz = SIMD[DType.float32, 4](0)
    for k in range(3):
        var d = ren.vertex(pi, pf, rv, rt, pose, cam0, f, p, 3 * t + k) - c[0]
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
            _ = Atomic.min[ordering=Ordering.RELAXED](vis.ptr.unsafe_offset(f * W * H + y * W + x), key)


def eye_shade(vis: TEVIS, ret: TRET, eye_id: Int32, pi: ren.TPI, pf: ren.TPF, rv: ren.TRV, rt: ren.TRT, rp: ren.TRP, pose: ren.TPOSE, cam0: ren.TCAM):
    """Luminance 0..255 of eye e: the nearest part's flat Lambert colour, else the constant background 0.6."""
    var e = Int(eye_id)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= F * W * H:
        return
    var f = i // (W * H)
    var key = vis[i]
    var lum = Float32(0.6)
    if key != ren.EMPTY:
        var t = Int(key & 0x1FFFF)
        var p = Int(rp[t])
        var a = ren.vertex(pi, pf, rv, rt, pose, cam0, f, p, 3 * t)
        var n = cross(ren.vertex(pi, pf, rv, rt, pose, cam0, f, p, 3 * t + 1) - a, ren.vertex(pi, pf, rv, rt, pose, cam0, f, p, 3 * t + 2) - a)
        n = n / max(sqrt(dot(n, n)), Float32(1e-30))
        var lit = Float32(0.35) + Float32(0.65) * abs(dot(n, V3(0.3, 0.2, 0.93, 0)))
        lum = (0.299 * pf[ren.PFS * p + 17] + 0.587 * pf[ren.PFS * p + 18] + 0.114 * pf[ren.PFS * p + 19]) * lit
    ret[(f * NE + e) * W * H + i % (W * H)] = UInt8(min(max(lum, 0), 1) * 255)


def eye_rowmin(ret: TRET, row: TROW):
    """AM_L's darkest pixel per column (the host's vertical-strip retina, agent/spider_eye_loop.py): F x W bytes instead of the
    F x NE x H x W retinas (-DEYEROW=1, live.mojo)."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= F * W:
        return
    var f, c = i // W, i % W
    var m = UInt8(255)
    for h in range(H):
        m = min(m, rebind[Scalar[DType.uint8]](ret[f * NE * W * H + h * W + c]))
    row[i] = m


def prey_set(q: eng.TQ, v: eng.TV, env: TENV, t: Float32):
    """The environment moves the prey: its free joint is the last 7 qpos rows (position, quaternion) and 6 dof rows."""
    var f = Int(block_idx.x * block_dim.x + thread_idx.x)
    if f >= F:
        return
    var on = t > env[5 * F + f]
    var dt = t - env[5 * F + f] if on else Float32(0)
    for k in range(3):
        var vel = env[(3 + k) * F + f] if k < 2 and on else Float32(0)
        var v0 = env[(3 + k) * F + f] if k < 2 else Float32(0)
        q[(eng.NQ - 7 + k) * F + f] = env[k * F + f] + v0 * dt
        v[(eng.NV - 6 + k) * F + f] = vel
        v[(eng.NV - 3 + k) * F + f] = 0
    q[(eng.NQ - 4) * F + f] = 1
    for k in range(1, 4):
        q[(eng.NQ - 4 + k) * F + f] = 0
