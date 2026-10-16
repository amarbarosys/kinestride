"""Living World as a Kinestride ground: a baked 128 x 128 torus (browser-games/living-world, tools/export_bin.mjs) read
from one float32 table, so the renderer and the walker evaluate the same surface. living_check.mojo is the gate.

Table: header [N, sea, ntrees, 0], then 5 floats per cell k = j N + i: height | rgb + ao bytes | horizon azimuths 0-3
bytes | 4-7 bytes | tree bytes (kind + 1, size x 100, x and y in the cell x 255). Units: one cell = one world unit; the
caller scales (render.mojo: LIVING_M metres per unit). Positions are unwrapped; the table index wraps (a torus).

Surface: each cell is two triangles split on its (0, 0)-(1, 1) diagonal, the web renderer's split, so lw_height is the
exact surface lw_hit draws. Trees: cones (pine two tiers, broadleaf up + down, glow-fern low), tested in the 3 x 3 cells
around each cell the ray crosses, only where the ray is low enough to touch them.
materials: 0 sky (miss), 1 ground, 2 water, 3 pine, 4 broadleaf, 5 glow-fern.
"""
from std.math import asin, atan2, cos, floor, pi, sin, sqrt
from std.memory import bitcast

comptime F4 = SIMD[DType.float32, 4]
comptime F8 = SIMD[DType.float32, 8]
comptime BIG = Float32(3.4e38)
comptime TS = Float32(1.6)  # tree height scale over the web's (a walker sees them from the ground)
comptime TOP = Float32(2.2) * TS * Float32(1.5)  # tallest tree above its cell's ground, the skip bound


@always_inline
def cell(T: Pointer[Float32, _], i: Int, j: Int, c: Int) -> Float32:
    var n = Int(T[0])
    return T[4 + 5 * (((j % n) + n) % n * n + ((i % n) + n) % n) + c]


@always_inline
def byte(v: Float32, b: Int) -> Float32:
    return Float32((bitcast[DType.uint32](v) >> UInt32(8 * b)) & 255)


@always_inline
def lw_height(T: Pointer[Float32, _], x: Float32, y: Float32) -> Float32:
    """Ground height at (x, y) on the cell's own triangle."""
    var i = Int(floor(x))
    var j = Int(floor(y))
    var fx = x - Float32(i)
    var fy = y - Float32(j)
    var h00 = cell(T, i, j, 0)
    var h11 = cell(T, i + 1, j + 1, 0)
    if fx >= fy:
        var h10 = cell(T, i + 1, j, 0)
        return h00 + (h10 - h00) * fx + (h11 - h10) * fy
    var h01 = cell(T, i, j + 1, 0)
    return h00 + (h11 - h01) * fx + (h01 - h00) * fy


@always_inline
def cone(o: F4, d: F4, c: F4, r: Float32, h: Float32) -> F4:
    """(t, normal) of the ray on the side of a cone, base circle (c, r) at c.z, apex c + (0, 0, h); h < 0 hangs down."""
    var ax = o[0] - c[0]
    var ay = o[1] - c[1]
    var az = o[2] - c[2] - h
    var k2 = (r / h) * (r / h)
    var A = d[0] * d[0] + d[1] * d[1] - k2 * d[2] * d[2]
    var B = 2 * (ax * d[0] + ay * d[1] - k2 * az * d[2])
    var C = ax * ax + ay * ay - k2 * az * az
    var disc = B * B - 4 * A * C
    if disc < 0 or A == 0:
        return F4(BIG, 0, 0, 0)
    var s = sqrt(disc)
    for k in range(2):
        var t = (-B - s) / (2 * A) if k == 0 else (-B + s) / (2 * A)
        if t > 1e-4:
            var qz = az + t * d[2]  # below the apex by -qz for h > 0
            var u = -qz / h
            if u >= 0 and u <= 1:
                var n = F4(ax + t * d[0], ay + t * d[1], -k2 * qz, 0)
                n = n / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
                if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0:
                    n = -n
                return F4(t, n[0], n[1], n[2])
    return F4(BIG, 0, 0, 0)


@always_inline
def tree(T: Pointer[Float32, _], i: Int, j: Int, o: F4, d: F4) -> F8:
    """The tree of cell (i, j) along the ray: (t, normal, material, glow, hue)."""
    var v = cell(T, i, j, 4)
    var kind = Int(byte(v, 0)) - 1
    if kind < 0:
        return F8(BIG, 0, 0, 0, 0, 0, 0, 0)
    var s = byte(v, 1) / 100
    var cx = Float32(i) + byte(v, 2) / 255
    var cy = Float32(j) + byte(v, 3) / 255
    var z = lw_height(T, cx, cy) - 0.1
    var best = F4(BIG, 0, 0, 0)
    for part in range(2):
        var h: F4
        if kind == 0:
            h = cone(o, d, F4(cx, cy, z + Float32(part) * 0.75 * s * TS, 0), 0.5 * s * (1 - 0.35 * Float32(part)), 1.3 * s * TS)
        elif kind == 1:
            h = cone(o, d, F4(cx, cy, z + 0.7 * s * TS, 0), 0.75 * s, (Float32(1.0) if part == 0 else Float32(-0.7)) * s * TS)
        else:
            if part == 1:
                break
            h = cone(o, d, F4(cx, cy, z, 0), 0.5 * s, 0.6 * s * TS)
        if h[0] < best[0]:
            best = h
    var hue = Float32(((i * 73856093) ^ (j * 19349663)) & 1023) / 1023
    return F8(best[0], best[1], best[2], best[3], Float32(3 + kind), Float32(1) if kind == 2 else Float32(0), hue, 0)


@always_inline
def ground(T: Pointer[Float32, _], i: Int, j: Int, o: F4, d: F4, t0: Float32, t1: Float32) -> F8:
    """The cell's two triangles along the ray between t0 and t1: (t, smooth normal, material)."""
    var h00 = cell(T, i, j, 0)
    var h10 = cell(T, i + 1, j, 0)
    var h01 = cell(T, i, j + 1, 0)
    var h11 = cell(T, i + 1, j + 1, 0)
    var ox = o[0] - Float32(i)
    var oy = o[1] - Float32(j)
    var best = BIG
    for tri in range(2):
        var b = (h10 - h00) if tri == 0 else (h11 - h01)
        var c = (h11 - h10) if tri == 0 else (h01 - h00)
        var den = d[2] - b * d[0] - c * d[1]
        if den == 0:
            continue
        var t = (h00 + b * ox + c * oy - o[2]) / den
        if t < t0 - 1e-4 or t > t1 + 1e-4 or t <= 0 or t >= best:
            continue
        var fx = ox + t * d[0]
        var fy = oy + t * d[1]
        if (tri == 0 and fx >= fy - 1e-4) or (tri == 1 and fy >= fx - 1e-4):
            best = t
    if best == BIG:
        return F8(BIG, 0, 0, 0, 0, 0, 0, 0)
    var fx = min(max(ox + best * d[0], Float32(0)), Float32(1))
    var fy = min(max(oy + best * d[1], Float32(0)), Float32(1))
    var gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy  # bilinear gradient: smooth shading over flat facets
    var gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
    var l = sqrt(gx * gx + gy * gy + 1)
    var water = o[2] + best * d[2] <= T[1] + 0.02
    return F8(best, -gx / l, -gy / l, 1 / l, Float32(2) if water else Float32(1), 0, 0, 0)


def lw_hit(T: Pointer[Float32, _], o: F4, d: F4, far: Float32) -> F8:
    """(t, normal, material, glow, hue) of the first surface along the ray (d unit) within far: a 2-D DDA over cells;
    per cell, the ground's triangles and the trees of the 3 x 3 cells around it, skipped where the ray is above both."""
    var cx = floor(o[0])
    var cy = floor(o[1])
    var sx = Float32(1) if d[0] > 0 else Float32(-1)
    var sy = Float32(1) if d[1] > 0 else Float32(-1)
    var mx = BIG
    var my = BIG
    var ex = BIG
    var ey = BIG
    if d[0] != 0:
        mx = (cx + Float32(1 if d[0] > 0 else 0) - o[0]) / d[0]
        ex = 1 / abs(d[0])
    if d[1] != 0:
        my = (cy + Float32(1 if d[1] > 0 else 0) - o[1]) / d[1]
        ey = 1 / abs(d[1])
    var enter = Float32(0)
    var best = F8(BIG, 0, 0, 0, 0, 0, 0, 0)
    for _ in range(512):
        if enter >= far or enter >= best[0]:
            break
        var leave = min(min(mx, my), far)
        var i = Int(cx)
        var j = Int(cy)
        var zlo = min(o[2] + enter * d[2], o[2] + leave * d[2])
        var top = max(max(cell(T, i, j, 0), cell(T, i + 1, j, 0)), max(cell(T, i, j + 1, 0), cell(T, i + 1, j + 1, 0)))
        if zlo <= top + TOP:  # low enough for a tree here; the ground only below its own top
            if zlo <= top:
                var g = ground(T, i, j, o, d, enter, leave)
                if g[0] < best[0]:
                    best = g
            for dj in range(-1, 2):
                for di in range(-1, 2):
                    var h = tree(T, i + di, j + dj, o, d)
                    if h[0] < best[0]:
                        best = h
        if mx < my:
            enter = mx
            mx += ex
            cx += sx
        else:
            enter = my
            my += ey
            cy += sy
    return best if best[0] <= far else F8(BIG, 0, 0, 0, 0, 0, 0, 0)


@always_inline
def lw_albedo(T: Pointer[Float32, _], x: Float32, y: Float32) -> F4:
    """Bilinear ground colour (display 0..1) and AO in the fourth lane."""
    var i = Int(floor(x))
    var j = Int(floor(y))
    var fx = x - Float32(i)
    var fy = y - Float32(j)
    var out = F4(0)
    for k in range(4):
        var v = cell(T, i + (k & 1), j + (k >> 1), 1)
        var w = (fx if k & 1 else 1 - fx) * (fy if k >> 1 else 1 - fy)
        out += F4(byte(v, 0), byte(v, 1), byte(v, 2), byte(v, 3)) * (w / 255)
    return out


@always_inline
def lw_lit(T: Pointer[Float32, _], x: Float32, y: Float32, sun_el: Float32, az8: Float32) -> Float32:
    """Sun visibility from the baked horizon angles (gl.mjs): bilinear over cells, azimuth az8 in eighths of a turn."""
    var a0 = Int(floor(az8)) & 7
    var a1 = (a0 + 1) & 7
    var fa = az8 - floor(az8)
    var i = Int(floor(x))  # angles live on the grid nodes, as the heights do
    var j = Int(floor(y))
    var fx = x - Float32(i)
    var fy = y - Float32(j)
    var hz = Float32(0)
    for k in range(4):
        var ci = i + (k & 1)
        var cj = j + (k >> 1)
        var b0 = byte(cell(T, ci, cj, 2 + (a0 >> 2)), a0 & 3)
        var b1 = byte(cell(T, ci, cj, 2 + (a1 >> 2)), a1 & 3)
        hz += (b0 * (1 - fa) + b1 * fa) * (fx if k & 1 else 1 - fx) * (fy if k >> 1 else 1 - fy)
    return min(max((sun_el - hz / 255 * 1.5707963) / 0.08 + 0.5, Float32(0)), Float32(1))


def sky_cam(mut c: List[Float32], t: Float32, m: Float32):
    """cam[16:32] for render.mojo LIVING=1 at time t (s; one day = 150 s, the web's sky(t)): metres per unit, light dir
    (sun, or the moon below the horizon), zenith and horizon colours, exposure, t, light elevation, azimuth in eighths,
    night, dusk."""
    var a = t / 150 * 2 * pi - pi / 2
    var sx = cos(a) * 0.6
    var sy = cos(a) * 0.4
    var sz = sin(a)
    var l = sqrt(sx * sx + sy * sy + sz * sz)
    sx /= l; sy /= l; sz /= l
    var day = min(max(sz * 4 + 0.35, Float32(0)), Float32(1))
    var dusk = min(max(1 - abs(sz) * 5, Float32(0)), Float32(1)) * min(day * 3, Float32(1))
    var g = Float32(1) if sz > -0.05 else Float32(-1)
    c[16] = m
    c[17] = g * sx; c[18] = g * sy; c[19] = g * sz
    c[20] = 0.01 + 0.17 * day; c[21] = 0.015 + 0.385 * day; c[22] = 0.05 + 0.8 * day
    c[23] = 0.04 + 0.68 * day + 0.5 * dusk; c[24] = 0.06 + 0.76 * day + 0.15 * dusk; c[25] = 0.12 + 0.83 * day - 0.2 * dusk
    c[26] = 1 + 0.6 * dusk + 0.6 * (1 - day)
    c[27] = t
    c[28] = asin(min(max(g * sz, Float32(-1)), Float32(1)))
    c[29] = (atan2(g * sy, g * sx) / (2 * pi) * 8 + 8) % 8
    c[30] = 1 - day
    c[31] = dusk
