"""Living World as a Kinestride ground: a baked torus (browser-games/living-world, tools/export_bin.mjs, docs/SCALE.md)
read from one float32 table, so the renderer and the walker evaluate the same surface. living_check.mojo is the gate.

Table: header [N, sea, C, TN]; 5 floats per terrain cell k = j N + i (height | rgb + ao bytes | horizon azimuths 0-3 |
4-7 | habitat); then a TN x TN grid of tree slots (species + 1, height / 40 m x 255, x and y in the slot x 255). Lengths are in
terrain cells of C metres (render.mojo cam[16] = C), so x, y and z share one factor. Positions are unwrapped; indices
wrap (a torus).

Surface: each terrain cell is two triangles split on its (0, 0)-(1, 1) diagonal, the web renderer's split, so lw_height
is the exact surface lw_hit draws. Within NEAR_M the ray walks the tree grid (5 m), testing the parent terrain cell's
triangles over each step and the cones of the 5 x 5 slots around it; beyond, it walks the terrain grid, ground only
(the forest is canopy colour there). materials: 0 sky, 1 ground, 2 water, 3 + species (spruce, beech, fern, birch,
alder, pine).
"""
from std.sys import get_defined_int
from std.math import asin, atan2, cos, floor, pi, sin, sqrt
from std.memory import bitcast

comptime F4 = SIMD[DType.float32, 4]
comptime F8 = SIMD[DType.float32, 8]
comptime BIG = Float32(3.4e38)
comptime NEAR_M = Float32(get_defined_int["LIVING_NEAR", 150]())  # trees are geometry within this range (SCALE.md section 2)
comptime HMAX_M = Float32(40)  # tallest tree, the slot height scale and the ray's skip bound
comptime HDR = 8  # world.bin header: N, sea, C, TN, max terrain height, then the offsets of cells (= HDR), slots, tail
comptime CROWN = SIMD[DType.float32, 8](0.085, 0.25, 0.6, 0.13, 0.2, 0.22, 0, 0)


@always_inline
def cell(T: Pointer[Float32, _], i: Int, j: Int, c: Int) -> Float32:
    var n = Int(T[0])
    return T[HDR + 5 * (((j % n) + n) % n * n + ((i % n) + n) % n) + c]


@always_inline
def slot(T: Pointer[Float32, _], i: Int, j: Int) -> Float32:
    var tn = Int(T[3])
    return T[Int(T[6]) + ((j % tn) + tn) % tn * tn + ((i % tn) + tn) % tn]


@always_inline
def cone_part(kind: Int, part: Int) -> F4:
    """Cone `part` of a tree of `kind` as (base, radius, height, exists) in fractions of the tree's height and crown
    radius: web/world.mjs TREE_CONES. 0 spruce one slender cone, 1 beech up and down cone over a bare trunk, 2 fern, 3 birch
    narrow and high, 4 alder up and down, 5 pine a flat umbrella high on the trunk."""
    if kind == 0:
        return F4(0.1, 0.85, 0.9, 1) if part == 0 else F4(0, 0, 0, 0)
    if kind == 1:
        return F4(0.35, 1, 0.65, 1) if part == 0 else F4(0.35, 1, -0.3, 1)
    if kind == 4:
        return F4(0.25, 1, 0.75, 1) if part == 0 else F4(0.25, 1, -0.2, 1)
    if kind == 5:
        return F4(0.65, 1, 0.2, 1) if part == 0 else F4(0.65, 1, -0.12, 1)
    var r = Float32(0.3) if kind == 3 else Float32(0)
    return F4(r, 1, 1 - r, 1) if part == 0 else F4(0, 0, 0, 0)


@always_inline
def tree_at(T: Pointer[Float32, _], ti: Int, tj: Int) -> F8:
    """Slot (ti, tj): (centre x, y, ground z, height, crown radius, kind) in terrain cells; kind -1 when empty."""
    var v = slot(T, ti, tj)
    var kind = Int(byte(v, 0)) - 1
    if kind < 0:
        return F8(0, 0, 0, 0, 0, -1, 0, 0)
    var tc = T[0] / T[3]
    var x = (Float32(ti) + byte(v, 2) / 255) * tc
    var y = (Float32(tj) + byte(v, 3) / 255) * tc
    var hh = byte(v, 1) / 255 * HMAX_M / T[2]
    var r = hh * CROWN[kind]  # crown radius / height, web/world.mjs SPECIES
    return F8(x, y, lw_height(T, x, y), hh, r, Float32(kind), 0, 0)


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
    """(t, normal) of the ray on a solid cone (side and base disc), base circle (c, r) at c.z, apex c + (0, 0, h); h < 0
    hangs down."""
    var ax = o[0] - c[0]
    var ay = o[1] - c[1]
    var az = o[2] - c[2] - h
    var k2 = (r / h) * (r / h)
    var A = d[0] * d[0] + d[1] * d[1] - k2 * d[2] * d[2]
    var B = 2 * (ax * d[0] + ay * d[1] - k2 * az * d[2])
    var C = ax * ax + ay * ay - k2 * az * az
    var best = F4(BIG, 0, 0, 0)
    if d[2] != 0:  # the base disc: on a slope the eye can sit below it
        var tc = (c[2] - o[2]) / d[2]
        var bx = ax + tc * d[0]
        var by = ay + tc * d[1]
        if tc > 1e-4 and bx * bx + by * by <= r * r:
            best = F4(tc, 0, 0, Float32(-1) if h > 0 else Float32(1))
    var disc = B * B - 4 * A * C
    if disc < 0 or A == 0:
        return best
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
                return F4(t, n[0], n[1], n[2]) if t < best[0] else best
    return best


@always_inline
def ellipsoid(o: F4, d: F4, c: F4, r: Float32, rz: Float32) -> F4:
    """(t, normal) of the ray on an ellipsoid crown: centre c, horizontal radius r, vertical radius rz."""
    var k = r / rz  # squash z so the crown is a sphere of radius r
    var ox = o[0] - c[0]
    var oy = o[1] - c[1]
    var oz = (o[2] - c[2]) * k
    var dz = d[2] * k
    var A = d[0] * d[0] + d[1] * d[1] + dz * dz
    var B = 2 * (ox * d[0] + oy * d[1] + oz * dz)
    var C = ox * ox + oy * oy + oz * oz - r * r
    var disc = B * B - 4 * A * C
    if disc < 0:
        return F4(BIG, 0, 0, 0)
    var s = sqrt(disc)
    var t = (-B - s) / (2 * A)
    if t <= 1e-4:
        t = (-B + s) / (2 * A)
        if t <= 1e-4:
            return F4(BIG, 0, 0, 0)
    var n = F4(ox + t * d[0], oy + t * d[1], (oz + t * dz) * k, 0)
    n = n / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    return F4(t, n[0], n[1], n[2])


@always_inline
def tree(T: Pointer[Float32, _], ti: Int, tj: Int, o: F4, d: F4) -> F8:
    """The tree of slot (ti, tj) along the ray: (t, normal, material, glow, hue)."""
    var tr = tree_at(T, ti, tj)
    var kind = Int(tr[5])
    if kind < 0:
        return F8(BIG, 0, 0, 0, 0, 0, 0, 0)
    var best = F4(BIG, 0, 0, 0)
    if kind == 1 or kind == 3 or kind == 4 or kind == 5:  # broadleaf and pine crowns: ellipsoids (centre, vertical radius) x H
        var cz = Float32(0.56) if kind == 1 else (Float32(0.62) if kind == 3 else (Float32(0.55) if kind == 4 else Float32(0.82)))
        var rz = Float32(0.42) if kind == 1 else (Float32(0.38) if kind == 3 else (Float32(0.45) if kind == 4 else Float32(0.14)))
        best = ellipsoid(o, d, F4(tr[0], tr[1], tr[2] + cz * tr[3], 0), tr[4] * (Float32(1) if kind == 5 else Float32(1.4)), rz * tr[3])
    for part in range(0 if kind == 1 or kind == 3 or kind == 4 or kind == 5 else 2):
        var c = cone_part(kind, part)
        if c[3] == 0:
            break
        var h = cone(o, d, F4(tr[0], tr[1], tr[2] + c[0] * tr[3], 0), c[1] * tr[4], c[2] * tr[3])
        if h[0] < best[0]:
            best = h
    var hue = Float32(((ti * 73856093) ^ (tj * 19349663)) & 1023) / 1023
    if kind != 2:  # the trunk: a thin cone to 0.9 H, material 9, species in the last lane (bark colour)
        var tk = cone(o, d, F4(tr[0], tr[1], tr[2], 0), 0.012 * tr[3], 0.9 * tr[3])
        if tk[0] < best[0]:
            return F8(tk[0], tk[1], tk[2], tk[3], 9, 0, hue, Float32(kind))
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


def march(T: Pointer[Float32, _], o: F4, d: F4, t0: Float32, t1: Float32, cs: Float32, trees: Bool) -> F8:
    """2-D DDA over cells of size cs (terrain cells) between ray parameters t0 and t1: each step tests the parent terrain
    cell's triangles over the step and, with trees, the cones of the 5 x 5 tree slots; skipped where the ray is above
    both. One body for the near (tree grid) and far (terrain grid) phases."""
    var p = o + d * t0
    var cx = floor(p[0] / cs)
    var cy = floor(p[1] / cs)
    var sx = Float32(1) if d[0] > 0 else Float32(-1)
    var sy = Float32(1) if d[1] > 0 else Float32(-1)
    var mx = BIG
    var my = BIG
    var ex = BIG
    var ey = BIG
    if d[0] != 0:
        mx = t0 + ((cx + Float32(1 if d[0] > 0 else 0)) * cs - p[0]) / d[0]
        ex = cs / abs(d[0])
    if d[1] != 0:
        my = t0 + ((cy + Float32(1 if d[1] > 0 else 0)) * cs - p[1]) / d[1]
        ey = cs / abs(d[1])
    var enter = t0
    var best = F8(BIG, 0, 0, 0, 0, 0, 0, 0)
    var hmax = HMAX_M / T[2]
    var full = True
    var stepx = True
    for _ in range(1024):
        if enter >= t1 or enter >= best[0] or (d[2] >= 0 and o[2] + enter * d[2] > T[4] + hmax):
            break  # done, hit, or rising above the highest ground and tree
        var leave = min(min(mx, my), t1)
        var mid = o + d * (0.5 * (enter + leave))
        var i = Int(floor(mid[0]))
        var j = Int(floor(mid[1]))
        var zlo = min(o[2] + enter * d[2], o[2] + leave * d[2])
        var top = max(max(cell(T, i, j, 0), cell(T, i + 1, j, 0)), max(cell(T, i, j + 1, 0), cell(T, i + 1, j + 1, 0)))
        if zlo <= top:
            var g = ground(T, i, j, o, d, enter, leave)
            if g[0] < best[0]:
                best = g
        var ttop = top
        if trees:  # the ground under any of the 5 x 5 slots: the terrain nodes their block spans
            for nj in range(Int(floor((cy - 2) * cs)), Int(floor((cy + 3) * cs)) + 2):
                for ni in range(Int(floor((cx - 2) * cs)), Int(floor((cx + 3) * cs)) + 2):
                    ttop = max(ttop, cell(T, ni, nj, 0))
        if trees and zlo <= ttop + hmax:  # 5 x 5 slots: a beech crown (up to 9 m radius) reaches two 5 m slots away
            for k in range(25 if full else 5):  # the whole window once, then only the column or row a step brings in
                var di = k % 5 - 2 if full else (2 * Int(sx) if stepx else k - 2)
                var dj = k // 5 - 2 if full else (k - 2 if stepx else 2 * Int(sy))
                var h = tree(T, Int(cx) + di, Int(cy) + dj, o, d)
                if h[0] < best[0] and h[0] <= t1:  # a crown reached from inside the phase may lie past its end
                    best = h
            full = False
        elif trees:
            full = True  # skipped slots: test the whole window at the next low step
        stepx = mx < my
        if stepx:
            enter = mx
            mx += ex
            cx += sx
        else:
            enter = my
            my += ey
            cy += sy
    return best


def lw_hit(T: Pointer[Float32, _], o: F4, d: F4, far: Float32) -> F8:
    """(t, normal, material, glow, hue) of the first surface along the ray (d unit) within far: trees and ground on the
    5 m tree grid out to NEAR_M, then ground alone on the terrain grid."""
    var near = min(NEAR_M / T[2], far)
    var h = march(T, o, d, 0, near, T[0] / T[3], True)
    if h[0] == BIG and far > near:
        h = march(T, o, d, near, far, 1, False)
    return h if h[0] <= far else F8(BIG, 0, 0, 0, 0, 0, 0, 0)


def lw_tree_near(T: Pointer[Float32, _], x: Float32, y: Float32, pad: Float32) -> Bool:
    """(x, y) within pad of a tree's trunk zone, in terrain cells (spruce and fern: the crown circle; beech: 0.3 of it,
    its crown is high): a walker or a camera target there would stand inside the tree."""
    var tn = T[3] / T[0]
    var ci = Int(floor(x * tn))
    var cj = Int(floor(y * tn))
    for dj in range(-2, 3):
        for di in range(-2, 3):
            var tr = tree_at(T, ci + di, cj + dj)
            if tr[5] < 0:
                continue
            var r = tr[4] * (Float32(0.3) if tr[5] == 1 or tr[5] >= 3 else Float32(1)) + pad  # high crowns: the trunk zone
            if (x - tr[0]) * (x - tr[0]) + (y - tr[1]) * (y - tr[1]) < r * r:
                return True
    return False


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
