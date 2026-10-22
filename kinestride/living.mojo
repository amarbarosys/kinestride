"""Living World as a Kinestride ground: a baked torus (browser-games/living-world, tools/export_bin.mjs, docs/SCALE.md)
read from one float32 table, so the renderer and the walker evaluate the same surface. living_check.mojo is the gate.

Table: header [N, sea, C, TN]; 5 floats per terrain cell k = j N + i (height | rgb + ao bytes | horizon azimuths 0-3 |
4-7 | habitat); then a TN x TN grid of tree slots (species + 1, height / 40 m x 255, x and y in the slot x 255). Lengths are in
terrain cells of C metres (render.mojo cam[16] = C), so x, y and z share one factor. Positions are unwrapped; indices
wrap (a torus) with a mask: N and TN are powers of two (an integer modulo by a runtime value took 2.3 ms of the frame).

The file ends with a tail (header word 7 = its offset, tools/export_bin.mjs): the trails, P polylines (word 0 = P, then each one's
vertex count, then the vertices, one word each: x | y << 16 as uint16 at N / 65536 cells, wrapping). The habitat word holds the trail
level in its fraction (habitat + level / 256), and the ground colour is without the trail tint.

After the file (lw_ctrl sets header word 7 to the file's length) the host appends the surface's control grid (lw_ctrl, never in the file): the
highest control, then N x N controls, then N x N cell bounds (the maximum of each cell's 16 controls), then the file's tail offset, then
the trail index: N x N + 1 offsets into a list of segment ids (the vertex a segment starts at) per terrain cell, for the segments within 2 m of the cell. The macro surface is the uniform cubic B-spline over them (C1 across cells, so the 20 m
facets and stair cliffs are gone); the controls are the height nodes deconvolved twice on dry nodes (the surface keeps within
a few metres of the nodes) and pushed down on wet nodes until the surface there is at or under the sea, so rivers and lakes stay
water. The spline is a convex combination of its 16 controls, so their maximum bounds a cell's ground exactly (and the cell bound is one load).

Surface: each terrain cell is one bicubic patch of its 4 x 4 controls, so lw_height is the exact surface lw_hit draws. Within NEAR_M the ray walks the tree grid (5 m), testing the parent terrain cell's
triangles over each step and the cones of the 5 x 5 slots around it; beyond, it walks the terrain grid, ground only
(the forest is canopy colour there). materials: 0 sky, 1 ground, 2 water, 3 + species (spruce, beech, fern, birch,
alder, pine).
"""
from std.sys import get_defined_int
from std.math import asin, atan2, cos, exp, floor, pi, sin, sqrt
from std.memory import bitcast

comptime F4 = SIMD[DType.float32, 4]
comptime F8 = SIMD[DType.float32, 8]
comptime BIG = Float32(3.4e38)
comptime NEAR_M = Float32(get_defined_int["LIVING_NEAR", 150]())  # trees are geometry within this range (SCALE.md section 2)
comptime CLUMP_M = Float32(get_defined_int["LW_CLUMP", 80]())  # crowns are leaf clumps and tiers within this range of the eye, one envelope beyond
comptime HMAX_M = Float32(40)  # tallest tree, the slot height scale and the ray's skip bound
comptime HDR = 8  # world.bin header: N, sea, C, TN, max terrain height, then the offsets of cells (= HDR), slots, tail
comptime CROWN = SIMD[DType.float32, 8](0.085, 0.25, 0.6, 0.13, 0.2, 0.22, 0, 0)


@always_inline
def cell(T: Pointer[Float32, _], i: Int, j: Int, c: Int) -> Float32:
    var n = Int(T[0])
    return T[HDR + 5 * ((j & (n - 1)) * n + (i & (n - 1))) + c]  # the mask wraps negatives too


@always_inline
def slot(T: Pointer[Float32, _], i: Int, j: Int) -> Float32:
    var tn = Int(T[3])
    return T[Int(T[6]) + (j & (tn - 1)) * tn + (i & (tn - 1))]


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


comptime F16 = SIMD[DType.float32, 16]


@always_inline
def ctl(T: Pointer[Float32, _], i: Int, j: Int) -> Float32:
    var n = Int(T[0])
    return T[Int(T[7]) + 1 + (j & (n - 1)) * n + (i & (n - 1))]


@always_inline
def cap(T: Pointer[Float32, _], i: Int, j: Int) -> Float32:
    """The upper bound of cell (i, j)'s ground: the maximum of its 16 controls."""
    var n = Int(T[0])
    return T[Int(T[7]) + 1 + n * n + (j & (n - 1)) * n + (i & (n - 1))]


@always_inline
def patch(T: Pointer[Float32, _], i: Int, j: Int) -> F16:
    """The 4 x 4 controls of cell (i, j): lane 4 b + a is control (i - 1 + a, j - 1 + b)."""
    var n = Int(T[0])
    var base = Int(T[7]) + 1
    var c = F16(0)
    for b in range(4):
        var row = base + ((j - 1 + b) & (n - 1)) * n
        for a in range(4):
            c[4 * b + a] = T[row + ((i - 1 + a) & (n - 1))]
    return c


@always_inline
def surf(c: F16, u: Float32, v: Float32) -> F4:
    """(height, d/du, d/dv) of the cell's cubic B-spline patch at (u, v) in 0..1."""
    var u2 = u * u
    var u3 = u2 * u
    var v2 = v * v
    var v3 = v2 * v
    var wu = F4((1 - u) * (1 - u) * (1 - u), 3 * u3 - 6 * u2 + 4, -3 * u3 + 3 * u2 + 3 * u + 1, u3) / 6
    var du = F4(-(1 - u) * (1 - u), 3 * u2 - 4 * u, -3 * u2 + 2 * u + 1, u2) / 2
    var wv = F4((1 - v) * (1 - v) * (1 - v), 3 * v3 - 6 * v2 + 4, -3 * v3 + 3 * v2 + 3 * v + 1, v3) / 6
    var dv = F4(-(1 - v) * (1 - v), 3 * v2 - 4 * v, -3 * v2 + 2 * v + 1, v2) / 2
    var h = Float32(0)
    var hu = Float32(0)
    var hv = Float32(0)
    for b in range(4):
        var r = c[4 * b] * wu[0] + c[4 * b + 1] * wu[1] + c[4 * b + 2] * wu[2] + c[4 * b + 3] * wu[3]
        var dr = c[4 * b] * du[0] + c[4 * b + 1] * du[1] + c[4 * b + 2] * du[2] + c[4 * b + 3] * du[3]
        h += wv[b] * r
        hu += wv[b] * dr
        hv += dv[b] * r
    return F4(h, hu, hv, 0)


@always_inline
def lw_height(T: Pointer[Float32, _], x: Float32, y: Float32) -> Float32:
    """Ground height at (x, y): the cell's bicubic patch."""
    var i = Int(floor(x))
    var j = Int(floor(y))
    return surf(patch(T, i, j), x - Float32(i), y - Float32(j))[0]


@always_inline
def wrapd(d: Float32, n: Float32) -> Float32:
    return d - n * floor(d / n + 0.5)  # a difference taken the short way round the torus


@always_inline
def trail_d(T: Pointer[Float32, _], x: Float32, y: Float32) -> Float32:
    """Distance in terrain cells from (x, y) to the nearest trail segment indexed for its cell (2 m or less away), 1 when none."""
    var n = Int(T[0])
    var base = Int(T[7]) + 1 + 2 * n * n  # the file's tail offset, then the index
    var tail = Int(T[base])
    var k = (Int(floor(y)) & (n - 1)) * n + (Int(floor(x)) & (n - 1))
    var vs = tail + 1 + Int(T[tail])
    var best = Float32(1)
    for q in range(Int(T[base + 1 + k]), Int(T[base + 2 + k])):
        var v = Int(T[base + 2 + n * n + q])
        var w0 = bitcast[DType.uint32](T[vs + v])
        var w1 = bitcast[DType.uint32](T[vs + v + 1])
        var x0 = Float32(w0 & 65535) / 65536 * Float32(n)
        var y0 = Float32(w0 >> 16) / 65536 * Float32(n)
        var ex = wrapd(Float32(w1 & 65535) / 65536 * Float32(n) - x0, Float32(n))
        var ey = wrapd(Float32(w1 >> 16) / 65536 * Float32(n) - y0, Float32(n))
        var px = wrapd(x - x0, Float32(n))
        var py = wrapd(y - y0, Float32(n))
        var t = min(max((px * ex + py * ey) / max(ex * ex + ey * ey, Float32(1e-9)), Float32(0)), Float32(1))
        best = min(best, sqrt((px - t * ex) * (px - t * ex) + (py - t * ey) * (py - t * ey)))
    return best


@always_inline
def lw_grass(T: Pointer[Float32, _], x: Float32, y: Float32) -> F4:
    """(meadow, heather) share of the four nodes around (x, y), bilinear: the grass fades across a habitat edge instead of stopping on it."""
    var i = Int(floor(x))
    var j = Int(floor(y))
    var fx = x - Float32(i)
    var fy = y - Float32(j)
    var out = F4(0)
    for k in range(4):
        var hb = Int(cell(T, i + (k & 1), j + (k >> 1), 4))
        var w = (fx if k & 1 else 1 - fx) * (fy if k >> 1 else 1 - fy)
        out += F4(w if hb == 2 else 0, w if hb == 4 else 0, 0, 0)
    return out


@always_inline
def trail_xy(T: Pointer[Float32, _], v: Int) -> F4:
    """Vertex v (counted over all polylines in the tail) as (x, y, polylines, vertices) in terrain cells."""
    var n = Int(T[0])
    var tail = Int(T[Int(T[7]) + 1 + 2 * n * n])
    var np = Int(T[tail])
    var nv = 0
    for p in range(np):
        nv += Int(T[tail + 1 + p])
    var w = bitcast[DType.uint32](T[tail + 1 + np + min(max(v, 0), nv - 1)])
    return F4(Float32(w & 65535) / 65536 * Float32(n), Float32(w >> 16) / 65536 * Float32(n), Float32(np), Float32(nv))


@always_inline
def lw_trail(T: Pointer[Float32, _], x: Float32, y: Float32) -> Float32:
    """The baked trail level 0..1 (0.85 of it is the far tint), bilinear over the nodes: the fraction of the habitat word."""
    var i = Int(floor(x))
    var j = Int(floor(y))
    var fx = x - Float32(i)
    var fy = y - Float32(j)
    var out = Float32(0)
    for k in range(4):
        var v = cell(T, i + (k & 1), j + (k >> 1), 4)
        out += (v - floor(v)) * (fx if k & 1 else 1 - fx) * (fy if k >> 1 else 1 - fy)
    return out * (256 / 255)


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
def rnd(x: Int, k: Int) -> Float32:
    """A hash in 0..1 of x and the stream k."""
    return Float32(((x * (2654435761 + k * 40503) + k * 97) >> 9) & 0xFFFF) / 65535


@always_inline
def crown(o: F4, d: F4, tr: F8, kind: Int, cz: Float32, rz: Float32, h0: Int, env: F4) -> F4:
    """The leaf-clump crown of a tree whose envelope the ray hit (env): a broadleaf is a core and 5-8 clumps on its shell, a spruce 3-5 stacked
    skirts, a pine 3-4 stacked discs, all inside the envelope; a per-tree hash places them. The closest hit, or a miss through the gaps."""
    var best = F4(BIG, 0, 0, 0)
    var R = tr[4] * (Float32(1) if kind == 5 else Float32(1.4))
    var RZ = rz * tr[3]
    if kind == 0:
        var n = 3 + Int(rnd(h0, 0) * 2.99)
        var rb = 0.85 * tr[4]
        for k in range(5):
            if k >= n:
                break
            var f = Float32(k) / Float32(n)
            var rr = rb * (1 - 0.75 * f) * (0.94 + 0.12 * rnd(h0, 1 + k))
            var tier = cone(o, d, F4(tr[0], tr[1], tr[2] + (0.1 + 0.675 * f) * tr[3], 0), rr, (0.12 + 0.55 * (1 - f) * 0.5 + 0.1 * Float32(n - 3)) * tr[3] * 0.9)
            if tier[0] < best[0]:
                best = tier
        return best
    if kind == 5:
        var n = 3 + Int(rnd(h0, 0) * 1.99)
        for k in range(4):
            if k >= n:
                break
            var disc = ellipsoid(o, d, F4(tr[0], tr[1], tr[2] + (0.72 + 0.08 * Float32(k)) * tr[3], 0), R * (0.95 - 0.1 * Float32(k)) * (0.94 + 0.12 * rnd(h0, 1 + k)), 0.075 * tr[3])
            if disc[0] < best[0]:
                best = disc
        return best
    var c = F4(tr[0], tr[1], tr[2] + cz * tr[3], 0)
    best = ellipsoid(o, d, c, 0.72 * R, 0.72 * RZ)
    var n = 5 + Int(rnd(h0, 0) * 3.99)
    for k in range(8):
        if k >= n:
            break
        var th = (Float32(k) + 0.8 * rnd(h0, 1 + 3 * k)) / Float32(n) * 6.2831853
        var zz = -0.25 + 1.15 * rnd(h0, 2 + 3 * k)
        var rr = sqrt(1 - zz * zz)
        var sz = 0.30 + 0.10 * rnd(h0, 3 + 3 * k)
        var h = ellipsoid(o, d, F4(c[0] + 0.58 * R * rr * cos(th), c[1] + 0.58 * R * rr * sin(th), c[2] + 0.58 * RZ * zz, 0), sz * R, sz * RZ)
        if h[0] < best[0]:
            best = h
    return best


@always_inline
def prop_at(T: Pointer[Float32, _], ti: Int, tj: Int) -> F8:
    """The ground prop of tree-grid cell (ti, tj), hashed (nothing stored): (x, y, z, half sizes x y z, yaw, kind) in terrain cells, kind -1 when none. Kind 0
    boulder (scree, riparian), 1 bush (meadow, heather), 2 fallen log (spruce, beech); 0.3-2 m, in no tree's slot and not within 0.5 m of the water."""
    var no = F8(0, 0, 0, 0, 0, 0, 0, -1)
    var h = (ti * 19349663) ^ (tj * 83492791)
    var hb = Int(cell(T, (ti + 2) >> 2, (tj + 2) >> 2, 4))  # the habitat of the node nearest the cell's middle, as the ground shows it (tree grid = 4 x the terrain's)
    var p = Float32(0.3) if hb == 0 else (Float32(0.18) if hb == 1 else (Float32(0.04) if hb == 2 else (Float32(0.1) if hb == 3 else (Float32(0.22) if hb == 4 else Float32(0.1)))))
    if rnd(h, 0) >= p or Int(byte(slot(T, ti, tj), 0)) != 0:
        return no
    var kind = 0 if hb < 2 else (1 if hb == 2 or hb == 4 else 2)
    var u = rnd(h, 1)
    var m = (0.3 + 1.7 * u * u * u) if kind == 0 else ((0.35 + 0.65 * u) if kind == 1 else (0.8 + 0.7 * u))  # half length / radius, m
    var s = m / T[2]
    var tc = T[0] / T[3]
    var g = min(max(m / 5, Float32(0)), Float32(0.45))
    var x = (Float32(ti) + g + (1 - 2 * g) * rnd(h, 2)) * tc
    var y = (Float32(tj) + g + (1 - 2 * g) * rnd(h, 3)) * tc
    var z = lw_height(T, x, y)
    if z < T[1] + 0.5 / T[2] or trail_d(T, x, y) < (0.9 + m) / T[2]:  # water, or a trail's ribbon (0.6 m half width and its soft edge)
        return no
    var sy = s * (0.8 if kind == 0 else (1.0 if kind == 1 else 0.2 / m))
    var sz = s * (0.6 if kind == 0 else (0.7 if kind == 1 else 0.2 / m))
    return F8(x, y, z + 0.35 * sz, s, sy, sz, rnd(h, 4) * 6.2831853, Float32(kind))


@always_inline
def blob(o: F4, d: F4, c: F4, sx: Float32, sy: Float32, sz: Float32, yaw: Float32) -> F4:
    """(t, normal) of the ray on an ellipsoid of half sizes sx, sy, sz about c, turned by yaw about z."""
    var cs = cos(yaw)
    var sn = sin(yaw)
    var px = o[0] - c[0]
    var py = o[1] - c[1]
    var lx = (px * cs + py * sn) / sx
    var ly = (py * cs - px * sn) / sy
    var lz = (o[2] - c[2]) / sz
    var dx = (d[0] * cs + d[1] * sn) / sx
    var dy = (d[1] * cs - d[0] * sn) / sy
    var dz = d[2] / sz
    var A = dx * dx + dy * dy + dz * dz
    var B = 2 * (lx * dx + ly * dy + lz * dz)
    var disc = B * B - 4 * A * (lx * lx + ly * ly + lz * lz - 1)
    if disc < 0:
        return F4(BIG, 0, 0, 0)
    var t = (-B - sqrt(disc)) / (2 * A)
    if t <= 1e-4:
        return F4(BIG, 0, 0, 0)
    var nx = (lx + t * dx) / sx
    var ny = (ly + t * dy) / sy
    var n = F4(nx * cs - ny * sn, nx * sn + ny * cs, (lz + t * dz) / sz, 0)
    n = n / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    return F4(t, n[0], n[1], n[2])


@always_inline
def prop(T: Pointer[Float32, _], ti: Int, tj: Int, o: F4, d: F4) -> F8:
    """The prop of cell (ti, tj) along the ray: (t, normal, material 10 + kind, 0, hue, 0); a log is the long axis of the same blob."""
    var p = prop_at(T, ti, tj)
    var kind = Int(p[7])
    if kind < 0:
        return F8(BIG, 0, 0, 0, 0, 0, 0, 0)
    var b = blob(o, d, F4(p[0], p[1], p[2], 0), p[3], p[4], p[5], p[6])
    return F8(b[0], b[1], b[2], b[3], Float32(10 + kind), 0, rnd((ti * 19349663) ^ (tj * 83492791), 7), 0)


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
        if best[0] < BIG and (tr[0] - o[0]) * (tr[0] - o[0]) + (tr[1] - o[1]) * (tr[1] - o[1]) < (CLUMP_M / T[2]) * (CLUMP_M / T[2]):
            best = crown(o, d, tr, kind, cz, rz, (ti * 73856093) ^ (tj * 19349663), best)
    for part in range(0 if kind == 1 or kind == 3 or kind == 4 or kind == 5 else 2):
        var c = cone_part(kind, part)
        if c[3] == 0:
            break
        var h = cone(o, d, F4(tr[0], tr[1], tr[2] + c[0] * tr[3], 0), c[1] * tr[4], c[2] * tr[3])
        if kind == 0 and h[0] < BIG and (tr[0] - o[0]) * (tr[0] - o[0]) + (tr[1] - o[1]) * (tr[1] - o[1]) < (CLUMP_M / T[2]) * (CLUMP_M / T[2]):
            h = crown(o, d, tr, kind, 0, 0, (ti * 73856093) ^ (tj * 19349663), h)
        if h[0] < best[0]:
            best = h
    var hue = Float32(((ti * 73856093) ^ (tj * 19349663)) & 1023) / 1023
    if kind != 2:  # the trunk: a thin cone to 0.9 H, material 9, species in the last lane (bark colour)
        var tk = cone(o, d, F4(tr[0], tr[1], tr[2], 0), 0.012 * tr[3], 0.9 * tr[3])
        if tk[0] < best[0]:
            return F8(tk[0], tk[1], tk[2], tk[3], 9, 0, hue, Float32(kind))
    return F8(best[0], best[1], best[2], best[3], Float32(3 + kind), Float32(1) if kind == 2 else Float32(0), hue, 0)


@always_inline
def ground(T: Pointer[Float32, _], c: F16, i: Int, j: Int, o: F4, d: F4, t0: Float32, t1: Float32) -> F8:
    """The cell's patch c along the ray between t0 and t1: (t, analytic normal, material). The first sign change of the ray's height over
    the patch on 4 samples, then 8 steps of false position."""
    var ox = o[0] - Float32(i)
    var oy = o[1] - Float32(j)
    var ta = t0
    var fa = o[2] + ta * d[2] - surf(c, min(max(ox + ta * d[0], Float32(0)), Float32(1)), min(max(oy + ta * d[1], Float32(0)), Float32(1)))[0]
    for k in range(1, 5):
        var tb = t0 + (t1 - t0) * Float32(k) * 0.25
        var fb = o[2] + tb * d[2] - surf(c, min(max(ox + tb * d[0], Float32(0)), Float32(1)), min(max(oy + tb * d[1], Float32(0)), Float32(1)))[0]
        if fa * fb <= 0 and tb > 0:
            var a = ta
            var b = tb
            var side = 0
            for _ in range(8):
                var t = (a * fb - b * fa) / (fb - fa) if fb != fa else 0.5 * (a + b)
                var ft = o[2] + t * d[2] - surf(c, min(max(ox + t * d[0], Float32(0)), Float32(1)), min(max(oy + t * d[1], Float32(0)), Float32(1)))[0]
                if ft * fb > 0:
                    b = t
                    fb = ft
                    if side == -1:
                        fa *= 0.5
                    side = -1
                else:
                    a = t
                    fa = ft
                    if side == 1:
                        fb *= 0.5
                    side = 1
            var th = (a * fb - b * fa) / (fb - fa) if fb != fa else 0.5 * (a + b)
            var q = surf(c, min(max(ox + th * d[0], Float32(0)), Float32(1)), min(max(oy + th * d[1], Float32(0)), Float32(1)))
            var l = sqrt(q[1] * q[1] + q[2] * q[2] + 1)
            var water = o[2] + th * d[2] <= T[1] + 0.02
            return F8(th, -q[1] / l, -q[2] / l, 1 / l, Float32(2) if water else Float32(1), 0, 0, 0)
        ta = tb
        fa = fb
    return F8(BIG, 0, 0, 0, 0, 0, 0, 0)


def lw_ctrl(world: List[Float32]) -> List[Float32]:
    """world.bin's floats plus the surface's control grid (built here, once, on the host): the highest control, then N x N controls."""
    var tab = world.copy()
    var T = world.unsafe_ptr()
    var n = Int(T[0])
    var sea = T[1]
    var tail = Int(T[7])
    tab[7] = Float32(len(world))  # the control grid starts where the file ends
    var z = List[Float32](length=n * n, fill=0)
    for j in range(n):
        for i in range(n):
            z[j * n + i] = cell(T, i, j, 0)
    var c = z.copy()
    var s = z.copy()
    for it in range(42):
        for j in range(n):  # s = the spline's value at the nodes: [1, 4, 1] / 6 along x, then along y
            for i in range(n):
                s[j * n + i] = (c[j * n + ((i - 1) & (n - 1))] + 4 * c[j * n + i] + c[j * n + ((i + 1) & (n - 1))]) / 6
        var t = s.copy()
        for j in range(n):
            for i in range(n):
                s[j * n + i] = (t[((j - 1) & (n - 1)) * n + i] + 4 * t[j * n + i] + t[((j + 1) & (n - 1)) * n + i]) / 6
        var worst = Float32(0)
        for k in range(n * n):
            if z[k] > sea + 1e-6:
                if it < 2:  # dry nodes: deconvolve twice, the surface keeps near the nodes
                    c[k] += z[k] - s[k]
            else:
                var ex = max(s[k] - sea, Float32(0))  # wet nodes: down until the surface there is at the sea
                worst = max(worst, ex)
                c[k] -= 1.4 * ex
        if it >= 2 and worst < 0.0025:
            break
    var hi = Float32(-1e30)
    for v in c:
        hi = max(hi, v)
    tab.append(hi)
    for v in c:
        tab.append(v)
    for j in range(n):  # the cell bounds
        for i in range(n):
            var m = Float32(-1e30)
            for b in range(4):
                for a in range(4):
                    m = max(m, c[((j - 1 + b) & (n - 1)) * n + ((i - 1 + a) & (n - 1))])
            tab.append(m)
    tab.append(Float32(tail))
    var np = Int(T[tail])  # the trail index: per cell, the segments within 2 m (0.1 cell) of it
    var vs = tail + 1 + np
    var segs = List[Int]()
    var voff = 0
    for p in range(np):
        for v in range(voff, voff + Int(T[tail + 1 + p]) - 1):
            segs.append(v)
        voff += Int(T[tail + 1 + p])
    var own = List[List[Int]]()
    for _ in range(n * n):
        own.append(List[Int]())
    for v in segs:
        var w0 = bitcast[DType.uint32](T[vs + v])
        var w1 = bitcast[DType.uint32](T[vs + v + 1])
        var x0 = Float32(w0 & 65535) / 65536 * Float32(n)
        var y0 = Float32(w0 >> 16) / 65536 * Float32(n)
        var x1 = x0 + wrapd(Float32(w1 & 65535) / 65536 * Float32(n) - x0, Float32(n))
        var y1 = y0 + wrapd(Float32(w1 >> 16) / 65536 * Float32(n) - y0, Float32(n))
        for cj in range(Int(floor(min(y0, y1) - 0.1)), Int(floor(max(y0, y1) + 0.1)) + 1):
            for ci in range(Int(floor(min(x0, x1) - 0.1)), Int(floor(max(x0, x1) + 0.1)) + 1):
                own[(cj & (n - 1)) * n + (ci & (n - 1))].append(v)
    var run = 0
    for k in range(n * n):
        tab.append(Float32(run))
        run += len(own[k])
    tab.append(Float32(run))
    for k in range(n * n):
        for v in own[k]:
            tab.append(Float32(v))
    return tab^


def march(T: Pointer[Float32, _], o: F4, d: F4, t0: Float32, t1: Float32, cs: Float32, trees: Bool) -> F8:
    """2-D DDA over cells of size cs (terrain cells) between ray parameters t0 and t1: each step tests the parent terrain
    cell's patch over the step and, with trees, the cones of the 5 x 5 tree slots; skipped where the ray is above
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
    var pi = -1000000  # the cell whose patch c holds: a near step is a quarter of a cell, so most steps keep it
    var pj = -1000000
    var c = F16(0)
    var full = True
    var stepx = True
    for _ in range(1024):
        if enter >= t1 or enter >= best[0] or (d[2] >= 0 and o[2] + enter * d[2] > T[Int(T[7])] + hmax):
            break  # done, hit, or rising above the highest ground and tree
        var leave = min(min(mx, my), t1)
        var mid = o + d * (0.5 * (enter + leave))
        var i = Int(floor(mid[0]))
        var j = Int(floor(mid[1]))
        var zlo = min(o[2] + enter * d[2], o[2] + leave * d[2])
        var top = cap(T, i, j)  # the patch is a convex combination of its 16 controls: their maximum bounds it
        if zlo <= top:
            if i != pi or j != pj:
                pi = i
                pj = j
                c = patch(T, i, j)
            var g = ground(T, c, i, j, o, d, enter, leave)
            if g[0] < best[0]:
                best = g
        var ttop = top
        if trees and zlo <= top + 2.2 / T[2]:  # the cell's prop (under 2 m high), near phase only
            var pr = prop(T, Int(cx), Int(cy), o, d)
            if pr[0] < best[0] and pr[0] <= t1:
                best = pr
        if trees:  # the ground under any of the 5 x 5 slots: the bounds of the patches their block spans
            for nj in range(Int(floor((cy - 2) * cs)), Int(floor((cy + 3) * cs)) + 1):
                for ni in range(Int(floor((cx - 2) * cs)), Int(floor((cx + 3) * cs)) + 1):
                    ttop = max(ttop, cap(T, ni, nj))
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


def sun_at(t: Float32) -> F4:
    """The sun's unit direction at time t (s; one day = 150 s, midnight at 0, so it rises at 6 h and sets at 18 h)."""
    var a = t / 150 * 2 * pi - pi / 2
    var s = F4(cos(a) * 0.6, cos(a) * 0.4, sin(a), 0)
    return s / sqrt((s * s).reduce_add())


def lw_pl(h: Float32, lo: Float32, hi: Float32) -> Float32:
    """Air temperature at hour h of a day with minimum lo and maximum hi under the game's sun (up 6-18 h): Parton and Logan
    (1981), 1.5 m air (a 1.86 h, b 2.2, c -0.17 h): a sine from the minimum (sunrise + c) to sunset, then exponential decay,
    normalised here to reach the next minimum, so the curve is continuous."""
    var m = (h - 5.83 + 24) % 24  # hours since the minimum
    if m <= 12.17:
        return lo + (hi - lo) * sin(pi * m / 15.72)
    var e = exp(-2.2 / 12 * (m - 12.17))
    var e1 = exp(Float32(-2.2 / 12 * 11.83))
    return lo + (hi - lo) * Float32(sin(pi * 12.17 / 15.72)) * (e - e1) / (1 - e1)  # from the sunset value down to lo


def wind_chill(t: Float32, u: Float32) -> Float32:
    """Felt temperature (C) at air t (C) and 10 m wind u (m/s): the NWS/MSC 2001 wind chill, valid at t <= 10 C and over
    4.8 km/h; the air itself otherwise."""
    var v = u * 3.6
    if t > 10 or v <= 4.8:
        return t
    var p = v ** 0.16
    return 13.12 + 0.6215 * t - 11.37 * p + 0.3965 * t * p


def lw_temp(T: Pointer[Float32, _], x: Float32, y: Float32, t: Float32, doy: Float32, cloud: Float32, clim: F8) -> F8:
    """Air temperature at (x, y) cells, 1.5 m up, at time t (s, sky_cam's clock) on day of year doy under cloud cover 0..1.
    clim (game/climate.mojo, fitted to Open-Meteo normals): mean C at height 0, its annual amplitude K and warmest day, the
    diurnal range K, its amplitude and widest day, the mean 10 m wind m/s. Returns (air C, felt C, wind m/s, cold pool K,
    slope sun K, daily mean C, diurnal range K, height over the 300 m ring m).
    Terms: lapse 6.5 K/km; annual harmonics; Parton-Logan day (lw_pl), the range cut by cloud (Dai et al. 1999: up to 60 %),
    by water within 300 m (half) and under a canopy (30 %); cold air pooling in hollows on clear calm nights, 0.04 K per metre
    below the ring mean up to 100 m; slopes facing the sun warmer than flat ground by up to 3 K per unit of extra beam;
    ridge speed-up of the wind (Taylor and Lee 1984, 3-D hill: 0.8 H / L); NWS wind chill."""
    var cm = T[2]
    var z = lw_height(T, x, y) * cm
    var w = Float32(2 * pi / 365.25)
    var td = clim[0] + clim[1] * cos(w * (doy - clim[2])) - 0.0065 * z
    var dtr = clim[3] + clim[4] * cos(w * (doy - clim[5]))
    var r = 300 / cm
    var ring = Float32(0)
    var wet = Float32(0)
    for k in range(16):
        var a = Float32(k) * pi / 8
        var h = lw_height(T, x + r * cos(a), y + r * sin(a))
        ring += h
        wet += Float32(1) if h <= T[1] + 0.02 else Float32(0)
    var rel = z - ring / 16 * cm
    var u = clim[6] * (1 + 0.8 * max(rel, Float32(0)) / 300)
    var forest = lw_tree_near(T, x, y, 0)
    dtr *= (1 - 0.6 * cloud) * (1 - 0.5 * wet / 16) * (Float32(0.7) if forest else Float32(1))
    var air = lw_pl((t % 150) / 150 * 24, td - dtr / 2, td + dtr / 2)
    var calm = (1 - cloud) * min(max(1 - u / 5, Float32(0)), Float32(1))
    var pool = -0.04 * min(max(-rel, Float32(0)), Float32(100)) * calm * min(max((td + dtr / 2 - air) / max(dtr, 0.1), Float32(0)), Float32(1))
    var s = sun_at(t)
    var dsun = Float32(0)
    if s[2] > 0 and not forest:
        var e = Float32(0.05)
        var n = F4(lw_height(T, x - e, y) - lw_height(T, x + e, y), lw_height(T, x, y - e) - lw_height(T, x, y + e), 2 * e, 0)
        n /= sqrt((n * n).reduce_add())
        var lit = lw_lit(T, x, y, asin(s[2]), (atan2(s[1], s[0]) / (2 * pi) * 8 + 8) % 8)
        dsun = 3 * (1 - cloud) * (lit * max((n * s).reduce_add(), Float32(0)) - s[2])
    air += pool + dsun
    return F8(air, wind_chill(air, u), u, pool, dsun, td, dtr, rel)


def sky_cam(mut c: List[Float32], t: Float32, m: Float32):
    """cam[16:32] for render.mojo LIVING=1 at time t (s; one day = 150 s, the web's sky(t)): metres per unit, light dir
    (sun, or the moon below the horizon), zenith and horizon colours, exposure, t, light elevation, azimuth in eighths,
    night, dusk."""
    var s = sun_at(t)
    var sx = s[0]
    var sy = s[1]
    var sz = s[2]
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
