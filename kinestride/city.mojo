"""An endless city as a function, not a table (Kinestride's world method at human scale): every 64 m cell's block,
lots and buildings follow from (seed, global cell), so the renderer and the camera evaluate the same city with no
stored geometry. city_check.mojo is the gate (a point march over city_height, rebase, streets, golden snapshot).

units: 1 = 1 m. Positions are float32 LOCAL coordinates; the caller's origin cell (ox, oy) is added to every cell
index, so content keys on the global cell and the camera rebases by whole cells without losing precision.
cell (gx, gy): a block inset by a street (6 m, avenues 14 m every 8 cells on the west and south sides), split by a hash
into 1-4 lots. A lot holds a box, or a polar footprint r(t) = R (1 + a cos(k t + phi)): square-ish (k 4), star
(k 3-8) or round; its height follows a district field of integer-period sines times a hash jitter, in 1-3 setback
tiers (each 0.75 of the one below).
materials: 0 sky (miss), 1 road, 2 sidewalk, 3-6 facade palettes.

"""
from std.math import cos, floor, sin, sqrt

from world import hash

comptime F4 = SIMD[DType.float32, 4]
comptime F8 = SIMD[DType.float32, 8]
comptime F16 = SIMD[DType.float32, 16]
comptime CELL = Float32(64)
comptime STREET = Float32(6)
comptime AVENUE = Float32(14)
comptime EVERY = 8
comptime INSET = Float32(1.5)
comptime TAU = Float32(6.2831853)
comptime BIG = Float32(1e30)
comptime MISS = F4(BIG, 0, 0, 0)


def district(gx: Int, gy: Int) -> Float32:
    """0-1 building height field over global cells: integer periods, so it is exact at any distance."""
    var a = Float32(gx % 37) * (TAU / 37)
    var b = Float32(gy % 29) * (TAU / 29)
    var c = Float32((gx + gy) % 89) * (TAU / 89)
    return 0.5 + 0.3 * sin(a + 1.3) * sin(b - 0.4) + 0.2 * sin(c)


def lot(seed: UInt32, gx: Int, gy: Int, bx: Float32, by: Float32, i: Int) -> F16:
    """Lot i (0-3) of global cell (gx, gy), cell corner at local (bx, by): centre x y, half size x y, height, kind
    (0 box, 1 polar), R, k, a, phi, tiers, material, -, -, -, valid (0: the block has fewer lots)."""
    var h = hash(Int32(gx), Int32(gy), seed)
    var nx = 1 + Int(h & 1)
    var ny = 1 + Int((h >> 1) & 1)
    var L = F16(0)
    if i >= nx * ny:
        return L
    var lx = i % nx
    var ly = i // nx
    var x0 = bx + (AVENUE if gx % EVERY == 0 else STREET)
    var y0 = by + (AVENUE if gy % EVERY == 0 else STREET)
    var w = (bx + CELL - STREET - x0) / Float32(nx)
    var e = (by + CELL - STREET - y0) / Float32(ny)
    L[0] = x0 + (Float32(lx) + 0.5) * w
    L[1] = y0 + (Float32(ly) + 0.5) * e
    L[2] = w / 2 - INSET
    L[3] = e / 2 - INSET
    var q = hash(Int32(2 * gx + lx), Int32(2 * gy + ly), seed + 7)
    var j = Float32(hash(Int32(2 * gx + lx), Int32(2 * gy + ly), seed + 8) >> 8) * (1.0 / 16777216.0)
    var d = district(gx, gy)
    L[4] = 6 + 150 * d * d * d * (0.4 + 0.8 * j)
    var c = Int(q & 15)
    if c >= 10:
        L[5] = 1
        if c < 13:
            L[7] = 4
            L[8] = 0.06
        elif c < 15:
            L[7] = Float32(3 + Int((q >> 8) % 6))
            L[8] = 0.15 + 0.15 * Float32((q >> 12) & 255) * (1.0 / 255.0)
            L[9] = TAU * Float32((q >> 20) & 255) * (1.0 / 256.0)
    L[6] = min(L[2], L[3]) / (1 + L[8])
    L[10] = Float32(1 + Int((q >> 4) % 3)) if L[4] > 40 else 1
    L[11] = Float32(3 + Int((q >> 28) & 3))
    L[15] = 1
    return L


def frac(n: Int, t: Int) -> Float32:
    """Height fraction where tier t of n starts (t = n: the roof)."""
    if t == 0:
        return 0
    if t == n:
        return 1
    if n == 2:
        return 0.7
    return 0.6 if t == 1 else 0.85


def scale(t: Int) -> Float32:
    return Float32(1) if t == 0 else (Float32(0.75) if t == 1 else Float32(0.5625))


def atan2(y: Float32, x: Float32) -> Float32:
    """atan2 for the card too (libm's atan2f does not link on amdgpu): octant reduction, a pi/8 split, then the odd
    series to t^15 (truncation < 2e-8 rad, under float32 rounding)."""
    var ax = abs(x)
    var ay = abs(y)
    var big = max(ax, ay)
    if big == 0:
        return 0
    var t = min(ax, ay) / big
    var off = Float32(0)
    if t > 0.41421356:
        t = (t - 1) / (t + 1)
        off = 0.78539816
    var t2 = t * t
    var r = off + t * (1 + t2 * (-1.0 / 3 + t2 * (1.0 / 5 + t2 * (-1.0 / 7 + t2 * (1.0 / 9 + t2 * (-1.0 / 11 + t2 * (1.0 / 13 - t2 / 15)))))))
    if ay > ax:
        r = 1.57079633 - r
    if x < 0:
        r = 3.14159265 - r
    return -r if y < 0 else r


def edge(x: Float32, y: Float32, R: Float32, k: Float32, a: Float32, phi: Float32) -> Float32:
    """Signed distance-like footprint test of a polar lot, (x, y) from its centre: < 0 inside."""
    return sqrt(x * x + y * y) - R * (1 + a * cos(k * atan2(y, x) + phi))


def side(x: Float32, y: Float32, t: Float32, R: Float32, k: Float32, a: Float32, phi: Float32) -> F4:
    """(t, outward wall normal) of a polar lot at (x, y) from its centre: the gradient of edge."""
    var r = sqrt(x * x + y * y)
    var dr = -R * a * k * sin(k * atan2(y, x) + phi)
    var gx = x / r + dr * y / (r * r)
    var gy = y / r - dr * x / (r * r)
    var n = sqrt(gx * gx + gy * gy)
    return F4(t, gx / n, gy / n, 0)


def box(o: F4, d: F4, lo: F4, hi: F4) -> F4:
    """(t, normal) of the ray's entry into the box lo..hi; MISS, or t 0 with the eye inside."""
    var t0 = -BIG
    var t1 = BIG
    var n = F4(0)
    for k in range(3):
        var inv = 1 / d[k]
        var a = (lo[k] - o[k]) * inv
        var b = (hi[k] - o[k]) * inv
        var s = Float32(-1)
        if a > b:
            var c = a
            a = b
            b = c
            s = 1
        if a > t0:
            t0 = a
            n = F4(0)
            n[k] = s
        t1 = min(t1, b)
    if t0 > t1 or t1 < 0:
        return MISS
    if t0 < 0:
        return F4(0)
    return F4(t0, n[0], n[1], n[2])


def polar(o: F4, d: F4, cx: Float32, cy: Float32, z0: Float32, z1: Float32, R: Float32, k: Float32, a: Float32, phi: Float32) -> F4:
    """(t, normal) of the ray's entry into an extruded polar footprint: its bounding cylinder and z slab clip the
    ray, 32 steps find the first inside sample, 8 bisections refine it."""
    var px = o[0] - cx
    var py = o[1] - cy
    var rm = R * (1 + a)
    var A = d[0] * d[0] + d[1] * d[1]
    var B = px * d[0] + py * d[1]
    var C = px * px + py * py - rm * rm
    var ta = -BIG
    var tb = BIG
    if A > 0:
        var disc = B * B - A * C
        if disc < 0:
            return MISS
        var s = sqrt(disc)
        ta = (-B - s) / A
        tb = (-B + s) / A
    elif C >= 0:
        return MISS
    var tc = -BIG
    var td = BIG
    var nz = Float32(0)
    if d[2] != 0:
        var u = (z0 - o[2]) / d[2]
        var v = (z1 - o[2]) / d[2]
        nz = -1 if d[2] > 0 else 1
        tc = min(u, v)
        td = max(u, v)
    elif o[2] < z0 or o[2] > z1:
        return MISS
    var t0 = max(max(ta, tc), 0)
    var t1 = min(tb, td)
    if t0 >= t1:
        return MISS
    if edge(px + t0 * d[0], py + t0 * d[1], R, k, a, phi) < 0:
        if t0 == tc:
            return F4(t0, 0, 0, nz)
        return side(px + t0 * d[0], py + t0 * d[1], t0, R, k, a, phi)
    var lo = t0
    var dt = (t1 - t0) / 32
    for i in range(1, 33):
        var t = t0 + Float32(i) * dt
        if edge(px + t * d[0], py + t * d[1], R, k, a, phi) < 0:
            var hi = t
            for _ in range(8):
                var m = (lo + hi) / 2
                if edge(px + m * d[0], py + m * d[1], R, k, a, phi) < 0:
                    hi = m
                else:
                    lo = m
            return side(px + hi * d[0], py + hi * d[1], hi, R, k, a, phi)
        lo = t
    return MISS


def cell_hit(seed: UInt32, gx: Int, gy: Int, bx: Float32, by: Float32, o: F4, d: F4) -> F8:
    """(t, normal, material) of the nearest building in one cell; t BIG on a miss."""
    var best = F8(BIG, 0, 0, 0, 0, 0, 0, 0)
    for i in range(4):
        var L = lot(seed, gx, gy, bx, by, i)
        if L[15] == 0:
            break
        var n = Int(L[10])
        for t in range(n):
            var s = scale(t)
            var z0 = L[4] * frac(n, t)
            var z1 = L[4] * frac(n, t + 1)
            var h: F4
            if L[5] == 0:
                h = box(o, d, F4(L[0] - L[2] * s, L[1] - L[3] * s, z0, 0), F4(L[0] + L[2] * s, L[1] + L[3] * s, z1, 0))
            else:
                h = polar(o, d, L[0], L[1], z0, z1, L[6] * s, L[7], L[8], L[9])
            if h[0] < best[0]:
                best = F8(h[0], h[1], h[2], h[3], L[11], 0, 0, 0)
    return best


def ground(seed: UInt32, ox: Int, oy: Int, x: Float32, y: Float32) -> Float32:
    """Ground material at local (x, y): 2 sidewalk inside a block, 1 road."""
    var cx = floor(x / CELL)
    var cy = floor(y / CELL)
    var gx = Int(cx) + ox
    var gy = Int(cy) + oy
    var u = x - cx * CELL
    var v = y - cy * CELL
    var inside = u > (AVENUE if gx % EVERY == 0 else STREET) and u < CELL - STREET and v > (AVENUE if gy % EVERY == 0 else STREET) and v < CELL - STREET
    return 2 if inside else 1


def city_hit(seed: UInt32, ox: Int, oy: Int, o: F4, d: F4, far: Float32) -> F8:
    """(t, normal, material) of the first surface along the ray (d unit, local eye o): buildings by a 2-D DDA over
    cells, then the z = 0 ground; t BIG, material 0 past far."""
    var tg = -o[2] / d[2] if d[2] < 0 else BIG
    var limit = min(far, tg)
    var cx = floor(o[0] / CELL)
    var cy = floor(o[1] / CELL)
    var sx = Float32(1) if d[0] > 0 else Float32(-1)
    var sy = Float32(1) if d[1] > 0 else Float32(-1)
    var mx = BIG
    var my = BIG
    var ex = BIG
    var ey = BIG
    if d[0] != 0:
        mx = ((cx + Float32(1 if d[0] > 0 else 0)) * CELL - o[0]) / d[0]
        ex = CELL / abs(d[0])
    if d[1] != 0:
        my = ((cy + Float32(1 if d[1] > 0 else 0)) * CELL - o[1]) / d[1]
        ey = CELL / abs(d[1])
    var enter = Float32(0)
    for _ in range(256):
        if enter >= limit:
            break
        var h = cell_hit(seed, Int(cx) + ox, Int(cy) + oy, cx * CELL, cy * CELL, o, d)
        if h[0] < BIG:
            return h if h[0] <= far else F8(BIG, 0, 0, 0, 0, 0, 0, 0)
        if mx < my:
            enter = mx
            mx += ex
            cx += sx
        else:
            enter = my
            my += ey
            cy += sy
    if tg <= far:
        return F8(tg, 0, 0, 1, ground(seed, ox, oy, o[0] + tg * d[0], o[1] + tg * d[1]), 0, 0, 0)
    return F8(BIG, 0, 0, 0, 0, 0, 0, 0)


def city_height(seed: UInt32, ox: Int, oy: Int, x: Float32, y: Float32) -> Float32:
    """Top of the buildings at local (x, y), 0 on the street: the camera's floor."""
    var cx = floor(x / CELL)
    var cy = floor(y / CELL)
    var top = Float32(0)
    for i in range(4):
        var L = lot(seed, Int(cx) + ox, Int(cy) + oy, cx * CELL, cy * CELL, i)
        if L[15] == 0:
            break
        var n = Int(L[10])
        for t in range(n):
            var s = scale(t)
            var inside = abs(x - L[0]) < L[2] * s and abs(y - L[1]) < L[3] * s if L[5] == 0 else edge(x - L[0], y - L[1], L[6] * s, L[7], L[8], L[9]) < 0
            if inside:
                top = max(top, L[4] * frac(n, t + 1))
    return top

