"""Gate for living.mojo on the CPU: lw_height reproduces the table at grid nodes and wraps as a torus; lw_hit agrees
with a brute 1e-3 march over an independent inside test (ground below lw_height, or inside a tree cone) on 256 rays.
usage: living_check WORLD.bin    (prints PASS / FAIL lines, exit 1 on any FAIL)
"""
from std.math import cos, floor, sin, sqrt
from std.sys import argv, exit

from living import BIG, F4, TS, byte, cell, lw_height, lw_hit


def inside_tree(T: Pointer[Float32, _], p: F4) -> Bool:
    """p inside a cone of the 3 x 3 cells around it, from each kind's radius profile."""
    var ci = Int(floor(p[0]))
    var cj = Int(floor(p[1]))
    for dj in range(-1, 2):
        for di in range(-1, 2):
            var i = ci + di
            var j = cj + dj
            var v = cell(T, i, j, 4)
            var kind = Int(byte(v, 0)) - 1
            if kind < 0:
                continue
            var s = byte(v, 1) / 100
            var cx = Float32(i) + byte(v, 2) / 255
            var cy = Float32(j) + byte(v, 3) / 255
            var z = lw_height(T, cx, cy) - 0.1
            var r = sqrt((p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy))
            for part in range(2):
                var zb = z
                var r0 = 0.5 * s
                var h = 0.6 * s * TS
                if kind == 0:
                    zb = z + Float32(part) * 0.75 * s * TS
                    r0 = 0.5 * s * (1 - 0.35 * Float32(part))
                    h = 1.3 * s * TS
                elif kind == 1:
                    zb = z + 0.7 * s * TS
                    r0 = 0.75 * s
                    h = (Float32(1.0) if part == 0 else Float32(-0.7)) * s * TS
                elif part == 1:
                    continue
                var u = (p[2] - zb) / h  # 0 at the base, 1 at the apex
                if u >= 0 and u <= 1 and r <= r0 * (1 - u):
                    return True
    return False


def lcg(mut s: UInt64, m: Int) -> Float32:
    s = s * 6364136223846793005 + 1442695040888963407
    return Float32(Int(s >> 40) % m)


def main() raises:
    var raw = open(String(argv()[1]), "r").read_bytes()
    var tab = List[Float32](length=len(raw) // 4, fill=0)
    var rp = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    for k in range(len(tab)):
        tab[k] = rp[k]
    var T = tab.unsafe_ptr()
    var n = Int(T[0])
    var bad = 0

    # 1. grid nodes and the torus wrap
    var worst = Float32(0)
    for j in range(n):
        for i in range(n):
            worst = max(worst, abs(lw_height(T, Float32(i), Float32(j)) - T[4 + 5 * (j * n + i)]))
            worst = max(worst, abs(lw_height(T, Float32(i + n) + 0.37, Float32(j) + 0.61) - lw_height(T, Float32(i) + 0.37, Float32(j - n) + 0.61)))
    var seam = abs(lw_height(T, Float32(n) - 1e-3, 5.5) - lw_height(T, 1e-3, 5.5))
    var ok1 = worst < 1e-4 and seam < 0.05
    print("PASS" if ok1 else "FAIL", "height: nodes and wrap max err", worst, "seam step", seam)
    bad += 0 if ok1 else 1

    # 2. lw_hit against a brute march, rays from a walker's eye height (0.85 units = 1.7 m)
    var s = UInt64(12345)
    var agree = 0
    var trees = 0
    var maxerr = Float32(0)
    var skipped = 0
    for r in range(256):
        var x = lcg(s, 12800) / 100
        var y = lcg(s, 12800) / 100
        var az = lcg(s, 6283) / 1000
        var el = -0.5 + lcg(s, 600) / 1000
        var o = F4(x, y, max(lw_height(T, x, y), T[1]) + 0.85, 0)
        var d = F4(cos(az) * cos(el), sin(az) * cos(el), sin(el), 0)
        if inside_tree(T, o):  # a walker never stands inside a trunk; such an eye sees the cone from inside
            skipped += 1
            continue
        var h = lw_hit(T, o, d, 40)
        var tb = BIG
        var t = Float32(1e-3)
        while t < 40:
            var p = o + d * t
            if p[2] < lw_height(T, p[0], p[1]) or inside_tree(T, p):
                tb = t
                break
            t += 1e-3
        var same = (h[0] == BIG and tb == BIG) or (h[0] < BIG and tb < BIG and abs(h[0] - tb) < 2e-3)
        if h[0] < BIG and tb < BIG:
            maxerr = max(maxerr, abs(h[0] - tb))
        if h[4] >= 3:
            trees += 1
        if same:
            agree += 1
        elif r - agree - skipped < 12:
            print("  ray", r, "hit", h[0], "mat", h[4], "brute", tb, "brute mat", Float32(1) if tb < BIG and (o + d * tb)[2] < lw_height(T, (o + d * tb)[0], (o + d * tb)[1]) else Float32(3))
    var ok2 = agree == 256 - skipped and skipped < 32 and trees > 4
    print("PASS" if ok2 else "FAIL", "hit vs brute march:", agree, "/", 256 - skipped, "agree (", skipped, "eyes inside a tree skipped ), max |dt|", maxerr, "tree hits", trees)
    bad += 0 if ok2 else 1
    if bad:
        exit(1)
    print("PASS all living checks")
