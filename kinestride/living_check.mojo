"""Gate for living.mojo on the CPU: lw_height reproduces the table at grid nodes and wraps as a torus; lw_hit agrees
with a brute 1e-3 march over an independent inside test (ground below lw_height, or inside a tree cone) on 256 rays.
usage: living_check WORLD.bin    (prints PASS / FAIL lines, exit 1 on any FAIL)
"""
from std.math import cos, floor, sin, sqrt
from std.sys import argv, exit

from living import BIG, F4, NEAR_M, cone_part, lw_height, lw_hit, tree_at


def inside_tree(T: Pointer[Float32, _], p: F4) -> Bool:
    """p inside a cone of the 5 x 5 tree slots around it, from the cone's radius profile (not a ray test)."""
    var tn = T[3] / T[0]
    var ci = Int(floor(p[0] * tn))
    var cj = Int(floor(p[1] * tn))
    for dj in range(-2, 3):
        for di in range(-2, 3):
            var tr = tree_at(T, ci + di, cj + dj)
            if tr[5] < 0:
                continue
            var r = sqrt((p[0] - tr[0]) * (p[0] - tr[0]) + (p[1] - tr[1]) * (p[1] - tr[1]))
            for part in range(2):
                var c = cone_part(Int(tr[5]), part)
                if c[3] == 0:
                    break
                var u = (p[2] - tr[2] - c[0] * tr[3]) / (c[2] * tr[3])  # 0 at the base, 1 at the apex
                if u >= 0 and u <= 1 and r <= c[1] * tr[4] * (1 - u):
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

    # 2. lw_hit against a brute march (trees within NEAR_M only, as lw_hit), rays from a walker's eye at 1.7 m, 2.5 km far
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
        var o = F4(x, y, max(lw_height(T, x, y), T[1]) + 1.7 / T[2], 0)
        var d = F4(cos(az) * cos(el), sin(az) * cos(el), sin(el), 0)
        if inside_tree(T, o):  # a walker never stands inside a trunk; such an eye sees the cone from inside
            skipped += 1
            continue
        var far = 2500 / T[2]
        var near = NEAR_M / T[2]
        var h = lw_hit(T, o, d, far)
        var tb = BIG
        var t = Float32(1e-3)
        while t < far:
            var p = o + d * t
            if p[2] < lw_height(T, p[0], p[1]) or (t < near and inside_tree(T, p)):
                tb = t
                break
            t += Float32(1e-3) if t < near else Float32(1e-2)
        var same = (h[0] == BIG and tb == BIG) or (h[0] < BIG and tb < BIG and abs(h[0] - tb) < 2e-3 * (Float32(1) if tb < near else Float32(10)))
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
    # 2b. aimed rays: 6 trees of every species seen from 40 m at crown height (random rays rarely meet a ridge pine)
    var tn2 = Int(T[3])
    var aimed = 0
    var aimed_ok = 0
    for sp in range(6):
        var found = 0
        var k = sp * 7919
        while found < 6 and k < tn2 * tn2 + sp * 7919:
            var tr = tree_at(T, k % tn2, (k // tn2) % tn2)
            k += 97
            if Int(tr[5]) != sp:
                continue
            found += 1
            var c = cone_part(sp, 0)
            var target = F4(tr[0], tr[1], tr[2] + (c[0] + 0.3 * c[2]) * tr[3], 0)
            var o = target + F4(40 / T[2], 0, 0, 0)
            o[2] = max(lw_height(T, o[0], o[1]), T[1]) + max(Float32(1.7) / T[2], target[2] - lw_height(T, o[0], o[1]))
            if inside_tree(T, o):
                continue
            var d = target - o
            d = d / sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            var h = lw_hit(T, o, d, NEAR_M / T[2])
            var tb = BIG
            var t = Float32(1e-3)
            while t < NEAR_M / T[2]:
                var q = o + d * t
                if q[2] < lw_height(T, q[0], q[1]) or inside_tree(T, q):
                    tb = t
                    break
                t += Float32(1e-3)
            aimed += 1
            if (h[0] == BIG and tb == BIG) or (h[0] < BIG and tb < BIG and abs(h[0] - tb) < 2e-3):
                aimed_ok += 1
            else:
                print("  aimed species", sp, "hit", h[0], "brute", tb)
    var ok2b = aimed_ok == aimed and aimed >= 24
    print("PASS" if ok2b else "FAIL", "aimed rays (6 per species):", aimed_ok, "/", aimed, "agree")
    bad += 0 if ok2b else 1

    # 4. shapes: cone_part is shared by lw_hit and the inside test above, so check it against the web's TREE_CONES
    # (the other front end of the same world), passed in as 36 numbers: species x 2 cones x (base, radius, height)
    if len(argv()) > 2:
        var web = String(argv()[2]).split(",")
        var off = 0
        for sp in range(6):
            for part in range(2):
                var c = cone_part(sp, part)
                for q in range(3):
                    off = max(off, Int(abs(Float64(c[q] if c[3] > 0 else 0) - atof(web[6 * sp + 3 * part + q])) * 1000))
        var ok4 = off == 0
        print("PASS" if ok4 else "FAIL", "cone shapes equal web/world.mjs TREE_CONES (max diff x1000:", off, ")")
        bad += 0 if ok4 else 1

    # 3. tree slots: the occupied slots tree_at sees over the whole grid equal a raw scan of the slot block
    var tn = Int(T[3])
    var raw_n = 0
    for k in range(tn * tn):
        raw_n += 1 if T[4 + 5 * n * n + k] != 0 else 0
    var seen = 0
    for tj in range(tn):
        for ti in range(tn):
            seen += 1 if tree_at(T, ti, tj)[5] >= 0 else 0
    var ok3 = seen == raw_n and raw_n > 1000
    print("PASS" if ok3 else "FAIL", "tree slots: tree_at sees", seen, "of", raw_n, "occupied")
    bad += 0 if ok3 else 1
    if bad:
        exit(1)
    print("PASS all living checks")
