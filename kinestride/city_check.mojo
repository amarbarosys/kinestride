"""city.mojo's gate, Mojo only:
1. no surface before a hit, every hit real: each ray's analytic hit against a 2 cm point march over city_height
2. a whole-cell rebase (origin one cell east, eyes 64 m west) gives the same hits, 10^6 m out
3. streets stay clear: no building top on the street strips
4. a golden snapshot of hits and building tops (city_golden.bin, written by the version that matched a numpy twin
   on 1.75e5 rays to the 6 mm polar bisection bracket, materials equal)

usage: city_check GOLDEN   |   city_check write GOLDEN
build: mojo build -I kinestride kinestride/city_check.mojo
"""
from std.math import cos, sin, sqrt
from std.memory import bitcast
from std.sys import argv, exit, stderr

from city import BIG, CELL, F4, F8, STREET, TAU, city_height, city_hit
from world import hash

comptime SEED = UInt32(1)
comptime Q = Float32(24.5 / 4096)  # the largest polar R / (32 steps x 2^8 bisections / 2): city.mojo's wall bracket
comptime FAR = Float32(600)
comptime STEP = Float32(0.02)


def u(i: Int, j: Int) -> Float32:
    return Float32(hash(Int32(i), Int32(j), 4242) >> 8) * (1.0 / 16777216.0)


def ray(i: Int) -> Tuple[F4, F4]:
    """Ray i: an eye over the 600 m square around the origin, 1.7-60 m up, a direction mostly near the horizon."""
    var z = u(i, 3) - 0.6
    var r = sqrt(1 - z * z)
    var a = TAU * u(i, 4)
    return (F4(600 * u(i, 0) - 300, 600 * u(i, 1) - 300, 1.7 + 58.3 * u(i, 2), 0), F4(r * cos(a), r * sin(a), z, 0))


def inside(p: F4) -> Bool:
    return p[2] < 0 or p[2] < city_height(SEED, 0, 0, p[0], p[1])


def same(a: F8, b: F8, d: F4) -> Bool:
    """Equal material, the hit point (t 0 for an eye inside) within 1e-5 of max(t, 1 m) or Q along the surface normal, the normal within 1e-4."""
    if a[4] != b[4]:
        return False
    if a[4] == 0:
        return True
    if b[0] == 0:  # the eye inside a building: t 0, no normal
        return a[0] == 0
    var dn = abs(d[0] * b[1] + d[1] * b[2] + d[2] * b[3])
    return abs(a[0] - b[0]) * dn <= max(1e-5 * max(b[0], 1), Q) and a[1] * b[1] + a[2] * b[2] + a[3] * b[3] >= 1 - 1e-4


def march(n: Int) -> Int:
    """Check 1 on n rays out to 200 m at the origin."""
    var missed = 0
    var false_hit = 0
    var grazed = 0
    var used = 0
    for i in range(n):
        var r = ray(i)
        var o = r[0]
        var d = r[1]
        if inside(o):
            continue
        used += 1
        var h = city_hit(SEED, 0, 0, o, d, 200)
        var dn = abs(d[0] * h[1] + d[1] * h[2] + d[2] * h[3])
        var steps = Int(min(h[0], 200) / STEP)
        var slack = Int((STEP + Q / max(dn, 0.2)) / STEP) + 1  # the march may enter a polar wall inside its bracket
        for s in range(1, steps - slack):
            if inside(o + Float32(s) * STEP * d):
                if dn < 0.2:
                    grazed += 1  # a corner or star tip clipped at a grazing angle: city.mojo's 32 steps can skip it
                else:
                    missed += 1
                    if missed <= 3:
                        print("  missed surface: ray", i, "march t", Float32(s) * STEP, "hit t", h[0])
                break
        if h[0] < BIG and h[4] >= 3 and dn >= 0.2:
            var e = STEP / dn
            if not inside(o + (h[0] + e) * d) or inside(o + (h[0] - e - Q / dn) * d):
                false_hit += 1
                if false_hit <= 3:
                    print("  false hit: ray", i, "t", h[0], "material", h[4])
    var ok = missed == 0 and false_hit == 0 and grazed * 200 <= used
    print("PASS" if ok else "FAIL", "point march (2 cm) vs hits:", used, "rays,", missed, "missed surfaces,", false_hit, "false hits,", grazed, "grazing clips (<= 0.5%)")
    return 0 if ok else 1


def rebase(n: Int) -> Int:
    var off = 0
    for i in range(n):
        var r = ray(i)
        var a = city_hit(SEED, 15625, -15625, r[0], r[1], FAR)  # 15625 cells = 10^6 m
        var b = city_hit(SEED, 15626, -15625, r[0] - F4(CELL, 0, 0, 0), r[1], FAR)
        off += Int(not same(b, a, r[1]))
    print("PASS" if off == 0 else "FAIL", "rebase by one cell at 10^6 m:", off, "/", n, "off")
    return 0 if off == 0 else 1


def streets(n: Int) -> Int:
    """Check 3: points on the 6 m strip east and north of each block, any cell."""
    var bad = 0
    for i in range(n):
        var cx = Float32(Int(200 * u(i, 5)) - 100) * CELL
        var cy = Float32(Int(200 * u(i, 6)) - 100) * CELL
        var x = cx + CELL - STREET * u(i, 7)
        var y = cy + CELL * u(i, 8)
        if i % 2:
            x = cx + CELL * u(i, 7)
            y = cy + CELL - STREET * u(i, 8)
        bad += Int(city_height(SEED, 0, 0, x, y) > 0)
    print("PASS" if bad == 0 else "FAIL", "streets clear:", bad, "/", n, "street points under a building")
    return 0 if bad == 0 else 1


def snapshot() -> List[Float32]:
    """512 hits at the origin and 512 at 10^6 m (t, normal, material), then 1024 building tops."""
    var v = List[Float32]()
    for k in range(2):
        for i in range(512):
            var r = ray(10000 + i)
            var h = city_hit(SEED, 15625 * k, -15625 * k, r[0], r[1], FAR)
            for c in range(5):
                v.append(h[c])
    for i in range(1024):
        v.append(city_height(SEED, 0, 0, 600 * u(i, 9) - 300, 600 * u(i, 10) - 300))
    return v^


def golden(path: String) raises -> Int:
    var b = open(path, "r").read_bytes()
    var want = List[Float32]()
    for q in range(0, len(b), 4):
        want.append(bitcast[DType.float32](UInt32(b[q]) | (UInt32(b[q + 1]) << 8) | (UInt32(b[q + 2]) << 16) | (UInt32(b[q + 3]) << 24)))
    var got = snapshot()
    if len(want) != len(got):
        print("FAIL golden: size", len(want), "want", len(got))
        return 1
    var off = 0
    for k in range(1024):
        var r = ray(10000 + k % 512)
        var g = F8(got[5 * k], got[5 * k + 1], got[5 * k + 2], got[5 * k + 3], got[5 * k + 4], 0, 0, 0)
        var w = F8(want[5 * k], want[5 * k + 1], want[5 * k + 2], want[5 * k + 3], want[5 * k + 4], 0, 0, 0)
        off += Int(not same(g, w, r[1]))
    var tops = 0
    for k in range(5120, len(got)):
        tops += Int(abs(got[k] - want[k]) > 1e-5 * max(want[k], 1))
    var ok = off == 0 and tops == 0
    print("PASS" if ok else "FAIL", "golden snapshot:", off, "/ 1024 hits,", tops, "/ 1024 building tops off")
    return 0 if ok else 1


def main() raises:
    var a = argv()
    if len(a) == 3 and a[1] == "write":
        var v = snapshot()
        with open(String(a[2]), "w") as f:
            f.write_bytes(Span(unsafe_ptr=v.unsafe_ptr().unsafe_bitcast[UInt8](), length=4 * len(v)))
        print("wrote", len(v), "floats to", a[2])
        return
    if len(a) != 2:
        print("usage: city_check GOLDEN | city_check write GOLDEN", file=stderr)
        exit(2)
    var bad = march(2000) + rebase(20000) + streets(20000) + golden(String(a[1]))
    if bad:
        print("FAIL city check:", bad, "checks", file=stderr)
        exit(1)
    print("PASS city check")
