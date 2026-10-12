"""Mesh builders for the renderer's world parts (model.meshes world=...): an icosphere, MuJoCo box and ellipsoid
geoms as triangles, and a splitter that keeps every triangle side short (render.raster gives one thread a triangle's
whole screen box). Port of assets.py, which stays as this file's oracle (assets_check.py).

usage: assets sphere N OUT | assets primitive BOX SX SY SZ OUT | assets split IN OUT EDGE
mesh files: uint32 nv, uint32 nf, float64 vertices (nv x 3), int32 faces (nf x 3), little-endian
"""
from std.math import sqrt
from std.memory import bitcast
from std.sys import argv, exit, stderr


struct Mesh(Movable):
    var v: List[Float64]  # x y z per vertex
    var f: List[Int]      # a b c per triangle

    def __init__(out self):
        self.v = List[Float64]()
        self.f = List[Int]()


def sphere(n: Int) -> Mesh:
    """Unit icosphere, n subdivisions, outward winding: the same vertex and face order as assets.py."""
    var t = (1.0 + sqrt(5.0)) / 2.0
    var base: List[Float64] = [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1]
    var m = Mesh()
    for i in range(12):
        var x = base[3 * i]
        var y = base[3 * i + 1]
        var z = base[3 * i + 2]
        var l = sqrt(x * x + y * y + z * z)
        m.v.append(x / l)
        m.v.append(y / l)
        m.v.append(z / l)
    m.f = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
           3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]
    for _ in range(n):
        var mid = Dict[Int, Int]()
        var g = List[Int](capacity=4 * len(m.f))
        for q in range(0, len(m.f), 3):
            var a = m.f[q]
            var b = m.f[q + 1]
            var c = m.f[q + 2]
            var ab = middle(m, mid, a, b)
            var bc = middle(m, mid, b, c)
            var ca = middle(m, mid, c, a)
            for e in [a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]:
                g.append(e)
        m.f = g^
    return m^


def middle(mut m: Mesh, mut mid: Dict[Int, Int], a: Int, b: Int) -> Int:
    """Index of the normalised midpoint of edge (a, b), made once per edge."""
    var k = min(a, b) * 1048576 + max(a, b)
    var got = mid.get(k)
    if got:
        return got.value()
    var x = m.v[3 * a] + m.v[3 * b]
    var y = m.v[3 * a + 1] + m.v[3 * b + 1]
    var z = m.v[3 * a + 2] + m.v[3 * b + 2]
    var l = sqrt(x * x + y * y + z * z)
    m.v.append(x / l)
    m.v.append(y / l)
    m.v.append(z / l)
    var i = len(m.v) // 3 - 1
    mid[k] = i
    return i


def primitive(box: Bool, sx: Float64, sy: Float64, sz: Float64) -> Mesh:
    """A MuJoCo box (half sizes) or ellipsoid (radii) in its geom frame."""
    if not box:
        var s = sphere(2)
        for i in range(len(s.v) // 3):
            s.v[3 * i] *= sx
            s.v[3 * i + 1] *= sy
            s.v[3 * i + 2] *= sz
        return s^
    var m = Mesh()
    for x in [-1.0, 1.0]:
        for y in [-1.0, 1.0]:
            for z in [-1.0, 1.0]:
                m.v.append(x * sx)
                m.v.append(y * sy)
                m.v.append(z * sz)
    m.f = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    return m^


def side(m: Mesh, a: Int, b: Int) -> Float64:
    var x = m.v[3 * a] - m.v[3 * b]
    var y = m.v[3 * a + 1] - m.v[3 * b + 1]
    var z = m.v[3 * a + 2] - m.v[3 * b + 2]
    return sqrt(x * x + y * y + z * z)


def later(n: Float64, i: Int, j: Int, k: Int, bn: Float64, bi: Int, bj: Int, bk: Int) -> Bool:
    """(n, i, j, k) > (bn, bi, bj, bk) as Python compares tuples: the tie order of assets.py's max(sides)."""
    if n != bn:
        return n > bn
    if i != bi:
        return i > bi
    if j != bj:
        return j > bj
    return k > bk


def split(var m: Mesh, edge: Float64) -> Mesh:
    """Halve the longest side until every side is under `edge`; the stack runs in assets.py's order (pop the end,
    push the two halves), so vertices and faces come out in the same order."""
    var todo = m.f.copy()
    var out = List[Int](capacity=len(m.f))
    while len(todo) > 0:
        var c = todo.pop()
        var b = todo.pop()
        var a = todo.pop()
        var n = side(m, a, b)
        var i = a
        var j = b
        var k = c
        var n2 = side(m, b, c)
        if later(n2, b, c, a, n, i, j, k):
            n = n2
            i = b
            j = c
            k = a
        var n3 = side(m, c, a)
        if later(n3, c, a, b, n, i, j, k):
            n = n3
            i = c
            j = a
            k = b
        if n < edge:
            out.append(a)
            out.append(b)
            out.append(c)
            continue
        for d in range(3):
            m.v.append((m.v[3 * i + d] + m.v[3 * j + d]) / 2.0)
        var mi = len(m.v) // 3 - 1
        for e in [i, mi, k, mi, j, k]:
            todo.append(e)
    m.f = out^
    return m^


def u32(b: List[UInt8], o: Int) -> Int:
    return Int(b[o]) | (Int(b[o + 1]) << 8) | (Int(b[o + 2]) << 16) | (Int(b[o + 3]) << 24)


def load(path: String) raises -> Mesh:
    var b = open(path, "r").read_bytes()
    var nv = u32(b, 0)
    var nf = u32(b, 4)
    var m = Mesh()
    for q in range(3 * nv):
        var w = UInt64(0)
        for s in range(8):
            w |= UInt64(b[8 + 8 * q + s]) << UInt64(8 * s)
        m.v.append(bitcast[DType.float64](w))
    var o = 8 + 24 * nv
    for q in range(3 * nf):
        m.f.append(Int(Int32(u32(b, o + 4 * q))))
    return m^


def save(m: Mesh, path: String) raises:
    var b = List[UInt8](capacity=8 + 8 * len(m.v) + 4 * len(m.f))
    for x in [len(m.v) // 3, len(m.f) // 3]:
        for s in range(4):
            b.append(UInt8((x >> (8 * s)) & 255))
    for x in m.v:
        var w = bitcast[DType.uint64](x)
        for s in range(8):
            b.append(UInt8(Int((w >> UInt64(8 * s)) & 255)))
    for x in m.f:
        for s in range(4):
            b.append(UInt8((x >> (8 * s)) & 255))
    with open(path, "w") as f:
        f.write_bytes(Span(b))


def main() raises:
    var a = argv()
    try:
        if len(a) == 4 and a[1] == "sphere":
            save(sphere(Int(String(a[2]))), String(a[3]))
        elif len(a) == 7 and a[1] == "primitive":
            save(primitive(String(a[2]) == "1", Float64(String(a[3])), Float64(String(a[4])), Float64(String(a[5]))), String(a[6]))
        elif len(a) == 5 and a[1] == "split":
            save(split(load(String(a[2])), Float64(String(a[4]))), String(a[3]))
        else:
            print("usage: assets sphere N OUT | assets primitive BOX SX SY SZ OUT | assets split IN OUT EDGE", file=stderr)
            exit(2)
    except e:
        print("FAIL assets:", e, file=stderr)
        exit(1)
