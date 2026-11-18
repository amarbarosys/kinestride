"""The world's ground as a function, not a table: height and gradient at any (x, y) from 8 parameters, so the engine
and the renderer evaluate the same endless terrain with no stored heights (private note WORLD). world.py is the
numpy twin and the check.

params (16): slope x, slope y, amplitude (mm), wavelength (mm), octaves, seed, lacunarity, gain,
              landmark density (0-1), landmark cell, cone radius, cone height (mm), 4 unused
height(x, y) = slope . (x, y) + sum_o amplitude gain^o noise(x f_o, y f_o, seed + o),  f_o = lacunarity^o / wavelength
               + a cone in each landmark cell whose hash falls under the density (jittered so it stays inside its
               cell: no neighbour lookup). Landmarks are ground: the engine collides with them, the renderer draws them.
noise: 2-D gradient noise, 8 gradient directions picked by an integer hash of (cell, seed), quintic fade.
Precision: positions are float32 world mm, so the terrain is exact to ~1e-3 mm out to ~10 m from the origin.

usage (check): world IN OUT N   IN = N x (x, y) float32 + 16 params, OUT = N x (h, gx, gy) float32
"""
from std.math import floor, sqrt
from std.sys import argv

comptime NP = 16
comptime PW = SIMD[DType.float32, NP]
comptime R = Float32(0.70710678)


@always_inline
def hash(ix: Int32, iy: Int32, seed: UInt32) -> UInt32:
    var h = ix.cast[DType.uint32]() * 0x8DA6B343 ^ iy.cast[DType.uint32]() * 0xD8163841 ^ seed * 0xCB1AB31F
    h ^= h >> 13
    h *= 0x85EBCA6B
    h ^= h >> 16
    return h


@always_inline
def grad(ix: Int32, iy: Int32, seed: UInt32) -> Tuple[Float32, Float32]:
    var k = Int(hash(ix, iy, seed) >> 29)  # 8 directions: axes and diagonals
    var gx = Float32(1) if k == 0 else (Float32(-1) if k == 1 else (R if k == 4 or k == 6 else (-R if k == 5 or k == 7 else 0)))
    var gy = Float32(1) if k == 2 else (Float32(-1) if k == 3 else (R if k == 4 or k == 7 else (-R if k == 5 or k == 6 else 0)))
    return (gx, gy)


@always_inline
def noise(x: Float32, y: Float32, seed: UInt32) -> SIMD[DType.float32, 4]:
    """(n, dn/dx, dn/dy, 0) of gradient noise at (x, y)."""
    var x0 = floor(x); var y0 = floor(y)
    var ix = Int32(Int(x0)); var iy = Int32(Int(y0))
    var fx = x - x0; var fy = y - y0
    var ga = grad(ix, iy, seed); var gb = grad(ix + 1, iy, seed)
    var gc = grad(ix, iy + 1, seed); var gd = grad(ix + 1, iy + 1, seed)
    var a = ga[0] * fx + ga[1] * fy
    var b = gb[0] * (fx - 1) + gb[1] * fy
    var c = gc[0] * fx + gc[1] * (fy - 1)
    var d = gd[0] * (fx - 1) + gd[1] * (fy - 1)
    var u = fx * fx * fx * (fx * (fx * 6 - 15) + 10); var du = 30 * fx * fx * (fx - 1) * (fx - 1)
    var v = fy * fy * fy * (fy * (fy * 6 - 15) + 10); var dv = 30 * fy * fy * (fy - 1) * (fy - 1)
    var k = a - b - c + d
    var n = a + (b - a) * u + (c - a) * v + k * u * v
    var kx = ga[0] - gb[0] - gc[0] + gd[0]; var ky = ga[1] - gb[1] - gc[1] + gd[1]
    var nx = ga[0] + (gb[0] - ga[0]) * u + (gc[0] - ga[0]) * v + kx * u * v + du * ((b - a) + k * v)
    var ny = ga[1] + (gb[1] - ga[1]) * u + (gc[1] - ga[1]) * v + ky * u * v + dv * ((c - a) + k * u)
    return SIMD[DType.float32, 4](n, nx, ny, 0)


@always_inline
def height_grad(x: Float32, y: Float32, p: PW) -> SIMD[DType.float32, 4]:
    """(height, dh/dx, dh/dy, 0) of the terrain at world (x, y) mm."""
    var res = SIMD[DType.float32, 4](p[0] * x + p[1] * y, p[0], p[1], 0)
    var amp = p[2]
    var f = 1 / p[3]
    for o in range(Int(p[4])):
        var n = noise(x * f, y * f, UInt32(Int(p[5])) + UInt32(o))
        res += SIMD[DType.float32, 4](amp * n[0], amp * f * n[1], amp * f * n[2], 0)
        amp *= p[7]
        f *= p[6]
    if p[8] > 0:  # landmark cone of this cell: present, centre and jitter from three hashes of (cell, seed)
        var cx = floor(x / p[9]); var cy = floor(y / p[9])
        var ix = Int32(Int(cx)); var iy = Int32(Int(cy))
        var s = UInt32(Int(p[5])) + 1000
        if Float32(hash(ix, iy, s) >> 8) * (1.0 / 16777216.0) < p[8]:
            var room = 0.5 * p[9] - p[10]
            var dx = x - ((cx + 0.5) * p[9] + room * (Float32(hash(ix, iy, s + 1) >> 8) * (2.0 / 16777216.0) - 1))
            var dy = y - ((cy + 0.5) * p[9] + room * (Float32(hash(ix, iy, s + 2) >> 8) * (2.0 / 16777216.0) - 1))
            var r = sqrt(dx * dx + dy * dy)
            if r < p[10] and r > 0:
                var k = p[11] / p[10]
                res += SIMD[DType.float32, 4](p[11] - k * r, -k * dx / r, -k * dy / r, 0)
    return res


def main() raises:
    var a = argv()
    var n = Int(String(a[3]))
    var raw = open(String(a[1]), "r").read_bytes()
    var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var p = PW()
    for k in range(NP):
        p[k] = src[2 * n + k]
    var res = List[Float32](length=3 * n, fill=0)
    for i in range(n):
        var r = height_grad(src[2 * i], src[2 * i + 1], p)
        res[3 * i] = r[0]; res[3 * i + 1] = r[1]; res[3 * i + 2] = r[2]
    with open(String(a[2]), "w") as f:
        f.write_bytes(Span(unsafe_ptr=res.unsafe_ptr().unsafe_bitcast[UInt8](), length=12 * n))
