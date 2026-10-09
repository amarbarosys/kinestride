"""Renderer step 2 (docs/RENDERER.md): does a 64-bit atomic min on global memory work from Mojo on
gfx1100, and how fast is it? N threads each write one hashed (depth << 32 | id) value into a 1920 x 1080
uint64 visibility buffer, the raster pass's access pattern; the result must equal the CPU's per-pixel min.

usage: proto_atomic N
"""
from std.atomic import Atomic, Ordering
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime W = 1920
comptime H = 1080
comptime LB = row_major[W * H]()
comptime TB = TileTensor[DType.uint64, type_of(LB), MutAnyOrigin]


@always_inline
def hash(x: UInt64) -> UInt64:
    var z = x * 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    return z ^ (z >> 27)


def write(buf: TB, n: Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(n):
        return
    var h = hash(UInt64(i))
    var pixel = Int(h % UInt64(W * H))
    var value = ((h >> 20) & 0xFFFFFFFF) << 32 | UInt64(i)
    _ = Atomic.min[ordering=Ordering.RELAXED](buf.ptr.unsafe_offset(pixel), value)


def main() raises:
    var n = Int(String(argv()[1]))
    var ctx = DeviceContext(0)
    var b = ctx.enqueue_create_buffer[DType.uint64](W * H)
    var host = List[UInt64](length=W * H, fill=UInt64.MAX)
    var k = ctx.compile_function[write]()
    ctx.enqueue_copy(b, host.unsafe_ptr())
    ctx.enqueue_function(k, TileTensor(b, LB), Int32(n), grid_dim=(n + 255) // 256, block_dim=256)
    ctx.synchronize()
    ctx.enqueue_copy(host.unsafe_ptr(), b)
    ctx.synchronize()
    var want = List[UInt64](length=W * H, fill=UInt64.MAX)
    for i in range(n):
        var h = hash(UInt64(i))
        var p = Int(h % UInt64(W * H))
        want[p] = min(want[p], ((h >> 20) & 0xFFFFFFFF) << 32 | UInt64(i))
    var bad = 0
    for p in range(W * H):
        bad += 1 if host[p] != want[p] else 0
    var t0 = perf_counter_ns()
    for _ in range(100):
        ctx.enqueue_function(k, TileTensor(b, LB), Int32(n), grid_dim=(n + 255) // 256, block_dim=256)
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / 100 / 1e6
    print(n, "64-bit atomic mins:", ms, "ms per pass,", Float64(n) / ms / 1e6, "G/s;", "PASS" if bad == 0 else "FAIL " + String(bad) + " pixels differ")
    if bad:
        raise Error("atomic min mismatch")
