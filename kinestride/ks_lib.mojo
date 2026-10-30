"""Kinestride as a C shared library (docs/plans/GODOT_SPIKE.md).
G0: `ks_ping` proves the Mojo runtime and a GPU launch work inside a host that is not Mojo (a C harness, then Godot): out[i] = i on the device, summed on the host.
build: mojo build --emit shared-lib kinestride/ks_lib.mojo -o .work/godot/libkinestride.so"""
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime N = 1024
comptime L = row_major[N]()
comptime T = TileTensor[DType.float32, type_of(L), MutAnyOrigin]


def fill(o: T):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < N:
        o[i] = Float32(i)


@export
def ks_ping() abi("C") -> Int:
    """Sum of out[i] = i over N GPU threads: N(N-1)/2 = 523776, or -1 on any error."""
    try:
        var ctx = DeviceContext()
        var d = ctx.enqueue_create_buffer[DType.float32](N)
        var k = ctx.compile_function[fill]()
        ctx.enqueue_function(k, TileTensor(d, L), grid_dim=N // 64, block_dim=64)
        var h = List[Float32](length=N, fill=0)
        ctx.enqueue_copy(h.unsafe_ptr(), d)
        ctx.synchronize()
        var s = 0
        for i in range(N):
            s += Int(h[i])
        return s
    except:
        return -1
