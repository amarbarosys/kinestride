# How many elements does ctx.enqueue_copy(host_ptr, device_buffer) write? A device buffer of 2N floats set to 1
# is copied into a host list of 3N floats set to -1: PASS if exactly the first 2N become 1, i.e. the pointer form
# copies the whole source buffer, so a host list shorter than the buffer is overrun (smooth.mojo main, before rows()).
# Then the sub-buffer form copies exactly N. usage: mojo build probes/copy_size_probe.mojo; gpu-wait run -- ./probe
from max.gpu.host import DeviceContext

comptime N = 1 << 20


def main() raises:
    var ctx = DeviceContext()
    var d = ctx.enqueue_create_buffer[DType.float32](2 * N)
    d.enqueue_fill(1)
    var h = List[Float32](length=3 * N, fill=-1)
    ctx.enqueue_copy(h.unsafe_ptr(), d)
    ctx.synchronize()
    var ones = 0
    for i in range(3 * N):
        ones += 1 if h[i] == 1 else 0
    var g = List[Float32](length=3 * N, fill=-1)
    ctx.enqueue_copy(g.unsafe_ptr(), d.create_sub_buffer[DType.float32](0, N))
    ctx.synchronize()
    var sub = 0
    for i in range(3 * N):
        sub += 1 if g[i] == 1 else 0
    print("pointer form wrote", ones, "of", 2 * N, "source elements; sub-buffer form wrote", sub, "of N =", N)
    if ones != 2 * N or sub != N:
        print("FAIL copy size")
        raise Error("copy size")
    print("PASS copy size: the pointer form copies the whole source buffer")
