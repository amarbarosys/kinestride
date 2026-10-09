"""GPU timing and resources of convex.mojo's narrowphase, one thread per case, on convex_check.py's case files:
results go to DIR/outgpu.f64 (compared with the CPU out.f64 by convex_check.py gpu), time per launch is printed. Same collide() as the CPU driver.

usage: gpu-wait run --priority 60 --timeout 300 -- convex_gpu DIR/   (build: -DNCASES=n -DNHV=.. -DNHF=.. -DNHE=.. from convex_check.py)"""
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from convex import *

comptime CW = 40
comptime OW = 36
comptime N = get_defined_int["NCASES"]()
comptime NHV = get_defined_int["NHV"]()
comptime NHF = get_defined_int["NHF"]()
comptime NHE = get_defined_int["NHE"]()
comptime NHM = get_defined_int["NHM"]()
comptime LC = row_major[N * CW]()
comptime LO = row_major[N * OW]()
comptime LV = row_major[3 * NHV]()
comptime LF = row_major[NHF]()
comptime LE = row_major[NHE]()
comptime LM = row_major[3 * NHM]()
comptime TC = TileTensor[T, type_of(LC), MutAnyOrigin]
comptime TO = TileTensor[T, type_of(LO), MutAnyOrigin]
comptime TV = TileTensor[DType.float32, type_of(LV), MutAnyOrigin]
comptime TF = TileTensor[DType.int32, type_of(LF), MutAnyOrigin]
comptime TE = TileTensor[DType.int32, type_of(LE), MutAnyOrigin]
comptime TM = TileTensor[DType.int32, type_of(LM), MutAnyOrigin]


@always_inline
def v3(c: TC, k: Int) -> V:
    return V(S(c[k]), S(c[k + 1]), S(c[k + 2]), S(0))


def kernel(cases: TC, hv: TV, hf: TF, he: TE, hm: TM, res: TO):
    var c = Int(block_idx.x * block_dim.x + thread_idx.x)
    if c >= N:
        return
    var b = c * CW
    var h = Int(cases[b + 29])
    var pv = hv.ptr + 3 * Int(hm[3 * h])
    var pf = hf.ptr + Int(hm[3 * h + 1])
    var pe = he.ptr
    var o1 = Sp(pv, pf, pe)
    var o2 = Sp(pv, pf, pe)
    o1.kind = Int(cases[b])
    o2.kind = Int(cases[b + 1])
    o1.size = v3(cases, b + 2)
    o1.pos = v3(cases, b + 5)
    o1.r0 = v3(cases, b + 8)
    o1.r1 = v3(cases, b + 11)
    o1.r2 = v3(cases, b + 14)
    o2.pos = v3(cases, b + 17)
    o2.r0 = v3(cases, b + 20)
    o2.r1 = v3(cases, b + 23)
    o2.r2 = v3(cases, b + 26)
    var cs = collide(o1, o2, S(cases[b + 30]), S(cases[b + 31]), S(cases[b + 32]))
    var r = c * OW
    for k in range(OW):
        res[r + k] = 0
    res[r] = Scalar[T](cs.n)
    for i in range(cs.n):
        res[r + 1 + 7 * i] = cs.dist[i]
        for k in range(3):
            res[r + 2 + 7 * i + k] = cs.pos[i][k]
            res[r + 5 + 7 * i + k] = cs.normal[i][k]


def main() raises:
    var dir = String(argv()[1])
    var ctx = DeviceContext(0)
    var cases = List[Scalar[T]](length=N * CW, fill=0)
    var raw = open(dir + "cases.f64", "r").read_bytes()
    var p = raw.unsafe_ptr().unsafe_bitcast[Float64]()
    for i in range(N * CW):
        cases[i] = Scalar[T](p[i])
    var hvr = open(dir + "hverts.f32", "r").read_bytes()
    var hfr = open(dir + "hfirst.i32", "r").read_bytes()
    var her = open(dir + "hedges.i32", "r").read_bytes()
    var hmr = open(dir + "hmeta.i32", "r").read_bytes()
    var c_b = ctx.enqueue_create_buffer[T](N * CW)
    var o_b = ctx.enqueue_create_buffer[T](N * OW)
    var v_b = ctx.enqueue_create_buffer[DType.float32](3 * NHV)
    var f_b = ctx.enqueue_create_buffer[DType.int32](NHF)
    var e_b = ctx.enqueue_create_buffer[DType.int32](NHE)
    var m_b = ctx.enqueue_create_buffer[DType.int32](3 * NHM)
    ctx.enqueue_copy(c_b, cases.unsafe_ptr())
    ctx.enqueue_copy(v_b, hvr.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(f_b, hfr.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(e_b, her.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(m_b, hmr.unsafe_ptr().unsafe_bitcast[Int32]())
    var k = ctx.compile_function[kernel]()
    var grid = (N + 63) // 64
    ctx.enqueue_function(k, TileTensor(c_b, LC), TileTensor(v_b, LV), TileTensor(f_b, LF), TileTensor(e_b, LE), TileTensor(m_b, LM), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(20):
        ctx.enqueue_function(k, TileTensor(c_b, LC), TileTensor(v_b, LV), TileTensor(f_b, LF), TileTensor(e_b, LE), TileTensor(m_b, LM), TileTensor(o_b, LO), grid_dim=grid, block_dim=64)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 20 / 1000
    var host = List[Scalar[T]](length=N * OW, fill=0)
    ctx.enqueue_copy(host.unsafe_ptr(), o_b)
    ctx.synchronize()
    var out = List[Float64](length=N * OW, fill=0)
    for i in range(N * OW):
        out[i] = Float64(host[i])
    with open(dir + "outgpu.f64", "w") as f:
        f.write_bytes(Span(unsafe_ptr=out.unsafe_ptr().unsafe_bitcast[UInt8](), length=N * OW * 8))
    print(N, "cases:", us, "us per launch,", us * 1000 / Float64(N), "ns per case")
