"""Optimisation probe (docs/BODY_ENGINE_OPT.md step 2): is one wave per animal with the matrix in LDS
faster than one thread per animal through global memory, for MuJoCo's sparse L'DL (mj_factorI) of the fly's
74-dof mass matrix? Both variants factor the same matrices (64 MuJoCo samples, animal a uses sample a % 64)
and are checked against MuJoCo's qLD and qLDiagInv before their times count.

Variant T: thread per animal, M in global [row][animal] (the engine today).
Variant W: wave (32 lanes) per animal, M copied to LDS, each row's eliminations spread over lanes: row k's
updates touch pairs (j, c), c <= j < diag, all independent because a tree's row pattern is its ancestor chain.

usage: proto_factor DIR/   (DIR from the probe's export: pat.bin, M.bin, L64.bin)
"""
from std.math import sqrt
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from max.gpu import block_idx, block_dim, thread_idx, lane_id
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation

comptime NV = 74
comptime NC = 828
comptime F = get_defined_int["FLIES"]()
comptime LP = row_major[3 * NV + NC]()
comptime LM = row_major[(NC + NV) * F]()
comptime TP = TileTensor[DType.int32, type_of(LP), MutAnyOrigin]
comptime TM = TileTensor[DType.float32, type_of(LM), MutAnyOrigin]


def factor_thread(pat: TP, m: TM):
    """Variant T: rows [row][animal]; the engine's factorI."""
    var f = block_idx.x * block_dim.x + thread_idx.x
    if f >= F:
        return
    for k in range(NV - 1, -1, -1):
        var start = Int(pat[k])
        var diag = Int(pat[NV + k]) - 1
        var end = start + diag
        var invD = 1 / m[end * F + f]
        m[(NC + k) * F + f] = invD
        for adr in range(end - 1, start - 1, -1):
            var i = Int(pat[2 * NV + adr])
            var scl = -m[adr * F + f] * invD
            var ri = Int(pat[i])
            for c in range(Int(pat[NV + i])):
                m[(ri + c) * F + f] += m[(start + c) * F + f] * scl
        for c in range(diag):
            m[(start + c) * F + f] = m[(start + c) * F + f] * invD


comptime PAT_LDS = get_defined_int["PAT_LDS", 0]() == 1


@always_inline
def pv(pat: TP, P: type_of(stack_allocation[DType.int32, address_space=AddressSpace.SHARED](row_major[3 * NV + NC]())), k: Int) -> Int32:
    comptime if PAT_LDS:
        return P[k]
    return pat[k]


def factor_wave(pat: TP, m: TM):
    """Variant W: animal = block (one wave), rows [animal][row]; L in LDS."""
    var f = block_idx.x
    var lane = Int(thread_idx.x)
    var L = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[NC + NV]())
    var P = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](row_major[3 * NV + NC]())
    for c in range(lane, NC, 32):
        L[c] = m[f * (NC + NV) + c]
    comptime if PAT_LDS:
        for c in range(lane, 3 * NV + NC, 32):
            P[c] = pat[c]
    barrier()
    for k in range(NV - 1, -1, -1):
        var start = Int(pv(pat, P, k))
        var diag = Int(pv(pat, P, NV + k)) - 1
        var invD = 1 / L[start + diag]
        # pair t -> (j, c): row j of the chain (ancestor colind[start + j]) gets L(k, c) * scl_j, c <= j
        for t in range(lane, diag * (diag + 1) // 2, 32):
            var j = Int((sqrt(Float32(8 * t + 1)) - 1) / 2)
            if (j + 1) * (j + 2) // 2 <= t:  # float rounding of the triangular root
                j += 1
            var c = t - j * (j + 1) // 2
            var i = Int(pv(pat, P, 2 * NV + start + j))
            L[Int(pv(pat, P, i)) + c] += L[start + c] * (-L[start + j] * invD)
        barrier()
        for c in range(lane, diag, 32):
            L[start + c] *= invD
        if lane == 0:
            L[NC + k] = invD
        barrier()
    for c in range(lane, NC + NV, 32):
        m[f * (NC + NV) + c] = L[c]


comptime G = get_defined_int["G", 4]()  # lanes per animal in factor_group; 32 / G animals per wave


def factor_group(pat: TP, m: TM):
    """Variant G: G lanes per animal, 32 / G animals per wave, every animal's L in LDS; all animals eliminate
    the same row together (one model), so one barrier per phase serves the whole wave."""
    comptime A = 32 // G
    var lane = Int(thread_idx.x)
    var slot = lane // G
    var sub = lane % G
    var f = Int(block_idx.x) * A + slot
    var L = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[A * (NC + NV)]())
    var base = slot * (NC + NV)
    for c in range(sub, NC, G):
        L[base + c] = m[f * (NC + NV) + c]
    barrier()
    for k in range(NV - 1, -1, -1):
        var start = Int(pat[k])
        var diag = Int(pat[NV + k]) - 1
        var invD = 1 / L[base + start + diag]
        for t in range(sub, diag * (diag + 1) // 2, G):
            var j = Int((sqrt(Float32(8 * t + 1)) - 1) / 2)
            if (j + 1) * (j + 2) // 2 <= t:
                j += 1
            var c = t - j * (j + 1) // 2
            var i = Int(pat[2 * NV + start + j])
            L[base + Int(pat[i]) + c] += L[base + start + c] * (-L[base + start + j] * invD)
        barrier()
        for c in range(sub, diag, G):
            L[base + start + c] *= invD
        if sub == 0:
            L[base + NC + k] = invD
        barrier()
    for c in range(sub, NC + NV, G):
        m[f * (NC + NV) + c] = L[base + c]


def load(path: String) raises -> List[UInt8]:
    return open(path, "r").read_bytes()


def main() raises:
    var dir = String(argv()[1])
    var pat = load(dir + "pat.bin")
    var ms = load(dir + "M.bin")
    var want = load(dir + "L64.bin")
    var mv = ms.unsafe_ptr().unsafe_bitcast[Float32]()
    var wv = want.unsafe_ptr().unsafe_bitcast[Float64]()
    var ctx = DeviceContext(0)
    var p_b = ctx.enqueue_create_buffer[DType.int32](3 * NV + NC)
    ctx.enqueue_copy(p_b, pat.unsafe_ptr().unsafe_bitcast[Int32]())
    var m_b = ctx.enqueue_create_buffer[DType.float32]((NC + NV) * F)
    var host = List[Float32](length=(NC + NV) * F, fill=0)
    for variant in range(3):
        for a in range(F):  # T: [row][animal]; W: [animal][row]
            for c in range(NC):
                host[c * F + a if variant == 0 else a * (NC + NV) + c] = mv[(a % 64) * NC + c]
        ctx.enqueue_copy(m_b, host.unsafe_ptr())
        var t = TileTensor(m_b, LM)
        var tp = TileTensor(p_b, LP)
        var kt = ctx.compile_function[factor_thread]()
        var kw = ctx.compile_function[factor_wave]()
        var kg = ctx.compile_function[factor_group]()
        if variant == 0:
            ctx.enqueue_function(kt, tp, t, grid_dim=(F + 63) // 64, block_dim=64)
        elif variant == 1:
            ctx.enqueue_function(kw, tp, t, grid_dim=F, block_dim=32)
        else:
            ctx.enqueue_function(kg, tp, t, grid_dim=F * G // 32, block_dim=32)
        ctx.synchronize()
        ctx.enqueue_copy(host.unsafe_ptr(), m_b)
        ctx.synchronize()
        var err: Float64 = 0
        var scale: Float64 = 0
        for a in range(64):
            for c in range(NC + NV):
                var got = Float64(host[c * F + a if variant == 0 else a * (NC + NV) + c])
                err = max(err, abs(got - wv[a * (NC + NV) + c]))
                scale = max(scale, abs(wv[a * (NC + NV) + c]))
        # time 100 factorizations of the (now factored) matrices: same work per call, values irrelevant
        var t0 = perf_counter_ns()
        for _ in range(100):
            if variant == 0:
                ctx.enqueue_function(kt, tp, t, grid_dim=(F + 63) // 64, block_dim=64)
            elif variant == 1:
                ctx.enqueue_function(kw, tp, t, grid_dim=F, block_dim=32)
            else:
                ctx.enqueue_function(kg, tp, t, grid_dim=F * G // 32, block_dim=32)
        ctx.synchronize()
        var us = Float64(perf_counter_ns() - t0) / 100 / 1000
        print("thread per animal" if variant == 0 else ("wave per animal  " if variant == 1 else "G=" + String(G) + " lanes/animal "), F, "animals:", us, "us per factorization; max rel err vs MuJoCo", err / scale)
