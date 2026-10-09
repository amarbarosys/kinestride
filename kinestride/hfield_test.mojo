"""CPU driver of convex.mojo's heightfield contact for hfield_check.py: DIR/hcases.f64 (HW doubles per case), hull tables,
DIR/hgrid.f32 (the MuJoCo hfield's samples), DIR/hmeta2.f64 (nrow, ncol, size0, size1, size2, size3, hfield origin z); writes DIR/hout.f64.

usage: hfield_test DIR/   (build with -DCCD64=1 for the double-precision parity run)"""
from std.sys import argv
from std.sys import get_defined_int
from convex import *
from world import PW, height_grad

comptime HF_FUNC = get_defined_int["HF_FUNC", 0]()  # 1: heights from world.mojo's float32 terrain function; 2: the same, rounded to the float32 elevation MuJoCo stores (exact origin and span); 0: MuJoCo's table

comptime HW = 20  # case: pos 3, mat rows 9, hull, margin, tol, rbound2 (hfield frame)
comptime HO = 1 + 7 * HF_SLOTS  # out: n, then per contact dist, pos 3, normal 3


struct Grid(Height):
    var data: List[Float32]
    var ncol: Int
    var size2: Float64
    var params: PW
    var lo: Float32
    var lo64: Float64
    var dx: Float32
    var size0: Float32
    var size1: Float32

    def __init__(out self, var data: List[Float32], ncol: Int, size2: Float64, params: PW, lo: Float32, lo64: Float64, dx: Float32, size0: Float32, size1: Float32):
        self.data = data^
        self.ncol = ncol
        self.size2 = size2
        self.params = params
        self.lo = lo
        self.lo64 = lo64
        self.dx = dx
        self.size0 = size0
        self.size1 = size1

    def h(self, r: Int, c: Int) -> S:
        comptime if HF_FUNC == 1:
            return S(height_grad(self.dx * Float32(c) - self.size0, self.dx * Float32(r) - self.size1, self.params)[0] - self.lo)
        comptime if HF_FUNC == 2:
            var hh = Float64(height_grad(self.dx * Float32(c) - self.size0, self.dx * Float32(r) - self.size1, self.params)[0])
            var e = Float32((hh - self.lo64) / self.size2)
            return S(Float64(e) * self.size2)
        return S(Float64(self.data[r * self.ncol + c]) * self.size2)


def read_bytes_f64(path: String) raises -> List[Float64]:
    var b = open(path, "r").read_bytes()
    var out = List[Float64](length=len(b) // 8, fill=0)
    var p = b.unsafe_ptr().unsafe_bitcast[Float64]()
    for i in range(len(out)):
        out[i] = p[i]
    return out^


def read_bytes_f32(path: String) raises -> List[Float32]:
    var b = open(path, "r").read_bytes()
    var out = List[Float32](length=len(b) // 4, fill=0)
    var p = b.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(len(out)):
        out[i] = p[i]
    return out^


def read_bytes_i32(path: String) raises -> List[Int32]:
    var b = open(path, "r").read_bytes()
    var out = List[Int32](length=len(b) // 4, fill=0)
    var p = b.unsafe_ptr().unsafe_bitcast[Int32]()
    for i in range(len(out)):
        out[i] = p[i]
    return out^


@always_inline
def v3(c: List[Float64], k: Int) -> V:
    return V(S(c[k]), S(c[k + 1]), S(c[k + 2]), S(0))


def main() raises:
    var dir = String(argv()[1])
    var cases = read_bytes_f64(dir + "hcases.f64")
    var meta = read_bytes_f64(dir + "hmeta2.f64")
    var hp = read_bytes_f32(dir + "hparams.f32")  # 16 terrain params, then the hfield origin z
    var tp = PW()
    for k in range(16):
        tp[k] = hp[k]
    var dx0 = Float32(2 * meta[2] / Float64(Int(meta[1]) - 1))
    var grid = Grid(read_bytes_f32(dir + "hgrid.f32"), Int(meta[1]), meta[4], tp, hp[16], meta[6], dx0, Float32(meta[2]), Float32(meta[3]))
    var hverts = read_bytes_f32(dir + "hverts.f32")
    var hfirst = read_bytes_i32(dir + "hfirst.i32")
    var hedges = read_bytes_i32(dir + "hedges.i32")
    var hmeta = read_bytes_i32(dir + "hmeta.i32")
    var n = len(cases) // HW
    var out = List[Float64](length=n * HO, fill=0)
    var only = Int(String(argv()[2])) if len(argv()) > 2 else -1
    for c in range(n):
        if only >= 0 and c != only:
            continue
        var b = c * HW
        var h = Int(cases[b + 12])
        var pv = hverts.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]() + 3 * Int(hmeta[3 * h])
        var pf = hfirst.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]() + Int(hmeta[3 * h + 1])
        var pe = hedges.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
        var mesh = Sp(pv, pf, pe)
        mesh.kind = MESH
        mesh.pos = v3(cases, b)
        mesh.r0 = v3(cases, b + 3)
        mesh.r1 = v3(cases, b + 6)
        mesh.r2 = v3(cases, b + 9)
        var cs = collide_hfield(grid, mesh, Int(meta[0]), Int(meta[1]), S(meta[2]), S(meta[3]), S(meta[4]), S(meta[5]),
                                S(cases[b + 13]), S(cases[b + 14]), S(cases[b + 15]), False, V(0))
        var r = c * HO
        out[r] = Float64(cs.n)
        for i in range(cs.n):
            out[r + 1 + 7 * i] = Float64(cs.dist[i])
            for k in range(3):
                out[r + 2 + 7 * i + k] = Float64(cs.pos[i][k])
                out[r + 5 + 7 * i + k] = Float64(cs.normal[i][k])
    with open(dir + "hout.f64", "w") as f:
        f.write_bytes(Span(unsafe_ptr=out.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(out) * 8))
