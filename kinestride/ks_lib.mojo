"""Kinestride as a C shared library (docs/plans/GODOT_SPIKE.md).
G0: `ks_ping` proves the Mojo runtime and a GPU launch work inside a host that is not Mojo (a C harness, then Godot): out[i] = i on the device, summed on the host.
G1: one world per handle, the engine's buffers on the card (viewer.Viewer's engine half, no renderer):
ks_create(dir) -> handle (0 on error), ks_step(h) = one launch (K steps), ks_poses(h, out) copies the pose rows
(xpos 3 + xquat 4 of bodies 1.., [row][animal], ks_npose() floats), ks_set_ctrl(h, u) (NU x F), ks_destroy(h).
DIR is a model.export directory with qpos.bin, qvel.bin, ctrl.bin. The kernel handle cannot be a struct field
(its type carries AnyOrigin), so ks_step compiles it per call (cached by the context).
build: mojo build --emit shared-lib -I kinestride $(cat DIR/sizes.txt) -DFLIES=F -DSTAGE=7 -DK=k kinestride/ks_lib.mojo -o LIB.so"""
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from std.memory.alloc import alloc, Layout
import smooth as eng

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


comptime NPOSE = 7 * (eng.NB - 1) * eng.F
comptime U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


struct World(Movable):
    var ctx: DeviceContext
    var mi_b: DeviceBuffer[DType.int32]
    var mf_b: DeviceBuffer[DType.float32]
    var q_b: DeviceBuffer[DType.float32]
    var v_b: DeviceBuffer[DType.float32]
    var u_b: DeviceBuffer[DType.float32]
    var o_b: DeviceBuffer[DType.float32]
    var pose: List[Float32]  # host copy of the pose rows, refreshed by ks_fetch, read by ks_part

    def __init__(out self, dir: String) raises:
        self.pose = List[Float32](length=NPOSE, fill=0)
        self.ctx = DeviceContext(0)
        self.mi_b = self.ctx.enqueue_create_buffer[DType.int32](eng.NI)
        self.mf_b = self.ctx.enqueue_create_buffer[DType.float32](eng.NFL)
        self.q_b = self.ctx.enqueue_create_buffer[DType.float32](eng.NQ * eng.F)
        self.v_b = self.ctx.enqueue_create_buffer[DType.float32](eng.NV * eng.F)
        self.u_b = self.ctx.enqueue_create_buffer[DType.float32](max(eng.NU, 1) * eng.F)
        self.o_b = self.ctx.enqueue_create_buffer[DType.float32](eng.NROW * eng.F)
        self.o_b.enqueue_fill(0)  # warm start reads OWARM before the first step writes it
        var mi = eng.load(dir + "mi.bin", eng.NI)
        var mf = eng.load(dir + "mf.bin", eng.NFL)
        var qp = eng.load(dir + "qpos.bin", eng.NQ * eng.F)
        var qv = eng.load(dir + "qvel.bin", eng.NV * eng.F)
        var uv = eng.load(dir + "ctrl.bin", eng.NU * eng.F)
        self.ctx.enqueue_copy(self.mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
        self.ctx.enqueue_copy(self.mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
        self.ctx.enqueue_copy(self.q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
        self.ctx.enqueue_copy(self.v_b, qv.unsafe_ptr().unsafe_bitcast[Float32]())
        if eng.NU:
            self.ctx.enqueue_copy(self.u_b, uv.unsafe_ptr().unsafe_bitcast[Float32]())
        self.ctx.synchronize()  # the host lists go out of scope


def world(h: Int) -> MutPointer[World, MutUntrackedOrigin]:
    return MutPointer[World, MutUntrackedOrigin](unsafe_from_address=h)


@export
def ks_npose() abi("C") -> Int:
    return NPOSE


@export
def ks_nf() abi("C") -> Int:
    return eng.F


@export
def ks_create(dir: U8P) abi("C") -> Int:
    var n = 0
    while dir[unsafe_offset=n] != 0:
        n += 1
    var buf = List[UInt8](length=n, fill=0)
    for i in range(n):
        buf[i] = dir[unsafe_offset=i]
    try:
        var w = World(String(StringSlice(unsafe_from_utf8=Span(buf))))  # may raise: before the allocation exists
        var a = alloc(Layout[World](count=1))  # Mojo 1.2: write into the allocation, leak it to C as an address
        var p = a.unsafe_ptr()
        p.unsafe_write(w^)
        var h = Int(p)
        _ = a^.unsafe_leak()
        return h
    except e:
        print("FAIL ks_create:", e)
        return 0


@export
def ks_step(h: Int) abi("C") -> Int:
    try:
        ref w = world(h)[]
        var k = w.ctx.compile_function[eng.smooth]()
        w.ctx.enqueue_function(k, TileTensor(w.mi_b, eng.LI), TileTensor(w.mf_b, eng.LF), TileTensor(w.q_b, eng.LQ),
            TileTensor(w.v_b, eng.LV), TileTensor(w.u_b, eng.LU), TileTensor(w.o_b, eng.LO), grid_dim=eng.GRID, block_dim=eng.BLOCK)
        return 0
    except:
        return -1


@export
def ks_step_n(h: Int, n: Int) abi("C") -> Int:
    """n launches with one kernel lookup."""
    try:
        ref w = world(h)[]
        var k = w.ctx.compile_function[eng.smooth]()
        for _ in range(n):
            w.ctx.enqueue_function(k, TileTensor(w.mi_b, eng.LI), TileTensor(w.mf_b, eng.LF), TileTensor(w.q_b, eng.LQ),
                TileTensor(w.v_b, eng.LV), TileTensor(w.u_b, eng.LU), TileTensor(w.o_b, eng.LO), grid_dim=eng.GRID, block_dim=eng.BLOCK)
        return 0
    except:
        return -1


@export
def ks_poses(h: Int, dst: F32P) abi("C") -> Int:
    try:
        ref w = world(h)[]
        w.ctx.enqueue_copy(dst, w.o_b.create_sub_buffer[DType.float32](0, NPOSE))
        w.ctx.synchronize()
        return 0
    except:
        return -1


@export
def ks_fetch(h: Int) abi("C") -> Int:
    """One GPU -> host copy of the pose rows for this frame's ks_part calls."""
    return ks_poses(h, world(h)[].pose.unsafe_ptr())


@export
def ks_part(h: Int, p: Int, gq: F32P, dst: F32P) abi("C"):
    """Godot MultiMesh 3D buffer (12 floats per instance: basis rows with origin, [R | t]) of body p + 1 for every animal,
    the geom's local quaternion gq (w, x, y, z) applied, from the last ks_fetch. Rows: xpos of bodies 1.., then xquat."""
    comptime F = eng.F
    comptime QB = 3 * (eng.NB - 1)
    ref h_ = world(h)[].pose
    var gw = gq[unsafe_offset=0]; var gx = gq[unsafe_offset=1]; var gy = gq[unsafe_offset=2]; var gz = gq[unsafe_offset=3]
    for a in range(F):
        var bw = h_[(QB + 4 * p) * F + a]; var bx = h_[(QB + 4 * p + 1) * F + a]
        var by = h_[(QB + 4 * p + 2) * F + a]; var bz = h_[(QB + 4 * p + 3) * F + a]
        var w = bw * gw - bx * gx - by * gy - bz * gz  # body quat times geom quat
        var x = bw * gx + bx * gw + by * gz - bz * gy
        var y = bw * gy - bx * gz + by * gw + bz * gx
        var z = bw * gz + bx * gy - by * gx + bz * gw
        var o = dst.unsafe_offset(12 * a)
        o[0] = 1 - 2 * (y * y + z * z); o[1] = 2 * (x * y - w * z); o[2] = 2 * (x * z + w * y); o[3] = h_[(3 * p) * F + a]
        o[4] = 2 * (x * y + w * z); o[5] = 1 - 2 * (x * x + z * z); o[6] = 2 * (y * z - w * x); o[7] = h_[(3 * p + 1) * F + a]
        o[8] = 2 * (x * z - w * y); o[9] = 2 * (y * z + w * x); o[10] = 1 - 2 * (x * x + y * y); o[11] = h_[(3 * p + 2) * F + a]


@export
def ks_qpos(h: Int, dst: F32P) abi("C") -> Int:
    """qpos rows [NQ][F] (the gate's MuJoCo kinematics input)."""
    try:
        ref w = world(h)[]
        w.ctx.enqueue_copy(dst, w.q_b)
        w.ctx.synchronize()
        return 0
    except:
        return -1


@export
def ks_set_ctrl(h: Int, u: F32P) abi("C") -> Int:
    try:
        ref w = world(h)[]
        if eng.NU:
            w.ctx.enqueue_copy(w.u_b, u)
        return 0
    except:
        return -1


@export
def ks_destroy(h: Int) abi("C"):
    var p = world(h)
    _ = p.take_pointee()
    p.unsafe_free()
