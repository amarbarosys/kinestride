"""Live brain view: every neuron a point at its place, coloured by class, lit by its activity, seen through an orbit
camera, drawn into a BW x BH RGBA image (the engine GUI shows it with imkit's Ui.image). Dataset-agnostic: a layout dir
(fly-connectome agent/body_engine/brain_layout.py writes one per brain) and an activity byte per neuron are all it reads.

  DIR/xyz.bin    float32 N x 3, centred, max |coord| = 1 (x lateral, y up, z front-back)
  DIR/class.bin  uint8 N: palette index; 255 = not drawn
  activity       uint8 N per frame: 0 at rest .. 255 just spiked (the brain worker's tap)

Two kernels, nothing allocated per frame: splat (one thread per neuron, additive RGB into three u32 per pixel, atomics;
a lit neuron also lights its 4 neighbours) and tone (one thread per pixel: 1 - exp(-k acc), RGBA out, acc cleared for
the next frame).

    var bv = BrainView(ctx, "DIR/")
    bv.draw(ctx, act_host_ptr, yaw, pitch, zoom)      # -> bv.img (device, BW x BH RGBA)
"""
from std.atomic import Atomic, Ordering
from std.math import cos, exp, sin
from std.sys import get_defined_int
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major

comptime BW = get_defined_int["BRAIN_W", 360]()
comptime BH = get_defined_int["BRAIN_H", 270]()
comptime NMAX = get_defined_int["BRAIN_N", 262144]()  # neuron capacity (FlyWire 138,639; MCNS fits)
comptime NCLS = 11
comptime TXYZ = TileTensor[DType.float32, type_of(row_major[3 * NMAX]()), MutAnyOrigin]
comptime TB8 = TileTensor[DType.uint8, type_of(row_major[NMAX]()), MutAnyOrigin]
comptime TACC = TileTensor[DType.uint32, type_of(row_major[3 * BW * BH]()), MutAnyOrigin]
comptime TIMG = TileTensor[DType.uint32, type_of(row_major[BW * BH]()), MutAnyOrigin]
comptime TCAM = TileTensor[DType.float32, type_of(row_major[8]()), MutAnyOrigin]
comptime REST = 3  # weight of a resting neuron: the brain's shape stays visible
comptime LIT = 40  # extra weight per 255 of activity


@always_inline
def palette(k: Int) -> SIMD[DType.uint32, 4]:
    """r, g, b (0..255) of class k: brain_layout.py's CLASSES order (optic, central, sensory, visual_projection,
    ascending, descending, sensory_ascending, visual_centrifugal, motor, endocrine, unknown)."""
    if k == 0:
        return SIMD[DType.uint32, 4](70, 110, 220, 0)
    if k == 1:
        return SIMD[DType.uint32, 4](60, 200, 190, 0)
    if k == 2:
        return SIMD[DType.uint32, 4](120, 220, 90, 0)
    if k == 3:
        return SIMD[DType.uint32, 4](80, 190, 255, 0)
    if k == 4:
        return SIMD[DType.uint32, 4](255, 170, 60, 0)
    if k == 5:
        return SIMD[DType.uint32, 4](255, 80, 120, 0)
    if k == 6:
        return SIMD[DType.uint32, 4](200, 230, 80, 0)
    if k == 7:
        return SIMD[DType.uint32, 4](170, 110, 255, 0)
    if k == 8:
        return SIMD[DType.uint32, 4](255, 60, 50, 0)
    if k == 9:
        return SIMD[DType.uint32, 4](255, 215, 70, 0)
    return SIMD[DType.uint32, 4](150, 150, 150, 0)


@always_inline
def project(x: Float32, y: Float32, z: Float32, cy: Float32, sy: Float32, cp: Float32, sp: Float32, zoom: Float32) -> Tuple[Int, Int]:
    """Pixel of a layout point (kernel and host): yaw (cos cy, sin sy) about y, pitch (cp, sp) about x, orthographic,
    zoom = pixels per layout unit."""
    var x1 = x * cy + z * sy
    var z1 = -x * sy + z * cy
    var y2 = y * cp - z1 * sp
    return (Int(Float32(BW) * 0.5 + x1 * zoom), Int(Float32(BH) * 0.5 - y2 * zoom))


def splat(xyz: TXYZ, cls: TB8, act: TB8, acc: TACC, cam: TCAM, n: Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(n) or Int(cls[i]) == 255:
        return
    var p = project(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], cam[0], cam[1], cam[2], cam[3], cam[4])
    var a = UInt32(act[i])
    var w = UInt32(REST) + a * UInt32(LIT) // 255
    var c = palette(Int(cls[i]))
    for k in range(5 if a > 32 else 1):  # a lit neuron is a small cross: centre full, neighbours half
        var px = p[0] + (1 if k == 1 else (-1 if k == 2 else 0))
        var py = p[1] + (1 if k == 3 else (-1 if k == 4 else 0))
        if px < 0 or px >= BW or py < 0 or py >= BH:
            continue
        var wk = w if k == 0 else w // 2
        var o = 3 * (py * BW + px)
        for ch in range(3):
            _ = Atomic.fetch_add[ordering=Ordering.RELAXED](acc.ptr.unsafe_offset(o + ch), c[ch] * wk)


def tone(acc: TACC, img: TIMG, gain: Float32):
    """1 - exp(-gain acc) per channel over a dark background, RGBA (alpha 255); clears acc for the next splat."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= BW * BH:
        return
    var out = UInt32(255) << 24
    for ch in range(3):
        var v = 1 - exp(-gain * Float32(acc[3 * i + ch]))
        var bg = Float32(14 if ch < 2 else 20)
        out |= UInt32(bg + v * (255 - bg)) << UInt32(8 * ch)
        acc[3 * i + ch] = 0
    img[i] = out


struct BrainView(Movable):
    var n: Int
    var xyz: DeviceBuffer[DType.float32]
    var cls: DeviceBuffer[DType.uint8]
    var act: DeviceBuffer[DType.uint8]
    var acc: DeviceBuffer[DType.uint32]
    var cam: DeviceBuffer[DType.float32]
    var img: DeviceBuffer[DType.uint32]
    var host_xyz: List[Float32]  # kept for picking and labels (project on the host)

    def __init__(out self, ctx: DeviceContext, dir: String) raises:
        var xb = open(dir + "xyz.bin", "r").read_bytes()
        var cb = open(dir + "class.bin", "r").read_bytes()
        self.n = len(cb)
        if self.n > NMAX or len(xb) != 12 * self.n:
            raise Error("brain layout " + dir + ": " + String(self.n) + " neurons (capacity " + String(NMAX) + "), xyz " + String(len(xb)) + " B")
        self.host_xyz = List[Float32](length=3 * self.n, fill=0)
        var xp = xb.unsafe_ptr().unsafe_bitcast[Float32]()
        for k in range(3 * self.n):
            self.host_xyz[k] = xp[k]
        self.xyz = ctx.enqueue_create_buffer[DType.float32](3 * NMAX)
        self.cls = ctx.enqueue_create_buffer[DType.uint8](NMAX)
        self.act = ctx.enqueue_create_buffer[DType.uint8](NMAX)
        self.acc = ctx.enqueue_create_buffer[DType.uint32](3 * BW * BH)
        self.cam = ctx.enqueue_create_buffer[DType.float32](8)
        self.img = ctx.enqueue_create_buffer[DType.uint32](BW * BH)
        self.act.enqueue_fill(0)
        self.acc.enqueue_fill(0)
        ctx.enqueue_copy(self.xyz.create_sub_buffer[DType.float32](0, 3 * self.n), self.host_xyz.unsafe_ptr())
        ctx.enqueue_copy(self.cls.create_sub_buffer[DType.uint8](0, self.n), cb.unsafe_ptr())
        ctx.synchronize()

    def draw(mut self, ctx: DeviceContext, act: Int, yaw: Float32, pitch: Float32, zoom: Float32, gain: Float32 = 0.0004) raises:
        """Upload this frame's activity (host bytes at `act`, n of them; 0 = keep the last) and draw into img."""
        if act:
            ctx.enqueue_copy(self.act.create_sub_buffer[DType.uint8](0, self.n), UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=act))
        var cam = List[Float32](length=8, fill=0)
        cam[0], cam[1], cam[2], cam[3], cam[4] = cos(yaw), sin(yaw), cos(pitch), sin(pitch), zoom
        ctx.enqueue_copy(self.cam, cam.unsafe_ptr())
        ctx.enqueue_function(ctx.compile_function[splat](), TileTensor(self.xyz, row_major[3 * NMAX]()), TileTensor(self.cls, row_major[NMAX]()),
            TileTensor(self.act, row_major[NMAX]()), TileTensor(self.acc, row_major[3 * BW * BH]()), TileTensor(self.cam, row_major[8]()), Int32(self.n),
            grid_dim=(self.n + 255) // 256, block_dim=256)
        ctx.enqueue_function(ctx.compile_function[tone](), TileTensor(self.acc, row_major[3 * BW * BH]()), TileTensor(self.img, row_major[BW * BH]()),
            gain, grid_dim=(BW * BH + 255) // 256, block_dim=256)
        ctx.synchronize()  # cam (a host list) must outlive the copy
