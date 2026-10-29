"""Live viewer core: the body engine and the renderer in one process and one DeviceContext. `Viewer` owns the engine's
buffers (smooth.main's), the renderer's (render.main's) and their kernels; the poses go from the engine's output rows to
the renderer without a copy. An app keeps its own loop and calls `step` and `frame` around its own work (a brain writing
`u_b`, sensors reading `q_b`/`o_b`, overlays drawing into `img_b`); `view.mojo` is the loop with no brain. Kernel handles
cannot be struct fields (their type carries AnyOrigin): the app compiles them once and passes them to `step` and `frame`.

DIR holds the engine files (mi.bin, mf.bin, qpos.bin, qvel.bin, ctrl.bin) and the mesh tables (rmi.bin, rmf.bin, rv.bin,
rt.bin, rp.bin; model.meshes). Build with both size files and the frame size:
  mojo build -I kinestride $(cat DIR/sizes.txt) $(cat DIR/rsizes.txt) -DFLIES=F -DSTAGE=7 -DK=k -DW=w -DH=h -DPROWS=7*(NBODY-1) ...
"""
import smooth as eng
import render as ren
from max.algorithm import parallelize
from std.ffi import external_call
from std.math import tan
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor


def write_all(fd: Int, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    """write(2) until n bytes left; False when the reader is gone."""
    var left = n
    var q = p
    while left > 0:
        var w = external_call["__write", Int](fd, q.unsafe_bitcast[NoneType](), left)
        if w <= 0:
            return False
        q = q.unsafe_offset(w)
        left -= w
    return True


struct Viewer(Movable):
    # engine: smooth.main's buffers; the host lists back the -DCPU=1 step (one host thread per animal)
    var mi_b: DeviceBuffer[DType.int32]
    var mf_b: DeviceBuffer[DType.float32]
    var q_b: DeviceBuffer[DType.float32]
    var v_b: DeviceBuffer[DType.float32]
    var u_b: DeviceBuffer[DType.float32]
    var o_b: DeviceBuffer[DType.float32]
    var mi: List[UInt8]
    var mf: List[UInt8]
    var qp: List[UInt8]
    var qv: List[UInt8]
    var uv: List[UInt8]
    var ob: List[UInt8]
    # renderer: render.main's buffers and passes; the pose tensor is the engine's output (rows xpos, xquat of bodies 1..)
    var pi_b: DeviceBuffer[DType.int32]
    var pf_b: DeviceBuffer[DType.float32]
    var rv_b: DeviceBuffer[DType.int16]
    var rt_b: DeviceBuffer[DType.uint16]
    var rp_b: DeviceBuffer[DType.uint8]
    var cam_b: DeviceBuffer[DType.float32]
    var vis_b: DeviceBuffer[DType.uint64]
    var sm_b: DeviceBuffer[DType.uint64]  # shadow map (render.mojo SHADOW=2)
    var scam_b: DeviceBuffer[DType.float32]
    var img_b: DeviceBuffer[DType.uint32]
    var low_b: DeviceBuffer[DType.uint32]
    var out_h: HostBuffer[DType.uint32]
    var camh: List[Float32]

    def __init__(out self, ctx: DeviceContext, dir: String, cam: List[Float32]) raises:
        """cam: eye 3, target 3, vertical field of view in degrees."""
        self.mi_b = ctx.enqueue_create_buffer[DType.int32](eng.NI)
        self.mf_b = ctx.enqueue_create_buffer[DType.float32](eng.NFL)
        self.q_b = ctx.enqueue_create_buffer[DType.float32](eng.NQ * eng.F)
        self.v_b = ctx.enqueue_create_buffer[DType.float32](eng.NV * eng.F)
        self.u_b = ctx.enqueue_create_buffer[DType.float32](max(eng.NU, 1) * eng.F)
        self.o_b = ctx.enqueue_create_buffer[DType.float32](eng.NROW * eng.F)
        self.o_b.enqueue_fill(0)  # warm start reads OWARM before the first step writes it
        self.mi = eng.load(dir + "mi.bin", eng.NI)
        self.mf = eng.load(dir + "mf.bin", eng.NFL)
        self.qp = eng.load(dir + "qpos.bin", eng.NQ * eng.F)
        self.qv = eng.load(dir + "qvel.bin", eng.NV * eng.F)
        self.uv = eng.load(dir + "ctrl.bin", eng.NU * eng.F)
        self.ob = List[UInt8](length=4 * eng.NROW * eng.F, fill=0)
        ctx.enqueue_copy(self.mi_b, self.mi.unsafe_ptr().unsafe_bitcast[Int32]())
        ctx.enqueue_copy(self.mf_b, self.mf.unsafe_ptr().unsafe_bitcast[Float32]())
        ctx.enqueue_copy(self.q_b, self.qp.unsafe_ptr().unsafe_bitcast[Float32]())
        ctx.enqueue_copy(self.v_b, self.qv.unsafe_ptr().unsafe_bitcast[Float32]())
        ctx.enqueue_copy(self.u_b, self.uv.unsafe_ptr().unsafe_bitcast[Float32]())
        var pi_h = ren.load(dir + "rmi.bin", 20 * ren.NPART)
        var pf_h = ren.load(dir + "rmf.bin", 4 * ren.PFS * ren.NPART)
        var rv_h = ren.load(dir + "rv.bin", 2 * ren.RVI * ren.NRV)
        var rt_h = ren.load(dir + "rt.bin", 2 * ren.NRT)
        var rp_h = ren.load(dir + "rp.bin", ren.NTRI)
        self.pi_b = ctx.enqueue_create_buffer[DType.int32](5 * ren.NPART)
        self.pf_b = ctx.enqueue_create_buffer[DType.float32](ren.PFS * ren.NPART)
        self.rv_b = ctx.enqueue_create_buffer[DType.int16](ren.RVI * ren.NRV)
        self.rt_b = ctx.enqueue_create_buffer[DType.uint16](ren.NRT)
        self.rp_b = ctx.enqueue_create_buffer[DType.uint8](ren.NTRI)
        self.cam_b = ctx.enqueue_create_buffer[DType.float32](ren.NCAM)
        self.vis_b = ctx.enqueue_create_buffer[DType.uint64](ren.W * ren.H)
        self.sm_b = ctx.enqueue_create_buffer[DType.uint64](ren.NSM)
        self.scam_b = ctx.enqueue_create_buffer[DType.float32](ren.NCAM)
        self.img_b = ctx.enqueue_create_buffer[DType.uint32](ren.W * ren.H)
        self.low_b = ctx.enqueue_create_buffer[DType.uint32](ren.HW * ren.HH)
        self.out_h = ctx.enqueue_create_host_buffer[DType.uint32](ren.W * ren.H)
        self.camh = List[Float32](length=ren.NCAM, fill=0)
        for k in range(6):
            self.camh[k] = cam[k]
        var fovy = cam[6] * 3.14159265 / 180
        self.camh[6] = fovy
        self.camh[7] = Float32(ren.H) * 0.5 / tan(fovy * 0.5)
        comptime if eng.HFG >= 0:  # the renderer draws the engine's own terrain
            var mfp = self.mf.unsafe_ptr().unsafe_bitcast[Float32]()
            for k in range(eng.NP):
                self.camh[16 + k] = mfp[eng.HFO + k]
        ctx.enqueue_copy(self.pi_b, pi_h.unsafe_ptr().unsafe_bitcast[Int32]())
        ctx.enqueue_copy(self.pf_b, pf_h.unsafe_ptr().unsafe_bitcast[Float32]())
        ctx.enqueue_copy(self.rv_b, rv_h.unsafe_ptr().unsafe_bitcast[Int16]())
        ctx.enqueue_copy(self.rt_b, rt_h.unsafe_ptr().unsafe_bitcast[UInt16]())
        ctx.enqueue_copy(self.rp_b, rp_h.unsafe_ptr().unsafe_bitcast[UInt8]())
        ctx.enqueue_copy(self.cam_b, self.camh.unsafe_ptr())
        ctx.synchronize()  # the loads above are host lists that go out of scope

    def step(mut self, ctx: DeviceContext, k: type_of(DeviceContext(0).compile_function[eng.smooth]())) raises:
        """One engine launch (K steps), state kept on the card. -DCPU=1: the same step on the host over the lists, one thread
        per animal (each writes only its own columns, as the kernel's threads do); the caller refreshes the device rows it reads."""
        comptime if eng.CPU:
            var hi = eng.host[DType.int32](self.mi, eng.LI); var hf = eng.host[DType.float32](self.mf, eng.LF)
            var hq = eng.host[DType.float32](self.qp, eng.LQ); var hv = eng.host[DType.float32](self.qv, eng.LV)
            var hu = eng.host[DType.float32](self.uv, eng.LU); var ho = eng.host[DType.float32](self.ob, eng.LO)
            def step_animal(f: Int) {imm hi, imm hf, imm hq, imm hv, imm hu, imm ho}:
                eng.smooth_at(hi, hf, hq, hv, hu, ho, f)
            parallelize(step_animal, eng.F)
        else:
            ctx.enqueue_function(k, TileTensor(self.mi_b, eng.LI), TileTensor(self.mf_b, eng.LF), TileTensor(self.q_b, eng.LQ),
                TileTensor(self.v_b, eng.LV), TileTensor(self.u_b, eng.LU), TileTensor(self.o_b, eng.LO), grid_dim=eng.GRID, block_dim=eng.BLOCK)

    def frame(mut self, ctx: DeviceContext, kc: type_of(DeviceContext(0).compile_function[ren.clear]()), kr: type_of(DeviceContext(0).compile_function[ren.raster]()),
              ks: type_of(DeviceContext(0).compile_function[ren.shade]()), ku: type_of(DeviceContext(0).compile_function[ren.upscale]()), ksc: type_of(DeviceContext(0).compile_function[ren.suncam]()),
              kcs: type_of(DeviceContext(0).compile_function[ren.clear_sm]())) raises:
        """The renderer's passes over the engine's current poses into img_b (camera from cam_b)."""
        ren.frame(ctx, kc, kr, ks, ku, ksc, kcs, TileTensor(self.vis_b, ren.LVIS), TileTensor(self.sm_b, ren.LVIS),
            TileTensor(self.img_b, ren.LVIS), TileTensor(self.low_b, ren.LLOW), TileTensor(self.pi_b, ren.LPI), TileTensor(self.pf_b, ren.LPF), TileTensor(self.rv_b, ren.LRV),
            TileTensor(self.rt_b, ren.LRT), TileTensor(self.rp_b, ren.LRP), TileTensor(self.o_b, ren.LPOSE), TileTensor(self.cam_b, ren.LCAM),
            TileTensor(self.scam_b, ren.LCAM))
