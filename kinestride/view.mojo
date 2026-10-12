"""Watch any exported body with no brain: ctrl.bin held for every step, one engine launch (K steps) and one frame per
iteration, the RGBA frame to stdout; with OUT, the last frame to OUT and the state after it to OUT.qpos instead.

usage: view DIR/ FRAMES ex ey ez tx ty tz fovy_deg [OUT]      (FRAMES 0 = until the reader closes the pipe)
       view DIR/ 0 ... | mpv --demuxer=rawvideo --demuxer-rawvideo-w=W --demuxer-rawvideo-h=H --demuxer-rawvideo-mp-format=rgba -
build: see viewer.mojo
"""
import smooth as eng
import render as ren
from viewer import Viewer, write_all
from std.sys import argv
from max.gpu.host import DeviceContext


def main() raises:
    var a = argv()
    var frames = Int(String(a[2]))
    var cam = List[Float32]()
    for k in range(7):
        cam.append(Float32(atof(String(a[3 + k]))))
    var ctx = DeviceContext(0)
    var v = Viewer(ctx, String(a[1]), cam)
    var ke = ctx.compile_function[eng.smooth]()
    var kc = ctx.compile_function[ren.clear]()
    var kr = ctx.compile_function[ren.raster]()
    var ks = ctx.compile_function[ren.shade]()
    var ksc = ctx.compile_function[ren.suncam]()
    var kcs = ctx.compile_function[ren.clear_sm]()
    var fr = 0
    while frames == 0 or fr < frames:
        v.step(ctx, ke)
        comptime if eng.CPU:  # the renderer reads the device rows
            ctx.enqueue_copy(v.o_b, v.ob.unsafe_ptr().unsafe_bitcast[Float32]())
        v.frame(ctx, kc, kr, ks, ksc, kcs)
        ctx.enqueue_copy(v.out_h, v.img_b)
        ctx.synchronize()
        fr += 1
        if len(a) <= 10 and not write_all(1, v.out_h.unsafe_ptr().unsafe_bitcast[UInt8](), ren.W * ren.H * 4):
            return  # reader gone
    if len(a) > 10:
        var out = String(a[10])
        with open(out, "w") as f:
            f.write_bytes(Span(unsafe_ptr=v.out_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=ren.W * ren.H * 4))
        var qh = ctx.enqueue_create_host_buffer[DType.float32](eng.NQ * eng.F)
        comptime if eng.CPU:
            ctx.enqueue_copy(v.q_b, v.qp.unsafe_ptr().unsafe_bitcast[Float32]())
        ctx.enqueue_copy(qh, v.q_b)
        ctx.synchronize()
        with open(out + ".qpos", "w") as f:
            f.write_bytes(Span(unsafe_ptr=qh.unsafe_ptr().unsafe_bitcast[UInt8](), length=eng.NQ * eng.F * 4))
