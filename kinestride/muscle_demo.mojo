"""Short sine driven whole spider run; engine state stays in the persistent result buffer."""
import smooth as eng
from std.math import abs, sin
from std.sys import argv
from std.collections import Span
from max.gpu.host import DeviceContext
from layout import TileTensor


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var steps = Int(String(args[2]))
    var mode = String(args[3])
    var trace_mode = mode == "trace"
    var game_mode = mode == "game"
    var recovery_mode = mode == "recovery0" or mode == "recovery001"
    var ctx = DeviceContext(0)
    var mi_b = ctx.enqueue_create_buffer[DType.int32](eng.NI)
    var mf_b = ctx.enqueue_create_buffer[DType.float32](eng.NFL)
    var q_b = ctx.enqueue_create_buffer[DType.float32](eng.NQ * eng.F)
    var v_b = ctx.enqueue_create_buffer[DType.float32](eng.NV * eng.F)
    var u_b = ctx.enqueue_create_buffer[DType.float32](max(eng.NUC, 1) * eng.F)
    var o_b = ctx.enqueue_create_buffer[DType.float32](eng.NROW * eng.F)
    o_b.enqueue_fill(0)
    var mi = eng.load(dir + "mi.bin", eng.NI)
    var mf = eng.load(dir + "mf.bin", eng.NFL)
    var qp = eng.load(dir + "qpos.bin", eng.NQ * eng.F)
    var qv = eng.load(dir + "qvel.bin", eng.NV * eng.F)
    var uv = eng.load(dir + "ctrl.bin", eng.NUC * eng.F)
    ctx.enqueue_copy(mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(v_b, qv.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(u_b, uv.unsafe_ptr().unsafe_bitcast[Float32]())
    var kernel = ctx.compile_function[eng.smooth]()
    var controls = uv.unsafe_ptr().unsafe_bitcast[Float32]()
    var output = List[Float32](length=eng.NROW * eng.F, fill=0)
    var trace = List[Float32](length=60, fill=0)
    var samples = 0
    var peak_fatigue: Float32 = 0
    var dt = mf.unsafe_ptr().unsafe_bitcast[Float32]()[eng.TSTEP]
    if recovery_mode:
        output[eng.OMUS * eng.F] = 1.0 / 7.5
        output[(eng.OMUS + 1) * eng.F] = 0.8
        ctx.enqueue_copy(o_b, output.unsafe_ptr())
    for i in range(steps):
        if recovery_mode:
            controls[0] = 0.0 if mode == "recovery0" else 0.01
        elif trace_mode or game_mode:
            if i < 40:
                controls[0] = Float32(i + 1) / 40.0
            elif i < 120:
                controls[0] = 1.0
            else:
                controls[0] = 0.0
        else:
            controls[0] = 0.5 + 0.5 * sin(6.2831853 * Float32(i) * dt)
        ctx.enqueue_copy(u_b, uv.unsafe_ptr().unsafe_bitcast[Float32]())
        ctx.enqueue_function(kernel, TileTensor(mi_b, eng.LI), TileTensor(mf_b, eng.LF), TileTensor(q_b, eng.LQ), TileTensor(v_b, eng.LV), TileTensor(u_b, eng.LU), TileTensor(o_b, eng.LO), grid_dim=eng.GRID, block_dim=eng.BLOCK)
        var step = i + 1
        var checkpoint = step == 1 or step == 40 or step == 120 or step == 160 or step == 200 or step == 240
        if ((trace_mode or game_mode) and checkpoint) or (recovery_mode and i == 0) or (not trace_mode and not game_mode and not recovery_mode and (i % 100 == 99 or i == steps - 1)):
            ctx.enqueue_copy(output.unsafe_ptr(), o_b)
            ctx.synchronize()
            if trace_mode or game_mode or recovery_mode:
                var stride = 10 if game_mode else 6
                var row = samples * stride
                trace[row] = Float32(step)
                trace[row + 1] = controls[0]
                trace[row + 2] = output[eng.OAF * eng.F]
                trace[row + 3] = output[eng.OQACT * eng.F]
                trace[row + 4] = output[eng.OMUS * eng.F]
                trace[row + 5] = output[(eng.OMUS + 1) * eng.F]
                if game_mode:
                    trace[row + 6] = output[eng.OGAME * eng.F]
                    trace[row + 7] = output[(eng.OGAME + 1) * eng.F]
                    trace[row + 8] = output[(eng.OGAME + 2) * eng.F]
                    trace[row + 9] = output[(eng.OGAME + 3) * eng.F]
                samples += 1
            else:
                peak_fatigue = max(peak_fatigue, output[(eng.OMUS + 1) * eng.F])
    if trace_mode or game_mode or recovery_mode:
        var expected_samples = 1 if recovery_mode else 6
        if samples != expected_samples:
            raise Error("FAIL muscle trace: unexpected oracle checkpoint count")
        var count = 6 if recovery_mode else (36 if trace_mode else 60)
        with open(dir + "trace.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=trace.unsafe_ptr().unsafe_bitcast[UInt8](), length=count * 4))
        if recovery_mode:
            print("PASS muscle recovery runner:", mode)
        elif game_mode:
            print("PASS muscle game runner:", steps, "steps,", samples, "checkpoints")
        else:
            print("PASS muscle trace runner:", steps, "steps,", samples, "checkpoints")
        return
    var qfinal = List[Float32](length=eng.NQ * eng.F, fill=0)
    var vfinal = List[Float32](length=eng.NV * eng.F, fill=0)
    ctx.enqueue_copy(qfinal.unsafe_ptr(), q_b)
    ctx.enqueue_copy(vfinal.unsafe_ptr(), v_b)
    ctx.enqueue_copy(output.unsafe_ptr(), o_b)
    ctx.synchronize()
    var final_fatigue = output[(eng.OMUS + 1) * eng.F]
    var final_activation = output[eng.OMUS * eng.F]
    var stable = final_fatigue >= 0 and final_fatigue < 1
    for i in range(eng.NQ):
        stable = stable and abs(qfinal[i * eng.F]) < 100
    for i in range(eng.NV):
        stable = stable and abs(vfinal[i * eng.F]) < 100
    if not stable or peak_fatigue <= 0:
        raise Error("FAIL muscle demo: unstable state or fatigue did not rise")
    print("PASS muscle sine demo:", steps, "steps, peak fatigue", peak_fatigue, "final fatigue", final_fatigue, "activation", final_activation)
