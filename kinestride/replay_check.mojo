"""check.sh replay: run N steps, snapshot, run M, restore, run M, restore, run M: the three M-step states are byte-equal."""
import smooth as eng
import game_state as game
from std.sys import argv
from max.gpu.host import DeviceContext
from layout import TileTensor


def main() raises:
    var dir = String(argv()[1])
    var n = Int(String(argv()[2]))
    var m = Int(String(argv()[3]))
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
    var kernel = ctx.compile_function[eng.smooth]()
    var controls = uv.unsafe_ptr().unsafe_bitcast[Float32]()

    var nact = 2 * eng.NMUSCLE
    var tail = (eng.NQ + eng.NV + nact + eng.NV) * eng.F
    var states = List[Float32](length=3 * tail, fill=0)
    var snap = game.Snapshot(ctx, eng.NQ, eng.NV, nact, eng.NV, eng.F)
    for phase in range(4):  # 0: run N and snapshot; 1..3: [restore] run M and read the state back (the ramp depends on the absolute step, so replays see the same inputs)
        if phase == 2 or phase == 3:
            game.restore(ctx, snap, q_b, v_b, o_b, eng.F, eng.OMUS, nact, eng.OWARM, eng.NV)
        var first = 0 if phase == 0 else n
        for i in range(first, first + (n if phase == 0 else m)):
            controls[0] = Float32(i + 1) / 40.0 if i < 40 else 1.0
            ctx.enqueue_copy(u_b, uv.unsafe_ptr().unsafe_bitcast[Float32]())
            ctx.enqueue_function(kernel, TileTensor(mi_b, eng.LI), TileTensor(mf_b, eng.LF), TileTensor(q_b, eng.LQ), TileTensor(v_b, eng.LV), TileTensor(u_b, eng.LU), TileTensor(o_b, eng.LO), grid_dim=eng.GRID, block_dim=eng.BLOCK)
        if phase == 0:
            snap = game.snapshot(ctx, q_b, v_b, o_b, eng.F, eng.OMUS, nact, eng.OWARM, eng.NV, eng.NQ, eng.NV)
            continue
        var at = states.unsafe_ptr().unsafe_offset((phase - 1) * tail)
        ctx.enqueue_copy(at, q_b)
        ctx.enqueue_copy(at.unsafe_offset(eng.NQ * eng.F), v_b)
        ctx.enqueue_copy(at.unsafe_offset((eng.NQ + eng.NV) * eng.F), o_b.create_sub_buffer[DType.float32](eng.OMUS * eng.F, nact * eng.F))
        ctx.enqueue_copy(at.unsafe_offset((eng.NQ + eng.NV + nact) * eng.F), o_b.create_sub_buffer[DType.float32](eng.OWARM * eng.F, eng.NV * eng.F))
        ctx.synchronize()
    var bad = 0
    for i in range(tail):
        if states[i] != states[tail + i] or states[tail + i] != states[2 * tail + i]:
            bad += 1
    if bad > 0:
        raise Error("FAIL replay: " + String(bad) + " of " + String(tail) + " state words differ after restore")
    print("PASS replay:", n, "+", m, "steps, restore twice byte-equal,", tail * 4, "B state,", snap.bytes, "B snapshot")
