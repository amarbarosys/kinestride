"""Tracking loop on the engine (docs/MOTION.md): every animal follows its own reference clip with hinge servos and a
root-assist weld, may be pushed, and after a push falls with the assist off: K physics steps per reference frame.

Per animal a, frame f -> f + 1 (ref = its reference row for f + 1, w = its servo influence):
  hinge servo target = q + w (ref - q) (w = 1: the reference itself; w scales the servo gain, q read back each frame)
  assist mocap = ref's root; assist weld active (MuJoCo eq_active) until the push frame, then off for good
  push slot (xfrc_applied) = its wrench on its body for its push frames, from the push frame
  w = 1 until the push, then a linear ramp to its fall influence over its ramp frames
DIR: model.py export (pairs '^g', actuators) built with -DNXF=1 -DEQACT=1, plus qpos.bin / qvel.bin (start state of
every animal), pairbody.bin (int32 per pair: the clip body its geom2 sits on), feet.bin (int32 count, then the clip body
of each foot), ref.bin (int32 T; float32 hz; float32 [T][NQ][animals]) and plan.bin (float32 [animals][11]: push frame
(-1: none), body, frames, force 3, torque 3, fall influence, ramp frames).
Writes DIR/out.bin: int32 T-1, bodies, feet, animals; float32 hz; int32 feet; then per frame float32 [3 x bodies][animals]
(xpos), [4 x bodies][animals] (xquat) of the character's bodies and [feet][animals] (1: any of that foot's pairs holds an
active contact slot). NOWRITE=1 runs the loop and writes nothing (throughput); COMPILE_ONLY=1 compiles the kernel
(into the cache) and exits.

usage: track DIR
build: mojo build $(cat DIR/sizes.txt) -DFLIES=F -DSTAGE=7 -DK=SUB -DNXF=1 -DEQACT=1 -I kinestride kinestride/track.mojo
"""
from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns

from layout import TileTensor
from max.gpu.host import DeviceContext

from smooth import BLOCK, EQACT, F, GRID, HTAB, K, LF, LI, LO, LQ, LU, LV, NB, NCP, NEQ, NFLT, NI, NMOCAP, NQ, NROW, NSLOT, NU, NUC, NV, NXF, OVALID, load, smooth


def words(path: String) raises -> List[UInt8]:
    return open(path, "r").read_bytes()


def put(mut fh: FileHandle, p: Pointer[Float32, _], n: Int) raises:
    fh.write_bytes(Span(unsafe_ptr=p.unsafe_bitcast[UInt8](), length=4 * n))


def main() raises:
    comptime assert NMOCAP == 1 and NXF == 1 and EQACT == 1 and NEQ == 1, "one assist mocap body, one push slot, its weld switchable"
    comptime assert HTAB or NSLOT == 3 * NCP, "every pair is plane-mesh: 3 contact slots each (HTAB: pooled heightfield slots, foot flags not read)"
    comptime assert NU == NQ - 7, "a servo on every hinge"
    var dir = String(argv()[1])
    var rb = words(dir + "ref.bin")
    var t = Int(rb.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_offset(0)[])
    var hz = rb.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(1)[]
    var refs = rb.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(2)  # [T][NQ][F]
    if len(rb) != 8 + 4 * t * NQ * F:
        raise Error("ref.bin: " + String(len(rb)) + " bytes, expected " + String(8 + 4 * t * NQ * F))
    var pl = words(dir + "plan.bin")
    var plan = pl.unsafe_ptr().unsafe_bitcast[Float32]()  # [F][11]
    var fb = words(dir + "feet.bin")
    var fi = fb.unsafe_ptr().unsafe_bitcast[Int32]()
    var nf = Int(fi.unsafe_offset(0)[])
    var pb = words(dir + "pairbody.bin")
    var pairbody = pb.unsafe_ptr().unsafe_bitcast[Int32]()
    var nbc = NB - 2  # the character's bodies: not the world, not the assist mocap body (the model's last)
    var write = getenv("NOWRITE") == ""

    var ctx = DeviceContext(0)
    var kernel = ctx.compile_function[smooth]()  # first, so COMPILE_ONLY allocates nothing on the card
    if getenv("COMPILE_ONLY") != "":  # fill the kernel cache from the CPU room; the GPU-room run then starts at once
        print("compiled", dir)
        return
    var mi_b = ctx.enqueue_create_buffer[DType.int32](NI)
    var mf_b = ctx.enqueue_create_buffer[DType.float32](NFLT)
    var q_b = ctx.enqueue_create_buffer[DType.float32](NQ * F)
    var v_b = ctx.enqueue_create_buffer[DType.float32](NV * F)
    var u_b = ctx.enqueue_create_buffer[DType.float32](NUC * F)
    var o_b = ctx.enqueue_create_buffer[DType.float32](NROW * F)
    o_b.enqueue_fill(0)
    var mi = load(dir + "mi.bin", NI)
    var mf = load(dir + "mf.bin", NFLT)
    var qp = load(dir + "qpos.bin", NQ * F)
    var qv = load(dir + "qvel.bin", NV * F)
    ctx.enqueue_copy(mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(v_b, qv.unsafe_ptr().unsafe_bitcast[Float32]())

    comptime NPOSE = 7 * (NB - 1)  # xpos 3 per body 1.., then xquat 4 per body 1.., rows [row][animal]
    var u = List[Float32](length=NUC * F, fill=0)
    var q = List[Float32](length=NQ * F, fill=0)
    var pose = List[Float32](length=NPOSE * F, fill=0)
    var valid = List[Float32](length=NSLOT * F, fill=0)
    var con = List[Float32](length=nf * F, fill=0)
    var footpair = List[Int](length=NCP, fill=-1)  # the foot index of each pair's body, -1 for other bodies
    for pr in range(NCP):
        for i in range(nf):
            if Int(pairbody.unsafe_offset(pr)[]) == Int(fi.unsafe_offset(1 + i)[]):
                footpair[pr] = i
    var fh = open(dir + "out.bin", "w")
    if write:
        var head: List[Int32] = [Int32(t - 1), Int32(nbc), Int32(nf), Int32(F)]
        fh.write_bytes(Span(unsafe_ptr=head.unsafe_ptr().unsafe_bitcast[UInt8](), length=16))
        var hzl: List[Float32] = [hz]
        put(fh, hzl.unsafe_ptr(), 1)
        fh.write_bytes(Span(unsafe_ptr=fi.unsafe_offset(1).unsafe_bitcast[UInt8](), length=nf * 4))
    var t0 = perf_counter_ns()
    for f in range(t - 1):
        ctx.enqueue_copy(q.unsafe_ptr(), q_b)
        ctx.synchronize()
        var row = refs.unsafe_offset((f + 1) * NQ * F)  # the reference for frame f + 1, [NQ][F]
        for a in range(F):
            var p = plan.unsafe_offset(11 * a)
            var push = Int(p[])
            var since = f - push if push >= 0 else -1
            var w = Float32(1)
            if since >= 0:
                w = 1 - (1 - p.unsafe_offset(9)[]) * min(Float32(1), Float32(since + 1) / max(Float32(1), p.unsafe_offset(10)[]))
            for i in range(NU):
                var qi = q[(7 + i) * F + a]
                var target = row.unsafe_offset((7 + i) * F + a)[]
                u[i * F + a] = target if w == 1 else qi + w * (target - qi)  # w = 1: the reference exactly
            for k in range(7):
                u[(NU + k) * F + a] = row.unsafe_offset(k * F + a)[]
            var pushing = since >= 0 and since < Int(p.unsafe_offset(2)[])
            u[(NU + 7) * F + a] = p.unsafe_offset(1)[] if pushing else 0  # slot body (0: empty), then the wrench
            for k in range(6):
                u[(NU + 8 + k) * F + a] = p.unsafe_offset(3 + k)[] if pushing else 0
            u[(NU + 14) * F + a] = 0 if since >= 0 else 1  # the assist weld, off from the push on
        ctx.enqueue_copy(u_b, u.unsafe_ptr())
        ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV),
                             TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=GRID, block_dim=BLOCK)
        if not write:
            continue
        ctx.enqueue_copy(pose.unsafe_ptr(), o_b.create_sub_buffer[DType.float32](0, NPOSE * F))
        ctx.enqueue_copy(valid.unsafe_ptr(), o_b.create_sub_buffer[DType.float32](OVALID * F, NSLOT * F))
        ctx.synchronize()
        for i in range(nf * F):
            con[i] = 0
        for pr in range(0 if HTAB else NCP):
            if footpair[pr] >= 0:
                for sl in range(3):
                    for a in range(F):
                        if valid[(3 * pr + sl) * F + a] != 0:
                            con[footpair[pr] * F + a] = 1
        put(fh, pose.unsafe_ptr(), 3 * nbc * F)  # xpos of the character's bodies
        put(fh, pose.unsafe_ptr().unsafe_offset(3 * (NB - 1) * F), 4 * nbc * F)  # their xquat
        put(fh, con.unsafe_ptr(), nf * F)
    ctx.synchronize()
    var s = Float64(perf_counter_ns() - t0) / 1e9
    print("track", dir, "frames", t - 1, "animals", F, "K", K, ":", s / Float64(t - 1) * 1000, "ms per frame,",
          Float64((t - 1) * F * K) / s, "animal-steps/s,", "written" if write else "not written (NOWRITE)")
