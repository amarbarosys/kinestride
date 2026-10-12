"""Tracking loop on the engine (docs/plans/MOTION_DATASET.md, unit B): a reference clip drives every animal's hinge
servos and its root-assist mocap body; K physics steps per reference frame; poses and solver contacts recorded.

Per frame f -> f + 1: ctrl = the hinges of qref[f + 1], mocap = its root (pos, quat), one launch of K steps.
DIR: model.py export (pairs '^g', actuators) plus qpos.bin / qvel.bin (start state of every animal) and pairbody.bin
(int32 per pair: the clip body its geom2 sits on). STEM: bvh clip's .qref and .mclip (feet).
Writes STEM.ks.mclip: frames 1..T-1 of animal 0's character bodies (the assist mocap body is the model's last) and per
foot whether any of its pairs holds an active contact slot. Every animal gets the same input, so every animal's poses
must equal animal 0's byte for byte: a mismatch exits non-zero.

usage: track DIR STEM
build: mojo build $(cat DIR/sizes.txt) -DFLIES=F -DSTAGE=7 -DK=SUB -I kinestride kinestride/track.mojo
"""
from std.sys import argv
from std.time import perf_counter_ns

from layout import TileTensor
from max.gpu.host import DeviceContext

from smooth import BLOCK, F, GRID, LF, LI, LO, LQ, LU, LV, NB, NCP, NFL, NI, NMOCAP, NQ, NROW, NSLOT, NU, NUC, NV, NXF, OVALID, load, smooth


def words(path: String) raises -> List[UInt8]:
    return open(path, "r").read_bytes()


def main() raises:
    comptime assert NMOCAP == 1 and NXF == 0, "one assist mocap body, no push slots"
    comptime assert NSLOT == 3 * NCP, "every pair is plane-mesh: 3 contact slots each"
    var dir = String(argv()[1])
    var stem = String(argv()[2])
    var qr = words(stem + ".qref")
    var qi = qr.unsafe_ptr().unsafe_bitcast[Int32]()
    var qf = qr.unsafe_ptr().unsafe_bitcast[Float32]()
    var t = Int(qi.unsafe_offset(0)[])
    if Int(qi.unsafe_offset(1)[]) != NQ:
        raise Error(stem + ".qref: nq " + String(qi.unsafe_offset(1)[]) + ", model NQ " + String(NQ))
    var hz = qf.unsafe_offset(2)[]
    var mc = words(stem + ".mclip")
    var mi32 = mc.unsafe_ptr().unsafe_bitcast[Int32]()
    var nbc = Int(mi32.unsafe_offset(1)[])
    var nf = Int(mi32.unsafe_offset(2)[])
    if nbc != NB - 2:
        raise Error("clip has " + String(nbc) + " bodies, the model " + String(NB - 2) + " besides world and assist")
    var feet = List[Int]()
    for i in range(nf):
        feet.append(Int(mi32.unsafe_offset(4 + i)[]))
    var pb = words(dir + "pairbody.bin")
    var pairbody = pb.unsafe_ptr().unsafe_bitcast[Int32]()

    var ctx = DeviceContext(0)
    var mi_b = ctx.enqueue_create_buffer[DType.int32](NI)
    var mf_b = ctx.enqueue_create_buffer[DType.float32](NFL)
    var q_b = ctx.enqueue_create_buffer[DType.float32](NQ * F)
    var v_b = ctx.enqueue_create_buffer[DType.float32](NV * F)
    var u_b = ctx.enqueue_create_buffer[DType.float32](NUC * F)
    var o_b = ctx.enqueue_create_buffer[DType.float32](NROW * F)
    o_b.enqueue_fill(0)
    var mi = load(dir + "mi.bin", NI)
    var mf = load(dir + "mf.bin", NFL)
    var qp = load(dir + "qpos.bin", NQ * F)
    var qv = load(dir + "qvel.bin", NV * F)
    ctx.enqueue_copy(mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(v_b, qv.unsafe_ptr().unsafe_bitcast[Float32]())
    var kernel = ctx.compile_function[smooth]()

    comptime NPOSE = 7 * (NB - 1)  # xpos 3 then xquat 4 per body 1.., rows [row][animal]
    var u = List[Float32](length=NUC * F, fill=0)
    var pose = List[Float32](length=NPOSE * F, fill=0)
    var valid = List[Float32](length=NSLOT * F, fill=0)
    var pos = List[Float32](capacity=(t - 1) * nbc * 3)
    var quat = List[Float32](capacity=(t - 1) * nbc * 4)
    var contact = List[Float32](capacity=(t - 1) * nf)
    var differ = 0
    var t0 = perf_counter_ns()
    for f in range(t - 1):
        var row = qf.unsafe_offset(3 + (f + 1) * NQ)  # qref[f + 1]
        for i in range(NU):
            for a in range(F):
                u[i * F + a] = row.unsafe_offset(7 + i)[]
        for k in range(7):
            for a in range(F):
                u[(NU + k) * F + a] = row.unsafe_offset(k)[]
        ctx.enqueue_copy(u_b, u.unsafe_ptr())
        ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV),
                             TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=GRID, block_dim=BLOCK)
        ctx.enqueue_copy(pose.unsafe_ptr(), o_b.create_sub_buffer[DType.float32](0, NPOSE * F))
        ctx.enqueue_copy(valid.unsafe_ptr(), o_b.create_sub_buffer[DType.float32](OVALID * F, NSLOT * F))
        ctx.synchronize()
        for r in range(NPOSE):
            for a in range(1, F):
                if pose[r * F + a] != pose[r * F]:
                    differ += 1
        for b in range(nbc):
            for k in range(3):
                pos.append(pose[(3 * b + k) * F])
        for b in range(nbc):
            for k in range(4):
                quat.append(pose[(3 * (NB - 1) + 4 * b + k) * F])
        for i in range(nf):
            var on = Float32(0)
            for p in range(NCP):
                if Int(pairbody.unsafe_offset(p)[]) == feet[i]:
                    for sl in range(3):
                        if valid[(3 * p + sl) * F] != 0:
                            on = 1
            contact.append(on)
    var fh = open(stem + ".ks.mclip", "w")
    var head: List[Int32] = [Int32(t - 1), Int32(nbc), Int32(nf)]
    fh.write_bytes(Span(unsafe_ptr=head.unsafe_ptr().unsafe_bitcast[UInt8](), length=12))
    var hzl: List[Float32] = [hz]
    fh.write_bytes(Span(unsafe_ptr=hzl.unsafe_ptr().unsafe_bitcast[UInt8](), length=4))
    var fl = List[Int32]()
    for b in feet:
        fl.append(Int32(b))
    fh.write_bytes(Span(unsafe_ptr=fl.unsafe_ptr().unsafe_bitcast[UInt8](), length=nf * 4))
    fh.write_bytes(Span(unsafe_ptr=pos.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(pos) * 4))
    fh.write_bytes(Span(unsafe_ptr=quat.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(quat) * 4))
    fh.write_bytes(Span(unsafe_ptr=contact.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(contact) * 4))
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    print("track", stem, "frames", t - 1, "animals", F, ":", ms / Float64(t - 1), "ms per frame (upload, launch, two row copies),",
          "pose values differing from animal 0:", differ)
    if differ > 0:
        raise Error("animals with the same input diverged: " + String(differ) + " pose values differ from animal 0")
