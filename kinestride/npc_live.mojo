"""npc-mind NPCs with kinestride bodies (docs/plans/BRAIN.md B4): the first F of the world's 100 NPCs walk as spiders.

Per tick (TICK_S of body time):
  acts: the scripted schedule or the npc-mind base brain (Brains; the gait shares its DeviceContext; obs from the world)
  world.step(acts): needs, food, wolves, and the tile and facing each NPC should reach (the grid move)
  bodies: every control step (K physics steps) intent.drive steers each spider toward its NPC's tile, then facing;
          fly-connectome's spider gait kernel turns the drives into servo targets on the card, the host engine steps
  the world then takes each embodied NPC's tile and facing from its body (a blocked tile keeps the grid move)
The other NPCs move on the grid as in npc-mind. Modes: body (as above), fake (the body lands exactly on the grid move:
the run must equal grid, byte for byte), grid (no bodies), calib (fixed drives by body index mod 4: walk (1, 0),
spin left, spin right (intent.SPIN, TURN), stand (0, 0); the world keeps its grid moves; prints each body's travel along and
across its start heading and its yaw change). -DMUT=1 zeroes the drives, -DMUT=2 flips the turn sign.

Prints: npclive ARM MODE SEED ticks T sat_body S sat_grid S track ACTION hit/n ... fallen n eats_body e eats_grid e
(eats per NPC). Writes OUT: per tick int32 [N][3] (x, y, face), float64 [N][4] needs, float32 [F][3] (body X, Y, yaw),
int32 [F][3] (the grid move: target x, y, face).

usage: npc_live DIR ARM MODE SEED TICKS OUT   ARM scripted | base (reads DIR/wout_SEED.bin, tools/npc_imitate.mojo)
build: mojo build $(cat DIR/sizes.txt) -DFLIES=F -DSTAGE=7 -DK=10 -DCPU=1 -I kinestride -I <npc-mind> -I .work/brain/inc
       -I <fly-connectome>/agent/body_engine kinestride/npc_live.mojo   (inc: npc_world.mojo -> npc-mind's world.mojo, whose
       module name kinestride's own world.mojo takes)
"""
from max.algorithm import parallelize
from std.math import cos, sin
from std.os import setenv
from std.random import random_float64, seed
from std.sys import argv, get_defined_int
from max.gpu.host import DeviceContext  # enqueue_function resolves the kernel handle against it
from layout import TileTensor

import smooth as eng
import intent as it
import spider_brain as sb
from mind import NH, O
from mind_gpu import Brains
from npc_world import EAT, FLEE, FWD, LEFT, N, RIGHT, World, make_base, make_minds

comptime MUT = get_defined_int["MUT", 0]()
comptime DT = Float32(1e-4)  # the spider's timestep (models/spider.py)
comptime TICK_S = Float32(0.8)
comptime NCS = Int(TICK_S / (Float32(eng.K) * DT) + Float32(0.5))  # control steps per tick
comptime NKIND = 5  # tracked per action kind: fwd, left, right, flee, stay (every other action)


def cal(k: Int) -> Tuple[Float32, Float32]:
    """calib drives: walk, spin left, spin right, stand."""
    if k == 0:
        return (Float32(1), Float32(0))
    if k == 3:
        return (Float32(0), Float32(0))
    return (it.SPIN, it.TURN * (it.CCW if k == 1 else -it.CCW))


def kind_of(a: Int) -> Int:
    return 0 if a == FWD else (1 if a == LEFT else (2 if a == RIGHT else (3 if a == FLEE else 4)))


def put[T: DType](mut fh: FileHandle, p: Pointer[Scalar[T], _], n: Int) raises:
    comptime bytes = 8 if T == DType.float64 else 4
    fh.write_bytes(Span(unsafe_ptr=p.unsafe_bitcast[UInt8](), length=n * bytes))


def main() raises:
    comptime assert eng.CPU, "host engine: 8 spiders step 9x faster on the CPU than on the card (fly-connectome spider_link.backend)"
    comptime assert eng.NUC == eng.NU, "the gait writes every ctrl row; the copy back takes the whole buffer"
    comptime assert eng.F <= N
    # Mojo's memory pools reserve 1.28 GiB pinned host + 417 MiB device per process up front (heaptide; Brain
    # mojo-baro/2026-10-08-leakd-engine-pinned-host.md); every buffer here is made once, so they buy nothing:
    # peak RSS 1,580 -> 238 MB (184 MB with one shared context), output byte-equal. Set before the first DeviceContext; an explicit env wins.
    _ = setenv("MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_SIZE", "0", False)
    _ = setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "0", False)
    var a = argv()
    var dir = String(a[1])
    var arm = String(a[2])
    var mode = String(a[3])
    var sd = Int(String(a[4]))
    var ticks = Int(String(a[5]))
    var fh = open(String(a[6]), "w")
    if arm != "scripted" and arm != "base":
        raise Error("FAIL arm: " + arm)
    if mode != "body" and mode != "fake" and mode != "grid" and mode != "calib":
        raise Error("FAIL mode: " + mode)

    # brains: the sweep's base arm (sweep_gpu.main seeds), one slot per NPC
    var g = Brains(N, 1)
    if arm == "base":
        seed(sd * 7919 + 1)
        var base = make_base()
        var wb = eng.load(dir + "wout_" + String(sd) + ".bin", 2 * len(base.wout))
        for k in range(len(base.wout)):
            base.wout[k] = wb.unsafe_ptr().unsafe_bitcast[Float64]().unsafe_offset(k)[]
        g.set_base(0, base)
        seed(sd * 7919 + 3)
        g.set_minds(0, make_minds(True, False), 0, 0)
    seed(sd * 104729 + 2)  # the sweep's test world for this seed
    var w = World()

    # bodies: host engine lists, each spider moved onto its NPC's tile and turned to its facing
    var mi = eng.load(dir + "mi.bin", eng.NI)
    var mf = eng.load(dir + "mf.bin", eng.NFL)
    var qp = eng.load(dir + "qpos.bin", eng.NQ * eng.F)
    var qv = eng.load(dir + "qvel.bin", eng.NV * eng.F)
    var uv = eng.load(dir + "ctrl.bin", eng.NU * eng.F)
    var ob = List[UInt8](length=4 * eng.NROW * eng.F, fill=0)
    var q = qp.unsafe_ptr().unsafe_bitcast[Float32]()
    var v = qv.unsafe_ptr().unsafe_bitcast[Float32]()
    comptime F = eng.F
    for b in range(F):
        var c = it.centre(w.x[b], w.y[b])
        q[b] = c[0]
        q[F + b] = c[1]
        var h = it.face_yaw(w.face[b]) / 2
        var cw = cos(h)
        var sz = sin(h)
        var w2 = q[3 * F + b]
        var x2 = q[4 * F + b]
        var y2 = q[5 * F + b]
        var z2 = q[6 * F + b]
        q[3 * F + b] = cw * w2 - sz * z2  # yaw (cw, 0, 0, sz) times the standing orientation
        q[4 * F + b] = cw * x2 - sz * y2
        q[5 * F + b] = cw * y2 + sz * x2
        q[6 * F + b] = cw * z2 + sz * w2
        var vx = v[b]
        var vy = v[F + b]
        v[b] = cos(2 * h) * vx - sin(2 * h) * vy  # free joint: linear velocity in the world frame
        v[F + b] = sin(2 * h) * vx + cos(2 * h) * vy

    var ctx = g.ctx  # one context for brains and gait: each DeviceContext maps its own 44 MB HSA queue area
    var kb = ctx.compile_function[sb.gait]()
    var u_b = ctx.enqueue_create_buffer[DType.float32](eng.NU * F)
    var drive_b = ctx.enqueue_create_buffer[DType.float32](2 * F)
    var drive_h = ctx.enqueue_create_host_buffer[DType.float32](2 * F)
    var hi = eng.host[DType.int32](mi, eng.LI)
    var hf = eng.host[DType.float32](mf, eng.LF)
    var hq = eng.host[DType.float32](qp, eng.LQ)
    var hv = eng.host[DType.float32](qv, eng.LV)
    var hu = eng.host[DType.float32](uv, eng.LU)
    var ho = eng.host[DType.float32](ob, eng.LO)

    var x0 = List[Float32](capacity=F)
    var y0 = List[Float32](capacity=F)
    var yaw0 = List[Float32](capacity=F)
    var turned = List[Float32](length=F, fill=0)
    for b in range(F):
        x0.append(q[b])
        y0.append(q[F + b])
        yaw0.append(it.yaw_of(q[3 * F + b], q[4 * F + b], q[5 * F + b], q[6 * F + b]))
    var hit = List[Int](length=NKIND, fill=0)
    var cnt = List[Int](length=NKIND, fill=0)
    var eats = List[Int](length=2, fill=0)
    var sat = List[Float64](length=2, fill=0)
    var fallen = 0
    var cs = 0
    var tx = List[Int](length=F, fill=0)
    var ty = List[Int](length=F, fill=0)
    var tf = List[Int](length=F, fill=0)
    var arrived = List[Bool](length=F, fill=False)
    var trace = List[Int32](length=3 * N, fill=0)
    var bpose = List[Float32](length=3 * F, fill=0)
    var tgt = List[Int32](length=3 * F, fill=0)
    for t in range(ticks):
        var acts = List[Int](capacity=N * NH)
        if arm == "base":
            var o = w.all_obs()
            for i in range(N * O):
                g.obs_h[i] = Float32(o[i])
            for i in range(N * NH):
                g.u_h[i] = Float32(random_float64())
            g.step()
            for i in range(N * NH):
                acts.append(Int(g.acts_h[i]))
        else:
            for i in range(N):
                var sc = w.scripted(i)
                acts.append(sc[0])
                acts.append(sc[1])
                acts.append(12)  # npc-mind world.SILENT (V = 12 words)
        for i in range(N):
            eats[0 if i < F else 1] += 1 if acts[i * NH] == EAT and w.food_here(i) >= 0 else 0
        _ = w.step(acts)
        for b in range(F):
            tx[b] = w.x[b]
            ty[b] = w.y[b]
            tf[b] = w.face[b]
            var tl = it.tile_of(q[b], q[F + b])
            arrived[b] = tl[0] == tx[b] and tl[1] == ty[b]  # a turn in place: no walk back to the centre first
            tgt[3 * b] = Int32(tx[b])
            tgt[3 * b + 1] = Int32(ty[b])
            tgt[3 * b + 2] = Int32(tf[b])
        if mode == "body" or mode == "calib":
            for _ in range(NCS):
                for b in range(F):
                    var d = it.drive(q[b], q[F + b], it.yaw_of(q[3 * F + b], q[4 * F + b], q[5 * F + b], q[6 * F + b]), tx[b], ty[b], tf[b], arrived[b])
                    if mode == "calib":
                        d = cal(b % 4)
                    drive_h[b] = 0 if MUT == 1 else d[0]
                    drive_h[F + b] = 0 if MUT == 1 else (-d[1] if MUT == 2 else d[1])
                ctx.enqueue_copy(drive_b, drive_h)
                ctx.enqueue_function(kb, TileTensor(u_b, eng.LU), TileTensor(drive_b, sb.LD), Float32(cs * eng.K) * DT,
                    grid_dim=(eng.NU * F + 63) // 64, block_dim=64)
                ctx.enqueue_copy(uv.unsafe_ptr().unsafe_bitcast[Float32](), u_b)
                ctx.synchronize()

                def step_animal(f: Int) {imm hi, imm hf, imm hq, imm hv, imm hu, imm ho}:
                    eng.smooth_at(hi, hf, hq, hv, hu, ho, f)

                parallelize(step_animal, F)
                cs += 1
                for b in range(F):  # yaw change summed per control step: a full turn is not lost to the wrap
                    var y1 = it.yaw_of(q[3 * F + b], q[4 * F + b], q[5 * F + b], q[6 * F + b])
                    turned[b] += it.wrap(y1 - yaw0[b] - turned[b])
        for b in range(F):
            var bx = tx[b]
            var by = ty[b]
            var bf = tf[b]
            var yaw = it.face_yaw(bf)
            if mode == "body":
                yaw = it.yaw_of(q[3 * F + b], q[4 * F + b], q[5 * F + b], q[6 * F + b])
                var tl = it.tile_of(q[b], q[F + b])
                bx = tl[0]
                by = tl[1]
                bf = it.face_of(yaw)
                if 1 - 2 * (q[4 * F + b] * q[4 * F + b] + q[5 * F + b] * q[5 * F + b]) < Float32(0.5):
                    fallen += 1
            var k = kind_of(acts[b * NH])
            cnt[k] += 1
            hit[k] += 1 if bx == tx[b] and by == ty[b] and bf == tf[b] else 0
            if mode == "body" or mode == "fake":
                if w.open(bx, by):
                    w.x[b] = bx
                    w.y[b] = by
                w.face[b] = bf
            var c = it.centre(bx, by)
            bpose[3 * b] = q[b] if mode == "body" else c[0]
            bpose[3 * b + 1] = q[F + b] if mode == "body" else c[1]
            bpose[3 * b + 2] = yaw
        if mode == "body" or mode == "fake":
            w.rebuild()
        for i in range(N):
            trace[3 * i] = Int32(w.x[i])
            trace[3 * i + 1] = Int32(w.y[i])
            trace[3 * i + 2] = Int32(w.face[i])
            sat[0 if i < F else 1] += 1 - (w.need[4 * i] + w.need[4 * i + 1] + w.need[4 * i + 2] + w.need[4 * i + 3]) / 4
        put(fh, trace.unsafe_ptr(), 3 * N)
        put(fh, w.need.unsafe_ptr(), 4 * N)
        put(fh, bpose.unsafe_ptr(), 3 * F)
        put(fh, tgt.unsafe_ptr(), 3 * F)
    fh.close()
    if mode == "calib":
        var cn: List[String] = ["walk", "left", "right", "stand"]
        for b in range(F):
            var ex = q[b] - x0[b]
            var ey = q[F + b] - y0[b]
            var along = ex * cos(yaw0[b]) + ey * sin(yaw0[b])
            var across = -ex * sin(yaw0[b]) + ey * cos(yaw0[b])  # + = to the left of the start heading
            print("calib", b, cn[b % 4], "s", Float32(ticks) * TICK_S, "along_mm", along, "left_mm", across, "yaw_deg", turned[b] * 180 / it.PI)
    var names: List[String] = ["fwd", "left", "right", "flee", "stay"]
    var line = String("npclive ") + arm + " " + mode + " " + String(sd) + " ticks " + String(ticks)
    line += " sat_body " + String(sat[0] / Float64(F * ticks)) + " sat_grid " + String(sat[1] / Float64((N - F) * ticks)) + " track"
    for k in range(NKIND):
        line += " " + names[k] + " " + String(hit[k]) + "/" + String(cnt[k])
    line += " fallen " + String(fallen) + " eats_body " + String(Float64(eats[0]) / Float64(F)) + " eats_grid " + String(Float64(eats[1]) / Float64(N - F))
    print(line)
