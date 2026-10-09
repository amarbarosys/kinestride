"""Body engine S0 spike: how many fly-steps per second can a fused GPU kernel sustain?

Reduced fly, enough to carry the real per-step work: a rigid thorax (6 DOF, quaternion, diagonal
inertia) on 6 kinematic legs of 3 joints (coxa yaw, femur and tibia pitch) that follow a tripod gait
exactly. A stance foot within REACH of the floor is pulled down by ADH (the NeuroMechFly adhesion
actuator, which also raises its friction limit). Each foot touching the floor (z < 0) pushes back with a soft spring-damper normal force and
viscous friction capped by mu x normal. Foot velocity is v + w x r from the current body state plus
the leg's own swing (finite difference of the foot in body frame, rotated): differencing the world
position instead lags the body term by a step, which pumped the tilt modes until the fly blew up. Semi-implicit Euler at dt 0.1 ms, the NeuroMechFly step.

Layout for bytes: one thread per fly, state in registers for K steps per launch, SoA in global
memory (13 float32 per fly: pos 3, vel 3, quat 4, omega 3 = 52 B), loaded and stored once per launch;
leg geometry and gait are comptime constants, so a step touches no memory at all.

usage: mojo build -DFLIES=N s0.mojo; s0 LAUNCHES      prints fly-steps/s and the state check
"""
from std.math import cos, sin, sqrt
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime F = get_defined_int["FLIES", 4096]()
comptime K = get_defined_int["STEPS", 1000]()  # steps per launch
comptime LS = row_major[13 * F]()
comptime DT = Float32(1e-4)
comptime M = Float32(1e-6)                 # kg
comptime I = Float32(4e-13)                # kg m^2 per axis, thorax + legs (m L^2 / 6, L 1.5 mm)
comptime KN = Float32(2.0)                 # N/m per foot
comptime CN = Float32(5e-4)                # N s/m; explicit Euler needs sum(c r^2 / I) dt < 2
comptime MU = Float32(0.3)                 # no adhesion pads in S0; mu 1 tipped every fly (1 g of thrust at foot height)
comptime CF = Float32(5e-4)                # viscous friction, capped by MU x normal
comptime ADH = Float32(40e-6)              # N per stance foot, body-v2 adhesion gain 40 (g mm/s^2), ~4 body weights
comptime REACH = Float32(0.02e-3)          # adhesion acts within 20 um of the floor (MuJoCo: in contact)
comptime L1 = Float32(0.7e-3)              # femur, m
comptime L2 = Float32(0.7e-3)              # tibia, m
comptime HZ = Float32(10.0)                # stride frequency


@always_inline
def rotate(qw: Float32, qx: Float32, qy: Float32, qz: Float32, x: Float32, y: Float32, z: Float32) -> SIMD[DType.float32, 4]:
    """Rotate v by q: v + w t + q x t, t = 2 q x v."""
    var tx = 2 * (qy * z - qz * y)
    var ty = 2 * (qz * x - qx * z)
    var tz = 2 * (qx * y - qy * x)
    return SIMD[DType.float32, 4](x + qw * tx + qy * tz - qz * ty, y + qw * ty + qz * tx - qx * tz, z + qw * tz + qx * ty - qy * tx, 0)


def step(s: TileTensor[DType.float32, type_of(LS), MutAnyOrigin], t0: Int32):
    var f = block_idx.x * block_dim.x + thread_idx.x
    if f >= F:
        return
    var px = s[f]; var py = s[F + f]; var pz = s[2 * F + f]
    var vx = s[3 * F + f]; var vy = s[4 * F + f]; var vz = s[5 * F + f]
    var qw = s[6 * F + f]; var qx = s[7 * F + f]; var qy = s[8 * F + f]; var qz = s[9 * F + f]
    var wx = s[10 * F + f]; var wy = s[11 * F + f]; var wz = s[12 * F + f]
    var prev = SIMD[DType.float32, 32](0)   # last body-frame foot positions, 6 legs x (x, y, z), slots 18..31 unused
    for k in range(K):
        var phase = Float32(2 * 3.14159265) * HZ * DT * Float32(Int(t0) + k)
        var fx: Float32 = 0; var fy: Float32 = 0; var fz: Float32 = -M * 9.81
        var tx: Float32 = 0; var ty: Float32 = 0; var tz: Float32 = 0
        comptime for leg in range(6):
            comptime side = Float32(1 if leg < 3 else -1)
            comptime hx = Float32(0.4e-3) * Float32(1 - leg % 3)   # front, middle, hind hips
            var ph = phase + Float32(3.14159265 if (leg % 2 == 1) != (leg >= 3) else 0.0)  # tripods
            var yaw = 0.35 * sin(ph)
            var lift = 0.25 * max(cos(ph), 0)                       # swing lifts, stance presses
            var femur = 0.6 + lift                                   # swing raises the femur: foot 0.08 mm
            var tibia = Float32(-1.9)                                # below the hip vs 0.28 in stance
            var reach = L1 * cos(femur) + L2 * cos(femur + tibia)
            var bx = hx + reach * sin(yaw)
            var by = side * (Float32(0.25e-3) + reach * cos(yaw))
            var bz = L1 * sin(femur) + L2 * sin(femur + tibia)
            var r = rotate(qw, qx, qy, qz, bx, by, bz)
            var x = px + r[0]; var y = py + r[1]; var z = pz + r[2]
            var stick = lift == 0 and z < REACH   # stance foot at the floor: adhesion pulls it down
            if stick:
                fz -= ADH
                tx -= r[1] * ADH; ty += r[0] * ADH
            if z < 0 and k > 0:
                var sw = rotate(qw, qx, qy, qz, (bx - prev[3 * leg]) / DT, (by - prev[3 * leg + 1]) / DT, (bz - prev[3 * leg + 2]) / DT)
                var ux = vx + wy * r[2] - wz * r[1] + sw[0]
                var uy = vy + wz * r[0] - wx * r[2] + sw[1]
                var uz = vz + wx * r[1] - wy * r[0] + sw[2]
                var n = max(-KN * z - CN * uz, 0)
                var cap = MU * (n + (ADH if stick else 0)) / max(CF * sqrt(ux * ux + uy * uy), Float32(1e-12))
                var g = CF * min(cap, 1)
                var ex = -g * ux; var ey = -g * uy
                fx += ex; fy += ey; fz += n
                tx += r[1] * n - r[2] * ey; ty += r[2] * ex - r[0] * n; tz += r[0] * ey - r[1] * ex
            prev[3 * leg] = bx; prev[3 * leg + 1] = by; prev[3 * leg + 2] = bz
        vx += DT * fx / M; vy += DT * fy / M; vz += DT * fz / M
        px += DT * vx; py += DT * vy; pz += DT * vz
        wx += DT * tx / I; wy += DT * ty / I; wz += DT * tz / I
        var dw = -0.5 * DT * (qx * wx + qy * wy + qz * wz)
        var dx = 0.5 * DT * (qw * wx - qy * wz + qz * wy)   # q' = q + dt/2 w q, w in world frame
        var dy = 0.5 * DT * (qw * wy - qz * wx + qx * wz)
        var dz = 0.5 * DT * (qw * wz - qx * wy + qy * wx)
        qw += dw; qx += dx; qy += dy; qz += dz
        var norm = 1 / sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw *= norm; qx *= norm; qy *= norm; qz *= norm
    s[f] = px; s[F + f] = py; s[2 * F + f] = pz
    s[3 * F + f] = vx; s[4 * F + f] = vy; s[5 * F + f] = vz
    s[6 * F + f] = qw; s[7 * F + f] = qx; s[8 * F + f] = qy; s[9 * F + f] = qz
    s[10 * F + f] = wx; s[11 * F + f] = wy; s[12 * F + f] = wz


def main() raises:
    var launches = Int(String(argv()[1]))
    var host = List[Float32](length=13 * F, fill=0)
    for f in range(F):
        host[2 * F + f] = 0.5e-3   # thorax 0.5 mm up, upright, at rest
        host[6 * F + f] = 1
    var ctx = DeviceContext(0)
    var buf = ctx.enqueue_create_buffer[DType.float32](13 * F)
    ctx.enqueue_copy(buf, host.unsafe_ptr())
    var s = TileTensor(buf, LS)
    var kernel = ctx.compile_function[step]()
    ctx.enqueue_function(kernel, s, Int32(0), grid_dim=(F + 255) // 256, block_dim=256)  # warm-up
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for l in range(launches):
        ctx.enqueue_function(kernel, s, Int32((l + 1) * K), grid_dim=(F + 255) // 256, block_dim=256)
    ctx.synchronize()
    var ns = perf_counter_ns() - t0
    ctx.enqueue_copy(host.unsafe_ptr(), buf)
    ctx.synchronize()
    var bad = 0  # a fly fails if it fell, flew off, blew up or did not walk
    var zlo: Float32 = 1; var zhi: Float32 = -1; var xs: Float32 = 0; var up: Float32 = 1
    for f in range(F):
        var z = host[2 * F + f]
        var qx = host[7 * F + f]; var qy = host[8 * F + f]
        var ok = z == z and z > 0.05e-3 and z < 2e-3 and abs(host[f]) > 1e-3 and 1 - 2 * (qx * qx + qy * qy) > 0.9   # finite, upright (< 26 deg tilt), walked > 1 mm
        bad += 0 if ok else 1
        zlo = min(zlo, z); zhi = max(zhi, z); xs += host[f]
        up = min(up, 1 - 2 * (qx * qx + qy * qy))   # cos of tilt
    var steps = Float64(F) * Float64(launches * K)
    print(F, "flies,", launches * K, "steps:", Int(steps / (Float64(ns) / 1e9)), "fly-steps/s;",
          "z", zlo * 1e3, "..", zhi * 1e3, "mm; mean x", xs / Float32(F) * 1e3, "mm; min cos tilt", up, ";",
          "PASS" if bad == 0 else "FAIL " + String(bad) + "/" + String(F) + " flies fell or blew up")
    if bad:
        raise Error("state check failed")
