"""BVH mocap -> an MJCF character and its reference clip (docs/plans/MOTION_DATASET.md, unit 2).

Every BVH joint becomes a body with three hinges; MuJoCo applies a body's joints in definition order, so hinge angles are
intrinsic Tait-Bryan angles. Their axis order is chosen per joint away from gimbal lock (hinge_axes) and every frame's
rotation is re-expressed in it. The root's channels become a free joint. Frames from FIRST (default 1: CMU's
conversions open on a rest pose), FRAMES of them (default all). BVH is y-up, MuJoCo z-up: C is +90 deg about x, (x, y, z) -> (x, -z, y),
on offsets, hinge axes and root position; orientations become C q C^-1. Units: SCALE m per BVH unit, default CMU's
0.056444 ((1 / 0.45) x 2.54 / 100, mocap.cs.cmu.edu FAQ).

usage: bvh mjcf IN.bvh OUT.xml   a convex capsule mesh per bone (a sphere-like hull on bodies without one), servos,
                                 floor pairs, no self-collision, root assist (write_mjcf)
       bvh clip IN.bvh OUT       OUT.qref: int32 T, NQ; float32 hz; float32 qpos[T][NQ]
                                 OUT.mclip: forward kinematics of the BVH itself, contacts on the four foot bodies
                                 (z within 3 cm of that body's 10th-percentile z in the clip, horizontal speed under 0.5 m/s)
build: mojo build -I kinestride kinestride/bvh.mojo
"""
from std.math import asin, atan2, cos, pi, round, sin, sqrt
from std.os import getenv
from std.sys import argv

comptime Q = SIMD[DType.float64, 4]  # w x y z
comptime FEET: List[String] = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]


@fieldwise_init
struct Skel(Movable):
    var name: List[String]
    var parent: List[Int]
    var offset: List[Float64]  # [joint][3], BVH units and frame
    var axes: List[Int]  # [joint][3] rotation axis per channel, 0 x 1 y 2 z
    var end_parent: List[Int]
    var end_offset: List[Float64]  # [end site][3]
    var t: Int
    var hz: Float64
    var motion: List[Float64]  # [t][6 + 3 (joints - 1)]


def env(name: String, default: Float64) raises -> Float64:
    var v = getenv(name)
    return atof(v) if v != "" else default


def parse(path: String) raises -> Skel:
    var tok = open(path, "r").read().split()
    var name = List[String]()
    var parent = List[Int]()
    var offset = List[Float64]()
    var axes = List[Int]()
    var end_parent = List[Int]()
    var end_offset = List[Float64]()
    var stack = List[Int]()  # open braces: joint index, -1 for an End Site
    var pending = -1
    var in_end = False
    var i = 0
    while String(tok[i]) != "MOTION":
        var w = String(tok[i])
        if w == "ROOT" or w == "JOINT":
            parent.append(stack[len(stack) - 1] if len(stack) > 0 else -1)
            name.append(String(tok[i + 1]))
            pending = len(name) - 1
            in_end = False
            i += 2
        elif w == "End":
            end_parent.append(stack[len(stack) - 1])
            in_end = True
            i += 2
        elif w == "HIERARCHY":
            i += 1
        elif w == "{":
            stack.append(-1 if in_end else pending)
            i += 1
        elif w == "}":
            _ = stack.pop()
            i += 1
        elif w == "OFFSET":
            for k in range(3):
                (end_offset if in_end else offset).append(atof(String(tok[i + 1 + k])))
            i += 4
        elif w == "CHANNELS":
            var n = Int(atof(String(tok[i + 1])))
            var root = len(name) == 1
            if n != (6 if root else 3):
                raise Error(name[len(name) - 1] + ": " + String(n) + " channels, expected 6 on the root and 3 elsewhere")
            for k in range(n):
                var ch = String(tok[i + 2 + k])
                if root and k < 3:
                    if ch != ["Xposition", "Yposition", "Zposition"][k]:
                        raise Error("root channels must start Xposition Yposition Zposition, got " + ch)
                elif ch == "Xrotation":
                    axes.append(0)
                elif ch == "Yrotation":
                    axes.append(1)
                elif ch == "Zrotation":
                    axes.append(2)
                else:
                    raise Error(name[len(name) - 1] + ": unexpected channel " + ch)
            i += 2 + n
        else:
            raise Error("unexpected token " + w)
    var t = Int(atof(String(tok[i + 2])))  # MOTION Frames: T Frame Time: dt
    var hz = 1.0 / atof(String(tok[i + 5]))
    i += 6
    var nch = 3 + 3 * len(name)
    if len(tok) - i != t * nch:
        raise Error(path + ": " + String(len(tok) - i) + " motion values, expected " + String(t * nch))
    var first = Int(env("FIRST", 1))  # CMU's BVH conversions open on a rest-pose frame, not motion
    var last = min(t, first + Int(env("FRAMES", Float64(t))))  # FRAMES: a shorter clip for quick gates
    var motion = List[Float64](capacity=(last - first) * nch)
    for k in range(first * nch, last * nch):
        motion.append(atof(String(tok[i + k])))
    return Skel(name^, parent^, offset^, axes^, end_parent^, end_offset^, last - first, hz, motion^)


def qmul(a: Q, b: Q) -> Q:
    return Q(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
             a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
             a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
             a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def rotate(q: Q, x: Float64, y: Float64, z: Float64) -> Q:
    """Rotate (x, y, z) by q: q (0, v) q^-1, returned as (0, x, y, z)."""
    return qmul(qmul(q, Q(0, x, y, z)), Q(q[0], -q[1], -q[2], -q[3]))


def euler(sk: Skel, j: Int, frame: Int) -> Q:
    """Joint j's local rotation at a frame: its channels applied in order (intrinsic)."""
    var q = Q(1, 0, 0, 0)
    var nch = 3 + 3 * len(sk.name)
    for c in range(3):
        var half = sk.motion[frame * nch + 3 + 3 * j + c] * pi / 360
        var a = Q(cos(half), 0, 0, 0)
        a[1 + sk.axes[3 * j + c]] = sin(half)
        q = qmul(q, a)
    return q


def to_mj(v: Q) -> Q:
    """C: (x, y, z) -> (x, -z, y) on the vector part."""
    return Q(v[0], v[1], -v[3], v[2])


comptime QC = Q(0.70710678118654752, 0.70710678118654752, 0, 0)  # C as a quaternion
comptime QCI = Q(0.70710678118654752, -0.70710678118654752, 0, 0)


def scale() raises -> Float64:
    var s = getenv("SCALE")
    return atof(s) if s != "" else 0.056444


def num(x: Float64) -> String:
    return String(Float32(x))


def vec(v: Q, s: Float64) -> String:
    return num(v[1] * s) + " " + num(v[2] * s) + " " + num(v[3] * s)


def child(sk: Skel, k: Int) -> Q:
    """Offset of child k (a joint, or end site k - joints) in its parent's frame, MuJoCo axes, BVH units."""
    var nj = len(sk.name)
    if k < nj:
        return to_mj(Q(0, sk.offset[3 * k], sk.offset[3 * k + 1], sk.offset[3 * k + 2]))
    var e = k - nj
    return to_mj(Q(0, sk.end_offset[3 * e], sk.end_offset[3 * e + 1], sk.end_offset[3 * e + 2]))


def hull(tip: Q, r: Float64) -> String:
    """Vertices of a capsule's convex stand-in (body frame, m): a ring of 8 at each end and a pole beyond each end, or 6
    poles and 8 corners for a sphere (tip 0). Plane-mesh contact is ported; plane-capsule is not."""
    var pts = List[Q]()
    var length = sqrt(tip[1] * tip[1] + tip[2] * tip[2] + tip[3] * tip[3])
    if length == 0:
        for k in range(3):
            for sg in range(2):
                var p = Q(0, 0, 0, 0)
                p[1 + k] = r if sg == 0 else -r
                pts.append(p)
        var c = r / sqrt(3.0)
        for k in range(8):
            pts.append(Q(0, c if k & 1 else -c, c if k & 2 else -c, c if k & 4 else -c))
    else:
        var u = tip / length
        var a = Q(0, 0, 1, 0) if abs(u[1]) > 0.9 else Q(0, 1, 0, 0)
        var v1 = Q(0, u[2] * a[3] - u[3] * a[2], u[3] * a[1] - u[1] * a[3], u[1] * a[2] - u[2] * a[1])
        v1 = v1 / sqrt(v1[1] * v1[1] + v1[2] * v1[2] + v1[3] * v1[3])
        var v2 = Q(0, u[2] * v1[3] - u[3] * v1[2], u[3] * v1[1] - u[1] * v1[3], u[1] * v1[2] - u[2] * v1[1])
        for e in range(2):
            var o = tip * Float64(e)
            for k in range(8):
                var ang = 2 * pi * Float64(k) / 8
                pts.append(o + v1 * (r * cos(ang)) + v2 * (r * sin(ang)))
        pts.append(u * -r)
        pts.append(tip + u * r)
    var x = String()
    for p in pts:
        x += vec(p, 1) + " "
    return x


def rm(q: Q, r: Int, c: Int) -> Float64:
    """Entry (r, c) of q's rotation matrix."""
    var w = q[0]
    if r == c:
        var a = q[1 + (r + 1) % 3]
        var b = q[1 + (r + 2) % 3]
        return 1 - 2 * (a * a + b * b)
    var k = 3 - r - c  # the third axis
    var sg = 1.0 if c == (r + 1) % 3 else -1.0  # (0,1) (1,2) (2,0): 2 (ab - w k); the transposes: 2 (ab + w k)
    return 2 * (q[1 + r] * q[1 + c] - sg * w * q[1 + k])


def angles(q: Q, i: Int, j: Int, k: Int) -> Q:
    """Intrinsic Tait-Bryan angles of q for axes i, j, k (R = Ri(a) Rj(b) Rk(c)), as (a, b, c, 0)."""
    var e = 1.0 if j == (i + 1) % 3 else -1.0
    return Q(atan2(-e * rm(q, j, k), rm(q, k, k)), asin(max(-1.0, min(1.0, e * rm(q, i, k)))),
             atan2(-e * rm(q, i, j), rm(q, i, i)), 0)


def hinge_axes(sk: Skel) -> List[Int]:
    """Per joint, the axis order of its three hinges: of the 6 Tait-Bryan orders, the one whose middle angle stays
    farthest from +-90 deg (gimbal lock) over the clip. The BVH's own order (Z Y X) sends CMU shoulders and thumbs
    through lock, where z and x swing by pi between frames."""
    comptime P: List[Int] = [0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 2, 1, 2, 1, 0, 1, 0, 2]
    var ax = List[Int](length=3 * len(sk.name), fill=0)
    for j in range(1, len(sk.name)):
        var best = 0
        var worst_best = 10.0
        for o in range(6):
            var worst = 0.0
            for f in range(sk.t):
                worst = max(worst, abs(angles(euler(sk, j, f), materialize[P]()[3 * o], materialize[P]()[3 * o + 1], materialize[P]()[3 * o + 2])[1]))
            if worst < worst_best:
                best = o
                worst_best = worst
        for c in range(3):
            ax[3 * j + c] = materialize[P]()[3 * best + c]
    return ax^


def low_z(sk: Skel, j: Int) raises -> Float64:
    """Joint j's 10th-percentile height over the clip (m, MuJoCo z): a foot's resting height above the floor."""
    var s = scale()
    var nch = 3 + 3 * len(sk.name)
    var chain = List[Int]()
    var b = j
    while b > 0:
        chain.append(b)
        b = sk.parent[b]
    var zs = List[Float64](capacity=sk.t)
    for f in range(sk.t):
        var p = Q(0, sk.motion[f * nch] * s, sk.motion[f * nch + 1] * s, sk.motion[f * nch + 2] * s)
        var q = euler(sk, 0, f)
        for i in range(len(chain) - 1, -1, -1):
            var c = chain[i]
            p = p + rotate(q, sk.offset[3 * c] * s, sk.offset[3 * c + 1] * s, sk.offset[3 * c + 2] * s)
            q = qmul(q, euler(sk, c, f))
        zs.append(to_mj(p)[3])
    sort(zs)
    return zs[(sk.t - 1) // 10]


def write_mjcf(sk: Skel, path: String, model: String) raises:
    """The character for tracking: a position servo per hinge, floor pairs on every geom, the total mass set to MASS kg,
    timestep 1 / (SUB x clip hz), and (ASSIST > 0) a mocap body welded to the root with that time constant.
    Gains follow each hinge's inertia: kp = I w^2 on the servo and damping 2 I w on the joint (critically damped at
    w = 2 pi OMEGA Hz; joint damping, unlike a servo's kv, is implicit under Euler, which the assist weld requires),
    I = the subtree's geoms as point masses about the hinge's axis line in the rest pose, plus armature."""
    var s = scale()
    var nj = len(sk.name)
    var ax = hinge_axes(sk)
    var mass = env("MASS", 70)
    var omega = 2 * pi * env("OMEGA", 10)
    var assist = env("ASSIST", 0.005)
    var arm = 0.01
    # geoms: body, tip (body frame, m; 0 for a sphere at the origin), radius; rest pose of every body origin
    var gbody = List[Int]()
    var gtip = List[Q]()
    var grad = List[Float64]()
    var rest = List[Q](length=nj, fill=Q(0, 0, 0, 0))
    for j in range(nj):
        if j > 0:
            rest[j] = rest[sk.parent[j]] + child(sk, j) * s
        var bones = 0
        var foot = 0.0  # a foot's bone reaches the floor: radius = the joint's resting height
        for name in materialize[FEET]():
            if sk.name[j] == name:
                foot = low_z(sk, j)
        for k in range(nj + len(sk.end_parent)):
            if (sk.parent[k] if k < nj else sk.end_parent[k - nj]) != j:
                continue
            var c = child(sk, k) * s
            var length = sqrt(c[1] * c[1] + c[2] * c[2] + c[3] * c[3])
            if length < 1e-6:
                continue
            gbody.append(j)
            gtip.append(c)
            grad.append(min(max(foot if foot > 0 else 0.2 * length, 0.015 if foot == 0 else 0.02), 0.06))
            bones += 1
        if bones == 0:
            gbody.append(j)
            gtip.append(Q(0, 0, 0, 0))
            grad.append(0.03)
    var ng = len(gbody)
    var gmass = List[Float64](capacity=ng)
    var raw = 0.0
    for g in range(ng):
        var t = gtip[g]
        var r = grad[g]
        var length = sqrt(t[1] * t[1] + t[2] * t[2] + t[3] * t[3])  # hull volume: octagon prism + a pyramid to each pole,
        gmass.append(1000 * (2 * sqrt(2.0) * r * r * (length + 2 * r / 3) if length > 0 else 8 / 3 * r * r * r))  # or cube + 6 pyramids
        raw += gmass[g]
    var density = 1000 * mass / raw
    var inertia = List[Float64](length=3 * nj, fill=arm)  # per hinge: subtree geoms as point masses about its axis line, rest pose
    for j in range(1, nj):
        for c in range(3):
            var a = Q(0, 0, 0, 0)
            a[1 + ax[3 * j + c]] = 1
            a = to_mj(a)
            for k in range(ng):
                var b = gbody[k]
                while b > j:
                    b = sk.parent[b]
                if b == j:
                    var d = rest[gbody[k]] + gtip[k] * 0.5 - rest[j]
                    var along = d[1] * a[1] + d[2] * a[2] + d[3] * a[3]
                    inertia[3 * j + c] += gmass[k] * density / 1000 * (d[1] * d[1] + d[2] * d[2] + d[3] * d[3] - along * along)
    var x = String('<mujoco model="') + model + '">\n  <compiler angle="radian"/>\n'
    x += '  <option timestep="' + num(1 / (sk.hz * env("SUB", 4))) + '"/>\n'
    x += '  <default>\n    <joint armature="' + num(arm) + '"/>\n    <geom contype="0" conaffinity="0" density="' + num(density) + '"/>\n  </default>\n'
    x += "  <asset>\n"
    for k in range(ng):
        x += '    <mesh name="m' + String(k) + '" vertex="' + hull(gtip[k], grad[k]) + '"/>\n'
    x += "  </asset>\n"
    x += '  <worldbody>\n    <geom name="floor" type="plane" size="0 0 1"/>\n'
    var stack = List[Int]()
    var g = 0
    for j in range(nj):
        while len(stack) > 0 and stack[len(stack) - 1] != sk.parent[j]:
            _ = stack.pop()
            x += "  " * (len(stack) + 2) + "</body>\n"
        var ind = "  " * (len(stack) + 2)
        x += ind + '<body name="' + sk.name[j] + '" pos="' + (vec(child(sk, j), s) if j > 0 else "0 0 1") + '">\n'
        if j == 0:
            x += ind + '  <freejoint name="root"/>\n'
        else:
            for c in range(3):
                var a = Q(0, 0, 0, 0)
                a[1 + ax[3 * j + c]] = 1
                x += ind + '  <joint name="' + sk.name[j] + "_" + ["x", "y", "z"][ax[3 * j + c]] + '" axis="' + vec(to_mj(a), 1) + '" damping="' + num(2 * inertia[3 * j + c] * omega) + '"/>\n'
        while g < ng and gbody[g] == j:
            x += ind + '  <geom name="g' + String(g) + '" type="mesh" mesh="m' + String(g) + '"/>\n'
            g += 1
        stack.append(j)
    while len(stack) > 0:
        _ = stack.pop()
        x += "  " * (len(stack) + 2) + "</body>\n"
    if assist > 0:
        x += '    <body name="assist" mocap="true" pos="0 0 1"><geom type="sphere" size="0.01" mass="0"/></body>\n'
    x += "  </worldbody>\n  <contact>\n"
    for k in range(ng):
        x += '    <pair geom1="floor" geom2="g' + String(k) + '"/>\n'
    x += "  </contact>\n  <actuator>\n"
    for j in range(1, nj):
        for c in range(3):
            var jn = sk.name[j] + "_" + ["x", "y", "z"][ax[3 * j + c]]
            x += '    <position joint="' + jn + '" kp="' + num(inertia[3 * j + c] * omega * omega) + '"/>\n'
    x += "  </actuator>\n"
    if assist > 0:
        x += '  <equality>\n    <weld body1="' + sk.name[0] + '" body2="assist" solref="' + num(assist) + ' 1"/>\n  </equality>\n'
    x += "</mujoco>\n"
    var fh = open(path, "w")
    fh.write(x)


def write_f32(mut fh: FileHandle, v: List[Float32]) raises:
    fh.write_bytes(Span(unsafe_ptr=v.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(v) * 4))


def write_clip(sk: Skel, dst: String) raises:
    var s = scale()
    var nj = len(sk.name)
    var ax = hinge_axes(sk)
    var t = sk.t
    var nq = 7 + 3 * (nj - 1)
    var nch = 3 + 3 * nj
    var qref = List[Float32](capacity=t * nq)
    var pos = List[Float64](length=t * nj * 3, fill=0.0)
    var quat = List[Float32](capacity=t * nj * 4)
    var wq = List[Q](length=nj, fill=Q(1, 0, 0, 0))
    var wp = List[Q](length=nj, fill=Q(0, 0, 0, 0))
    var last = List[Float64](length=3 * nj, fill=0.0)
    for f in range(t):
        for j in range(nj):
            var q = euler(sk, j, f)
            if j == 0:
                wp[0] = Q(0, sk.motion[f * nch] * s, sk.motion[f * nch + 1] * s, sk.motion[f * nch + 2] * s)
                wq[0] = q
            else:
                var p = sk.parent[j]
                var o = rotate(wq[p], sk.offset[3 * j] * s, sk.offset[3 * j + 1] * s, sk.offset[3 * j + 2] * s)
                wp[j] = wp[p] + o
                wq[j] = qmul(wq[p], q)
            var mp = to_mj(wp[j])
            var mq = qmul(qmul(QC, wq[j]), QCI)
            for k in range(3):
                pos[(f * nj + j) * 3 + k] = mp[1 + k]
            for k in range(4):
                quat.append(Float32(mq[k]))
            if j == 0:
                for k in range(3):
                    qref.append(Float32(mp[1 + k]))
                for k in range(4):
                    qref.append(Float32(mq[k]))
            else:  # hinge qpos in hinge_axes' order, continuous: of the two solutions for one rotation, (a, b, c) and
                # (a + pi, pi - b, c + pi), each unwrapped by whole turns, the one nearer the previous frame
                var best = Q(0, 0, 0, 0)
                var dbest = 1e30
                for alt in range(2):
                    var cand = Q(0, 0, 0, 0)
                    var dist = 0.0
                    var e3 = angles(q, ax[3 * j], ax[3 * j + 1], ax[3 * j + 2])
                    for c in range(3):
                        var ang = e3[c]
                        if alt == 1:
                            ang = (pi - ang) if c == 1 else ang + pi
                        if f > 0:
                            ang += 2 * pi * round((last[3 * j + c] - ang) / (2 * pi))
                            dist += abs(ang - last[3 * j + c])
                        cand[c] = ang
                    if dist < dbest:
                        best = cand
                        dbest = dist
                for c in range(3):
                    last[3 * j + c] = best[c]
                    qref.append(Float32(best[c]))
    var feet = List[Int]()
    for name in materialize[FEET]():
        for j in range(nj):
            if sk.name[j] == name:
                feet.append(j)
    var nf = len(feet)
    var contact = List[Float32](length=t * nf, fill=0.0)
    for fi in range(nf):
        var b = feet[fi]
        var zs = List[Float64](capacity=t)
        for f in range(t):
            zs.append(pos[(f * nj + b) * 3 + 2])
        sort(zs)
        var zlow = zs[(t - 1) // 10]  # 10th percentile, lower order statistic: the clip minimum is a mocap dip
        for f in range(t):
            var g = f + 1 if f + 1 < t else f - 1  # forward difference, backward on the last frame
            var dx = pos[(g * nj + b) * 3] - pos[(f * nj + b) * 3]
            var dy = pos[(g * nj + b) * 3 + 1] - pos[(f * nj + b) * 3 + 1]
            var speed = sqrt(dx * dx + dy * dy) * sk.hz
            if t > 1 and pos[(f * nj + b) * 3 + 2] < zlow + 0.03 and speed < 0.5:
                contact[f * nf + fi] = 1
    var fq = open(dst + ".qref", "w")
    var head: List[Int32] = [Int32(t), Int32(nq)]
    fq.write_bytes(Span(unsafe_ptr=head.unsafe_ptr().unsafe_bitcast[UInt8](), length=8))
    write_f32(fq, [Float32(sk.hz)])
    write_f32(fq, qref)
    var fm = open(dst + ".mclip", "w")
    var mh: List[Int32] = [Int32(t), Int32(nj), Int32(nf)]
    fm.write_bytes(Span(unsafe_ptr=mh.unsafe_ptr().unsafe_bitcast[UInt8](), length=12))
    write_f32(fm, [Float32(sk.hz)])
    var fidx = List[Int32]()
    for b in feet:
        fidx.append(Int32(b))
    fm.write_bytes(Span(unsafe_ptr=fidx.unsafe_ptr().unsafe_bitcast[UInt8](), length=nf * 4))
    var p32 = List[Float32](capacity=len(pos))
    for v in pos:
        p32.append(Float32(v))
    write_f32(fm, p32)
    write_f32(fm, quat)
    write_f32(fm, contact)
    print("clip", dst, "frames", t, "hz", sk.hz, "bodies", nj, "nq", nq, "feet", nf)


def main() raises:
    var args = argv()
    if len(args) != 4 or (String(args[1]) != "mjcf" and String(args[1]) != "clip"):
        raise Error("usage: bvh mjcf IN.bvh OUT.xml | bvh clip IN.bvh OUT")
    var sk = parse(String(args[2]))
    if String(args[1]) == "mjcf":
        var stem = String(args[2]).split("/")
        write_mjcf(sk, String(args[3]), String(stem[len(stem) - 1]).replace(".bvh", ""))
        print("mjcf", String(args[3]), "bodies", len(sk.name))
    else:
        write_clip(sk, String(args[3]))
