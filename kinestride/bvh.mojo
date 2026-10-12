"""BVH mocap -> an MJCF character and its reference clip (docs/plans/MOTION_DATASET.md, unit 2).

Every BVH joint becomes a body and its rotation channels become hinges in channel order: BVH applies them intrinsically
(Z Y X: R = Rz Ry Rx) and MuJoCo applies a body's joints in definition order, so a frame's Euler angles ARE the hinge
qpos. The root's channels become a free joint. BVH is y-up, MuJoCo z-up: C is +90 deg about x, (x, y, z) -> (x, -z, y),
on offsets, hinge axes and root position; orientations become C q C^-1. Units: SCALE m per BVH unit, default CMU's
0.056444 ((1 / 0.45) x 2.54 / 100, mocap.cs.cmu.edu FAQ).

usage: bvh mjcf IN.bvh OUT.xml   capsule per bone, a sphere on bodies without one, no self-collision, a floor
       bvh clip IN.bvh OUT       OUT.qref: int32 T, NQ; float32 hz; float32 qpos[T][NQ]
                                 OUT.mclip: forward kinematics of the BVH itself, contacts on the four foot bodies
                                 (z within 3 cm of that body's 10th-percentile z in the clip, horizontal speed under 0.5 m/s)
build: mojo build -I kinestride kinestride/bvh.mojo
"""
from std.math import cos, pi, sin, sqrt
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
    var motion = List[Float64](capacity=t * nch)
    for k in range(t * nch):
        motion.append(atof(String(tok[i + k])))
    return Skel(name^, parent^, offset^, axes^, end_parent^, end_offset^, t, hz, motion^)


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


def write_mjcf(sk: Skel, path: String, model: String) raises:
    var s = scale()
    var nj = len(sk.name)
    var x = String('<mujoco model="') + model + '">\n  <compiler angle="radian"/>\n'
    x += '  <default>\n    <joint damping="0.5" armature="0.01"/>\n    <geom contype="1" conaffinity="0" density="1000"/>\n  </default>\n'
    x += '  <worldbody>\n    <geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="1"/>\n'
    var stack = List[Int]()
    for j in range(nj):
        while len(stack) > 0 and stack[len(stack) - 1] != sk.parent[j]:
            _ = stack.pop()
            x += "  " * (len(stack) + 2) + "</body>\n"
        var ind = "  " * (len(stack) + 2)
        var o = to_mj(Q(0, sk.offset[3 * j], sk.offset[3 * j + 1], sk.offset[3 * j + 2]))
        x += ind + '<body name="' + sk.name[j] + '" pos="' + (vec(o, s) if j > 0 else "0 0 1") + '">\n'
        if j == 0:
            x += ind + '  <freejoint name="root"/>\n'
        else:
            for c in range(3):
                var a = Q(0, 0, 0, 0)
                a[1 + sk.axes[3 * j + c]] = 1
                x += ind + '  <joint name="' + sk.name[j] + "_" + ["x", "y", "z"][sk.axes[3 * j + c]] + '" axis="' + vec(to_mj(a), 1) + '"/>\n'
        var bones = 0
        for k in range(nj + len(sk.end_parent)):  # child joints, then end sites
            var e = k - nj
            if (sk.parent[k] if e < 0 else sk.end_parent[e]) != j:
                continue
            var c = Q(0, sk.offset[3 * k], sk.offset[3 * k + 1], sk.offset[3 * k + 2]) if e < 0 else Q(
                0, sk.end_offset[3 * e], sk.end_offset[3 * e + 1], sk.end_offset[3 * e + 2])
            var length = sqrt(c[1] * c[1] + c[2] * c[2] + c[3] * c[3]) * s
            if length < 1e-6:
                continue
            x += ind + '  <geom type="capsule" fromto="0 0 0 ' + vec(to_mj(c), s) + '" size="' + num(min(max(0.2 * length, 0.015), 0.06)) + '"/>\n'
            bones += 1
        if bones == 0:
            x += ind + '  <geom type="sphere" size="0.03"/>\n'
        stack.append(j)
    while len(stack) > 0:
        _ = stack.pop()
        x += "  " * (len(stack) + 2) + "</body>\n"
    x += "  </worldbody>\n</mujoco>\n"
    var fh = open(path, "w")
    fh.write(x)


def write_f32(mut fh: FileHandle, v: List[Float32]) raises:
    fh.write_bytes(Span(unsafe_ptr=v.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(v) * 4))


def write_clip(sk: Skel, dst: String) raises:
    var s = scale()
    var nj = len(sk.name)
    var t = sk.t
    var nq = 7 + 3 * (nj - 1)
    var nch = 3 + 3 * nj
    var qref = List[Float32](capacity=t * nq)
    var pos = List[Float64](length=t * nj * 3, fill=0.0)
    var quat = List[Float32](capacity=t * nj * 4)
    var wq = List[Q](length=nj, fill=Q(1, 0, 0, 0))
    var wp = List[Q](length=nj, fill=Q(0, 0, 0, 0))
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
            else:
                for c in range(3):
                    qref.append(Float32(sk.motion[f * nch + 3 + 3 * j + c] * pi / 180))
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
