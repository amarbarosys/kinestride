"""BVH mocap -> one canonical character and per-clip reference motion (docs/MOTION.md, provenance docs/provenance/motion-godot.md).

Any BVH skeleton is parsed into the MuJoCo frame (BVH is y-up: C = +90 deg about x, (x, y, z) -> (x, -z, y); SCALE m per
unit, default CMU's 0.056444) and retargeted onto the 21-body profile below: each canonical body follows the world
rotation of its source joint, corrected at rest so the canonical bone lies along the source bone (R), and, where
unmapped joints sit between a body and its main child, re-aimed along the current source chain (A); the root is scaled
by hip height, then the clip is grounded (the lower toe's 10th-percentile height = the foot radius). Unmapped joints (fingers, CMU's LHipJoint, LowerBack, Neck) vanish. Hinge angles are intrinsic
Tait-Bryan angles in each body's frozen axis order, kept on the branch nearest the previous frame. Before the retarget,
isolated spikes in the source are repaired (repair; REPAIR=0 skips it).
Frames from FIRST (default 1: CMU's conversions open on a rest pose), FRAMES of them (default all).

usage: bvh mjcf OUT.xml          the character: a convex capsule mesh per body, servos, floor pairs, root assist
       bvh clip IN.bvh OUT       OUT.qref: int32 T, NQ; float32 hz; float32 qpos[T][NQ]
                                 OUT.mclip: the canonical bodies' forward kinematics, contacts on the four foot bodies
                                 (contacts(): taken before planting, which holds each contact run's limb end still)
build: mojo build -I kinestride kinestride/bvh.mojo
"""
from std.math import asin, atan2, cos, pi, round, sin, sqrt
from std.os import getenv
from std.sys import argv

from max.algorithm import parallelize

comptime Q = SIMD[DType.float64, 4]  # w x y z; vectors as (0, x, y, z)

# The profile, depth-first. MuJoCo frame, rest pose (T-pose, facing -y, left +x), metres; per body: name, source joint
# candidates (after stripping "mixamorig:"; first found wins), parent, offset from the parent, bone vector (capsule and
# retarget direction), main child (-1: a leaf), capsule radius, hinge axis order (0 x, 1 y, 2 z).
comptime NAMES: List[String] = ["Hips", "Spine", "Chest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes", "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"]
comptime ALIAS: List[String] = ["Hips", "Spine", "Spine2|Spine1", "Neck1|Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase|LeftToe", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase|RightToe"]
comptime PARENT: List[Int] = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
comptime MAIN: List[Int] = [1, 2, 3, 4, -1, 6, 7, 8, -1, 10, 11, 12, -1, 14, 15, 16, -1, 18, 19, 20, -1]
comptime OFFSET: List[Float64] = [0, 0, 0, 0, 0, 0.10, 0, 0, 0.20, 0, 0, 0.22, 0, 0, 0.10,
    0.02, 0, 0.19, 0.16, 0, 0, 0.28, 0, 0, 0.25, 0, 0, -0.02, 0, 0.19, -0.16, 0, 0, -0.28, 0, 0, -0.25, 0, 0,
    0.09, 0, -0.07, 0, 0, -0.43, 0, 0, -0.42, 0, -0.13, -0.045, -0.09, 0, -0.07, 0, 0, -0.43, 0, 0, -0.42, 0, -0.13, -0.045]
comptime TIP: List[Float64] = [0, 0, 0.10, 0, 0, 0.20, 0, 0, 0.22, 0, 0, 0.10, 0, 0, 0.18,
    0.16, 0, 0, 0.28, 0, 0, 0.25, 0, 0, 0.18, 0, 0, -0.16, 0, 0, -0.28, 0, 0, -0.25, 0, 0, -0.18, 0, 0,
    0, 0, -0.43, 0, 0, -0.42, 0, -0.13, -0.045, 0, -0.06, 0, 0, 0, -0.43, 0, 0, -0.42, 0, -0.13, -0.045, 0, -0.06, 0]
comptime RADIUS: List[Float64] = [0.11, 0.10, 0.12, 0.05, 0.09, 0.05, 0.045, 0.04, 0.035, 0.05, 0.045, 0.04, 0.035,
    0.07, 0.05, 0.035, 0.03, 0.07, 0.05, 0.035, 0.03]
comptime ORDER: List[Int] = [2, 1, 0, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1,  # frozen from CMU 02_05 10_01 55_22 60_01:
    0, 2, 1, 2, 0, 1, 2, 1, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2, 1, 0, 0, 2, 1,  # worst middle angle 70 deg (RightUpperArm)
    0, 2, 1, 1, 2, 0, 2, 1, 0, 2, 1, 0, 0, 2, 1, 1, 2, 0, 2, 1, 0, 2, 1, 0]
comptime FEET: List[Int] = [15, 16, 19, 20]
comptime LIMBS: List[Int] = [13, 14, 15, 17, 18, 19, 6, 7, 8, 10, 11, 12]  # upper, lower, end: legs, then arms
comptime NB = 21


@fieldwise_init
struct Clip(Movable):
    """A skeleton in the MuJoCo frame (m) and its motion: local rotations and root position per frame."""
    var name: List[String]
    var parent: List[Int]
    var offset: List[Q]  # from the parent, in its frame
    var tip: List[Q]  # rest vector from the joint to the end of its first-child chain
    var t: Int
    var hz: Float64
    var root: List[Q]  # [t]
    var rot: List[Q]  # [t][joints]


def env(name: String, default: Float64) raises -> Float64:
    var v = getenv(name)
    return atof(v) if v != "" else default


def qmul(a: Q, b: Q) -> Q:
    return Q(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
             a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
             a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
             a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def conj(q: Q) -> Q:
    return Q(q[0], -q[1], -q[2], -q[3])


def rot(q: Q, v: Q) -> Q:
    return qmul(qmul(q, v), conj(q))


def unit(v: Q) -> Q:
    return v / sqrt((v * v).reduce_add())


def rmin(a: Q, b: Q) -> Q:
    """The smallest rotation taking unit vector a to unit vector b."""
    var d = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if d < -0.999999:  # opposite: half a turn about any perpendicular axis
        return unit(Q(0, 0, -a[3], a[2]) if abs(a[1]) < 0.9 else Q(0, a[3], 0, -a[1]))
    return unit(Q(1 + d, a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]))


def to_mj(v: Q) -> Q:
    """C on a vector: (x, y, z) -> (x, -z, y)."""
    return Q(v[0], v[1], -v[3], v[2])


comptime QC = Q(0.70710678118654752, 0.70710678118654752, 0, 0)  # C as a quaternion


def row(table: List[Float64], i: Int) -> Q:
    return Q(0, table[3 * i], table[3 * i + 1], table[3 * i + 2])


def parse(path: String) raises -> Clip:
    var tok = open(path, "r").read().split()
    var s = env("SCALE", 0.056444)
    var name = List[String]()
    var parent = List[Int]()
    var offset = List[Q]()
    var endo = List[Q]()  # end site offset per joint (0 if none)
    var axes = List[Int]()
    var stack = List[Int]()  # open braces: joint index, -1 for an End Site
    var in_end = False
    var i = 0
    while String(tok[i]) != "MOTION":
        var w = String(tok[i])
        if w == "ROOT" or w == "JOINT":
            parent.append(stack[len(stack) - 1] if len(stack) > 0 else -1)
            name.append(String(tok[i + 1]).replace("mixamorig:", ""))
            endo.append(Q(0, 0, 0, 0))
            in_end = False
            i += 2
        elif w == "End":
            in_end = True
            i += 2
        elif w == "HIERARCHY":
            i += 1
        elif w == "{":
            stack.append(-1 if in_end else len(name) - 1)
            i += 1
        elif w == "}":
            _ = stack.pop()
            i += 1
        elif w == "OFFSET":
            var o = to_mj(Q(0, atof(String(tok[i + 1])), atof(String(tok[i + 2])), atof(String(tok[i + 3])))) * s
            if in_end:
                endo[len(name) - 1] = o
            else:
                offset.append(o)
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
                elif ch == "Xrotation" or ch == "Yrotation" or ch == "Zrotation":
                    axes.append(0 if ch == "Xrotation" else (1 if ch == "Yrotation" else 2))
                else:
                    raise Error(name[len(name) - 1] + ": unexpected channel " + ch)
            i += 2 + n
        else:
            raise Error("unexpected token " + w)
    var nj = len(name)
    var tip = endo.copy()  # depth-first order: children come after their parent, so fill from the end
    for j in range(nj - 1, 0, -1):
        var p = parent[j]
        if p >= 0 and tip[p][1] == 0 and tip[p][2] == 0 and tip[p][3] == 0 and (j == p + 1):
            tip[p] = offset[j] + tip[j]
    var t = Int(atof(String(tok[i + 2])))  # MOTION Frames: T Frame Time: dt
    var hz = 1.0 / atof(String(tok[i + 5]))
    i += 6
    var nch = 3 + 3 * nj
    if len(tok) - i != t * nch:
        raise Error(path + ": " + String(len(tok) - i) + " motion values, expected " + String(t * nch))
    var first = Int(env("FIRST", 1))  # CMU's BVH conversions open on a rest-pose frame, not motion
    var last = min(t, first + Int(env("FRAMES", Float64(t))))
    var rootp = List[Q](capacity=last - first)
    var rots = List[Q](capacity=(last - first) * nj)
    for f in range(first, last):
        var b = i + f * nch
        rootp.append(to_mj(Q(0, atof(String(tok[b])), atof(String(tok[b + 1])), atof(String(tok[b + 2])))) * s)
        for j in range(nj):
            var q = Q(1, 0, 0, 0)
            for c in range(3):
                var half = atof(String(tok[b + 3 + 3 * j + c])) * pi / 360
                var a = Q(cos(half), 0, 0, 0)
                a[1 + axes[3 * j + c]] = sin(half)
                q = qmul(q, a)
            rots.append(qmul(qmul(QC, q), conj(QC)))
    return Clip(name^, parent^, offset^, tip^, last - first, hz, rootp^, rots^)


def fk(c: Clip, f: Int, mut wp: List[Q], mut wq: List[Q]):
    var n = len(c.name)
    for j in range(n):
        if j == 0:
            wp[0] = c.root[f]
            wq[0] = c.rot[f * n]
        else:
            var p = c.parent[j]
            wp[j] = wp[p] + rot(wq[p], c.offset[j])
            wq[j] = qmul(wq[p], c.rot[f * n + j])


def lowest(pos: List[Q], tip: List[Q]) -> Float64:
    var z = 1e30
    for j in range(len(pos)):
        z = min(z, min(pos[j][3], pos[j][3] + tip[j][3]))
    return z


def angle(a: Q, b: Q) -> Float64:
    """Rotation angle between two orientations."""
    var d = qmul(conj(a), b)
    return 2 * atan2(sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3]), abs(d[0]))


def repair(mut c: Clip) raises -> Int:
    """Spike repair, non-causal: a joint's rotation at frame f that is more than CAP x dt (rad/s, default 30) from both
    neighbours while the neighbours are within CAP x dt of each other is replaced by their midpoint. Judged on the
    unrepaired clip, so the result does not depend on order; every other frame is untouched. Returns the count."""
    var lim = env("CAP", 30) / c.hz
    var n = len(c.name)
    var fix = List[Int]()
    for f in range(1, c.t - 1):
        for j in range(n):
            var a = c.rot[(f - 1) * n + j]
            var m = c.rot[f * n + j]
            var b = c.rot[(f + 1) * n + j]
            if angle(a, m) > lim and angle(m, b) > lim and angle(a, b) <= lim:
                fix.append(f * n + j)
    for i in fix:
        var a = c.rot[i - n]
        var b = c.rot[i + n]
        c.rot[i] = unit(a + b if (a * b).reduce_add() >= 0 else a - b)
    return len(fix)


def retarget(src: Clip) raises -> Clip:
    var names = materialize[NAMES]()
    var alias = materialize[ALIAS]()
    var main = materialize[MAIN]()
    var ns = len(src.name)
    var smap = List[Int](capacity=NB)
    for b in range(NB):
        var found = -1
        for cand in alias[b].split("|"):
            for j in range(ns):
                if found < 0 and src.name[j] == String(cand):
                    found = j
        if found < 0:
            raise Error("no source joint for " + names[b] + " (candidates " + alias[b] + ")")
        smap.append(found)
    var offset = List[Q](capacity=NB)
    var tip = List[Q](capacity=NB)
    var crest = List[Q](length=NB, fill=Q(0, 0, 0, 0))
    for b in range(NB):
        offset.append(row(materialize[OFFSET](), b))
        tip.append(row(materialize[TIP](), b))
        if b > 0:
            crest[b] = crest[materialize[PARENT]()[b]] + offset[b]
    var srest = List[Q](length=ns, fill=Q(0, 0, 0, 0))
    for j in range(1, ns):
        srest[j] = srest[src.parent[j]] + src.offset[j]
    var r = List[Q](capacity=NB)  # rest correction: canonical bone onto the source bone
    for b in range(NB):
        var d = (srest[smap[main[b]]] - srest[smap[b]]) if main[b] >= 0 else src.tip[smap[b]]
        r.append(rmin(unit(tip[b]), unit(d)))
    var ratio = (crest[0][3] - lowest(crest, tip)) / (srest[smap[0]][3] - lowest(srest, src.tip))
    var parent = materialize[PARENT]()
    var rootp = List[Q](length=src.t, fill=Q(0, 0, 0, 0))
    var rots = List[Q](length=src.t * NB, fill=Q(1, 0, 0, 0))
    var low = List[Float64](length=src.t, fill=0.0)  # the lower toe's height per frame, for grounding
    var pp = rootp.unsafe_ptr()
    var rp = rots.unsafe_ptr()
    var lp = low.unsafe_ptr()
    var feet = materialize[FEET]()

    def frames(k: Int) {imm src, imm smap, imm main, imm parent, imm tip, imm offset, imm feet, imm r, imm ratio, imm ns, imm pp, imm rp, imm lp}:
        var swp = List[Q](length=ns, fill=Q(0, 0, 0, 0))
        var swq = List[Q](length=ns, fill=Q(1, 0, 0, 0))
        var w = List[Q](length=NB, fill=Q(1, 0, 0, 0))
        var cp = List[Q](length=NB, fill=Q(0, 0, 0, 0))
        for f in range(64 * k, min(src.t, 64 * k + 64)):  # 64 frames per task, each task its own scratch
            fk(src, f, swp, swq)
            for b in range(NB):
                var wb = qmul(swq[smap[b]], r[b])
                if b > 0 and main[b] >= 0:  # re-aim along the source chain when unmapped joints bend inside it
                    wb = qmul(rmin(unit(rot(wb, tip[b])), unit(swp[smap[main[b]]] - swp[smap[b]])), wb)
                w[b] = wb
                rp.unsafe_offset(f * NB + b)[] = wb if b == 0 else qmul(conj(w[parent[b]]), wb)
                cp[b] = swp[smap[0]] * ratio if b == 0 else cp[parent[b]] + rot(w[parent[b]], offset[b])
            pp.unsafe_offset(f)[] = cp[0]
            lp.unsafe_offset(f)[] = min(cp[feet[1]][3], cp[feet[3]][3])

    parallelize(frames, (src.t + 63) // 64)
    var c = Clip(names^, parent^, offset^, tip^, src.t, src.hz, rootp^, rots^)
    sort(low)  # ground the clip: the lower toe's 10th-percentile height = the foot radius
    var lift = materialize[RADIUS]()[feet[0]] - low[(c.t - 1) // 10]
    for f in range(c.t):
        c.root[f][3] += lift
    return c^


def num(x: Float64) -> String:
    return String(Float32(x))


def vec(v: Q) -> String:
    return num(v[1]) + " " + num(v[2]) + " " + num(v[3])


def hull(tip: Q, r: Float64) -> String:
    """Vertices of a capsule's convex stand-in (body frame, m): a ring of 8 at each end and a pole beyond each end.
    Plane-mesh contact is ported; plane-capsule is not."""
    var u = unit(tip)
    var a = Q(0, 0, 1, 0) if abs(u[1]) > 0.9 else Q(0, 1, 0, 0)
    var v1 = unit(Q(0, u[2] * a[3] - u[3] * a[2], u[3] * a[1] - u[1] * a[3], u[1] * a[2] - u[2] * a[1]))
    var v2 = Q(0, u[2] * v1[3] - u[3] * v1[2], u[3] * v1[1] - u[1] * v1[3], u[1] * v1[2] - u[2] * v1[1])
    var x = String()
    for e in range(2):
        for k in range(8):
            var ang = 2 * pi * Float64(k) / 8
            x += vec(tip * Float64(e) + v1 * (r * cos(ang)) + v2 * (r * sin(ang))) + " "
    return x + vec(u * -r) + " " + vec(tip + u * r)


def write_mjcf(path: String) raises:
    """The character: a position servo per hinge, a floor pair per body, total mass MASS kg, timestep 1 / (SUB x HZ), and
    (ASSIST > 0) a mocap body welded to the root with that time constant. Gains follow each hinge's inertia: kp = I w^2
    on the servo, damping 2 I w on the joint (critically damped at w = 2 pi OMEGA Hz; joint damping is implicit under
    Euler, which the weld requires, a servo's kv is not), I = the subtree's capsules as point masses about the hinge
    axis line in the rest pose, plus armature."""
    var parent = materialize[PARENT]()
    var order = materialize[ORDER]()
    var radius = materialize[RADIUS]()
    var omega = 2 * pi * env("OMEGA", 10)
    var assist = env("ASSIST", 0.005)
    var arm = 0.01
    var rest = List[Q](length=NB, fill=Q(0, 0, 0, 0))
    var vol = List[Float64](capacity=NB)
    var total = 0.0
    for b in range(NB):
        if b > 0:
            rest[b] = rest[parent[b]] + row(materialize[OFFSET](), b)
        var t = row(materialize[TIP](), b)
        var r = radius[b]
        vol.append(2 * sqrt(2.0) * r * r * (sqrt((t * t).reduce_add()) + 2 * r / 3))  # octagon prism + a pyramid per pole
        total += vol[b]
    var density = env("MASS", 70) / total
    var inertia = List[Float64](length=3 * NB, fill=arm)
    for b in range(1, NB):
        for c in range(3):
            var a = Q(0, 0, 0, 0)
            a[1 + order[3 * b + c]] = 1
            for k in range(NB):
                var u = k
                while u > b:
                    u = parent[u]
                if u == b:
                    var d = rest[k] + row(materialize[TIP](), k) * 0.5 - rest[b]
                    var along = (d * a).reduce_add()
                    inertia[3 * b + c] += vol[k] * density * ((d * d).reduce_add() - along * along)
    var x = String('<mujoco model="kinestride-humanoid">\n  <compiler angle="radian"/>\n')
    x += '  <option timestep="' + num(1 / (env("HZ", 120) * env("SUB", 4))) + '"/>\n'
    x += '  <default>\n    <joint armature="' + num(arm) + '"/>\n    <geom contype="0" conaffinity="0" density="' + num(density) + '"/>\n  </default>\n  <asset>\n'
    for b in range(NB):
        x += '    <mesh name="m' + String(b) + '" vertex="' + hull(row(materialize[TIP](), b), radius[b]) + '"/>\n'
    x += '  </asset>\n  <worldbody>\n    <geom name="floor" type="plane" size="0 0 1"/>\n'
    var depth = List[Int](length=NB, fill=0)
    for b in range(NB):
        depth[b] = depth[parent[b]] + 1 if b > 0 else 0
        if b > 0:
            for _ in range(depth[b - 1] - depth[b] + 1):
                x += "  " * (depth[b - 1] + 2) + "</body>\n"  # close the bodies the previous one left open
        var ind = "  " * (depth[b] + 2)
        x += ind + '<body name="' + materialize[NAMES]()[b] + '" pos="' + (vec(row(materialize[OFFSET](), b)) if b > 0 else "0 0 1") + '">\n'
        if b == 0:
            x += ind + '  <freejoint name="root"/>\n'
        for c in range(3 if b > 0 else 0):
            var a = Q(0, 0, 0, 0)
            a[1 + order[3 * b + c]] = 1
            x += ind + '  <joint name="' + materialize[NAMES]()[b] + "_" + ["x", "y", "z"][order[3 * b + c]] + '" axis="' + vec(a) + '" damping="' + num(2 * inertia[3 * b + c] * omega) + '"/>\n'
        x += ind + '  <geom name="g' + String(b) + '" type="mesh" mesh="m' + String(b) + '"/>\n'
    for d in range(depth[NB - 1], -1, -1):
        x += "  " * (d + 2) + "</body>\n"
    if assist > 0:
        x += '    <body name="assist" mocap="true" pos="0 0 1"><geom type="sphere" size="0.01" mass="0"/></body>\n'
    x += "  </worldbody>\n  <contact>\n"
    for b in range(NB):
        x += '    <pair geom1="floor" geom2="g' + String(b) + '"/>\n'
    x += "  </contact>\n  <actuator>\n"
    for b in range(1, NB):
        for c in range(3):
            x += '    <position joint="' + materialize[NAMES]()[b] + "_" + ["x", "y", "z"][order[3 * b + c]] + '" kp="' + num(inertia[3 * b + c] * omega * omega) + '"/>\n'
    x += "  </actuator>\n"
    if assist > 0:
        x += '  <equality>\n    <weld body1="Hips" body2="assist" solref="' + num(assist) + ' 1"/>\n  </equality>\n'
    var fh = open(path, "w")
    fh.write(x + "</mujoco>\n")


def rm(q: Q, r: Int, c: Int) -> Float64:
    """Entry (r, c) of q's rotation matrix."""
    if r == c:
        var a = q[1 + (r + 1) % 3]
        var b = q[1 + (r + 2) % 3]
        return 1 - 2 * (a * a + b * b)
    var sg = 1.0 if c == (r + 1) % 3 else -1.0  # (0,1) (1,2) (2,0): 2 (ab - w k); the transposes: 2 (ab + w k)
    return 2 * (q[1 + r] * q[1 + c] - sg * q[0] * q[1 + 3 - r - c])


def angles(q: Q, i: Int, j: Int, k: Int) -> Q:
    """Intrinsic Tait-Bryan angles of q for axes i, j, k (R = Ri(a) Rj(b) Rk(c)), as (a, b, c, 0)."""
    var e = 1.0 if j == (i + 1) % 3 else -1.0
    return Q(atan2(-e * rm(q, j, k), rm(q, k, k)), asin(max(-1.0, min(1.0, e * rm(q, i, k)))),
             atan2(-e * rm(q, i, j), rm(q, i, i)), 0)


def poses(c: Clip, mut wp: List[Q], mut wq: List[Q]):
    """World positions and rotations of every body at every frame ([t][body]), 64 frames per parallel task."""
    var t = c.t
    wp = List[Q](length=t * NB, fill=Q(0, 0, 0, 0))
    wq = List[Q](length=t * NB, fill=Q(1, 0, 0, 0))
    var wpp = wp.unsafe_ptr()
    var wqp = wq.unsafe_ptr()

    def task(k: Int) {imm c, imm t, imm wpp, imm wqp}:
        var p = List[Q](length=NB, fill=Q(0, 0, 0, 0))
        var q = List[Q](length=NB, fill=Q(1, 0, 0, 0))
        for f in range(64 * k, min(t, 64 * k + 64)):
            fk(c, f, p, q)
            for b in range(NB):
                wpp.unsafe_offset(f * NB + b)[] = p[b]
                wqp.unsafe_offset(f * NB + b)[] = q[b]

    parallelize(task, (t + 63) // 64)


def contacts(c: Clip, pos: List[Q]) raises -> List[Float32]:
    """[t][body] contact, on the feet (Foot, Toes: z within 3 cm of that body's 10th-percentile z in the clip) and hands
    (z under HANDZ, default 6.5 cm: the hand radius + 3 cm), both at horizontal speed under 0.5 m/s."""
    var t = c.t
    var con = List[Float32](length=t * NB, fill=0.0)
    var ends: List[Int] = [15, 16, 19, 20, 8, 12]
    for i in range(len(ends)):
        var b = ends[i]
        var zlim = env("HANDZ", 0.065)
        if i < 4:
            var zs = List[Float64](capacity=t)
            for f in range(t):
                zs.append(pos[f * NB + b][3])
            sort(zs)
            var zlow = zs[(t - 1) // 10]  # 10th percentile, lower order statistic: the clip minimum is a mocap dip
            zlim = zlow + 0.03
        for f in range(t):
            var g = f + 1 if f + 1 < t else f - 1  # forward difference, backward on the last frame
            var d = pos[g * NB + b] - pos[f * NB + b]
            if t > 1 and pos[f * NB + b][3] < zlim and sqrt(d[1] * d[1] + d[2] * d[2]) * c.hz < 0.5:
                con[f * NB + b] = 1
    return con^


def plant(mut c: Clip, con: List[Float32], wp: List[Q], wq: List[Q]) raises:
    """Limb planting: in each contact run of a limb's end (Foot for a leg, Hand for an arm; runs closer than 2 BLEND frames
    joined) the end is held at the run's
    mean position by one analytic two-bone solve (law of cosines, bending about the limb's current bend normal); outside runs
    each neighbouring run's edge correction fades out over BLEND frames (default 6), summed, so the limb never jumps; out
    of reach the limb straightens toward the goal; the end body keeps its world orientation. PLANT=0 skips it."""
    var blend = Int(env("BLEND", 6))
    var limbs = materialize[LIMBS]()
    var parent = materialize[PARENT]()
    var t = c.t
    var delta = List[Q](length=4 * t, fill=Q(0, 0, 0, 0))  # [limb][frame]: goal - end
    for li in range(4):
        var e = limbs[3 * li + 2]
        var held = List[Bool](length=t, fill=False)
        var edges = List[Int]()  # contact runs [start, end), joined across gaps of up to 2 BLEND frames (label flicker)
        for f in range(t):
            if con[f * NB + e] != 0 and (f == 0 or con[(f - 1) * NB + e] == 0):
                if len(edges) > 0 and f - edges[len(edges) - 1] <= 2 * blend:
                    _ = edges.pop()
                else:
                    edges.append(f)
            if con[f * NB + e] != 0 and (f == t - 1 or con[(f + 1) * NB + e] == 0):
                edges.append(f + 1)
        for r in range(0, len(edges), 2):
            var sum = Q(0, 0, 0, 0)
            for g in range(edges[r], edges[r + 1]):
                sum += wp[g * NB + e]
            for g in range(edges[r], edges[r + 1]):
                held[g] = True
                delta[li * t + g] = sum / Float64(edges[r + 1] - edges[r]) - wp[g * NB + e]
        for g in range(t):  # outside runs: each neighbouring run's edge correction, fading over BLEND frames, summed
            if held[g]:
                continue
            for k in range(1, blend + 1):
                if g - k >= 0 and held[g - k]:
                    delta[li * t + g] += delta[li * t + g - k] * (Float64(blend + 1 - k) / Float64(blend + 1))
                    break
            for k in range(1, blend + 1):
                if g + k < t and held[g + k]:
                    delta[li * t + g] += delta[li * t + g + k] * (Float64(blend + 1 - k) / Float64(blend + 1))
                    break
    var wpp = wp.unsafe_ptr()
    var wqp = wq.unsafe_ptr()
    var dp = delta.unsafe_ptr()
    var rp = c.rot.unsafe_ptr()

    def solve(k: Int) {imm t, imm limbs, imm parent, imm wpp, imm wqp, imm dp, imm rp}:  # 64 frames per task, 4 limbs each
        for g in range(64 * k, min(t, 64 * k + 64)):
            for li in range(4):
                var dg = dp.unsafe_offset(li * t + g)[]
                if dg[1] == 0 and dg[2] == 0 and dg[3] == 0:
                    continue
                var u = limbs[3 * li]
                var l = limbs[3 * li + 1]
                var e = limbs[3 * li + 2]
                var p0 = wpp.unsafe_offset(g * NB + u)[]
                var p1 = wpp.unsafe_offset(g * NB + l)[]
                var p2 = wpp.unsafe_offset(g * NB + e)[]
                var goal = p2 + dg
                var a = sqrt(((p1 - p0) * (p1 - p0)).reduce_add())
                var b = sqrt(((p2 - p1) * (p2 - p1)).reduce_add())
                var d = min(max(sqrt(((goal - p0) * (goal - p0)).reduce_add()), abs(a - b) + 1e-9), a + b - 1e-9)
                var ud = unit(goal - p0)
                var a1 = p1 - p0
                var a2 = p2 - p0
                var n = Q(0, a1[2] * a2[3] - a1[3] * a2[2], a1[3] * a2[1] - a1[1] * a2[3], a1[1] * a2[2] - a1[2] * a2[1])  # bend normal
                if (n * n).reduce_add() < 1e-12:  # straight limb: bend toward the upper bone's forward (-y at rest)
                    var fw = rot(wqp.unsafe_offset(g * NB + u)[], Q(0, 0, -1, 0))
                    n = Q(0, a2[2] * fw[3] - a2[3] * fw[2], a2[3] * fw[1] - a2[1] * fw[3], a2[1] * fw[2] - a2[2] * fw[1])
                var side = Q(0, ud[2] * n[3] - ud[3] * n[2], ud[3] * n[1] - ud[1] * n[3], ud[1] * n[2] - ud[2] * n[1])  # keeps the bend's sense
                var ca = (a * a + d * d - b * b) / (2 * a * d)
                var knee = p0 + ud * (a * ca) + unit(side) * (a * sqrt(max(0.0, 1 - ca * ca)))
                var ru = rmin(unit(p1 - p0), unit(knee - p0))
                var rl = rmin(unit(rot(ru, p2 - p1)), unit(p0 + ud * d - knee))
                var wu = qmul(ru, wqp.unsafe_offset(g * NB + u)[])
                var wl = qmul(rl, qmul(ru, wqp.unsafe_offset(g * NB + l)[]))
                rp.unsafe_offset(g * NB + u)[] = qmul(conj(wqp.unsafe_offset(g * NB + parent[u])[]), wu)
                rp.unsafe_offset(g * NB + l)[] = qmul(conj(wu), wl)
                rp.unsafe_offset(g * NB + e)[] = qmul(conj(wl), wqp.unsafe_offset(g * NB + e)[])

    parallelize(solve, (t + 63) // 64)


def write_f32(mut fh: FileHandle, v: List[Float32]) raises:
    fh.write_bytes(Span(unsafe_ptr=v.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(v) * 4))


def write_clip(c: Clip, con: List[Float32], dst: String) raises:
    var order = materialize[ORDER]()
    var feet = materialize[FEET]()
    var t = c.t
    var nq = 7 + 3 * (NB - 1)
    var qref = List[Float32](capacity=t * nq)
    var pos = List[Q](capacity=t * NB)
    var quat = List[Float32](capacity=t * NB * 4)
    var wp = List[Q](length=NB, fill=Q(0, 0, 0, 0))
    var wq = List[Q](length=NB, fill=Q(1, 0, 0, 0))
    var last = List[Float64](length=3 * NB, fill=0.0)
    for f in range(t):
        fk(c, f, wp, wq)
        for b in range(NB):
            pos.append(wp[b])
            for k in range(4):
                quat.append(Float32(wq[b][k]))
        for k in range(3):
            qref.append(Float32(c.root[f][1 + k]))
        for k in range(4):
            qref.append(Float32(c.rot[f * NB][k]))
        for b in range(1, NB):  # of the two solutions (a, b, c) and (a + pi, pi - b, c + pi), each unwrapped by whole
            var best = Q(0, 0, 0, 0)  # turns, the one nearer the previous frame
            var dbest = 1e30
            var e3 = angles(c.rot[f * NB + b], order[3 * b], order[3 * b + 1], order[3 * b + 2])
            for alt in range(2):
                var cand = Q(0, 0, 0, 0)
                var dist = 0.0
                for k in range(3):
                    var ang = e3[k] if alt == 0 else ((pi - e3[k]) if k == 1 else e3[k] + pi)
                    if f > 0:
                        ang += 2 * pi * round((last[3 * b + k] - ang) / (2 * pi))
                        dist += abs(ang - last[3 * b + k])
                    cand[k] = ang
                if dist < dbest:
                    best = cand
                    dbest = dist
            for k in range(3):
                last[3 * b + k] = best[k]
                qref.append(Float32(best[k]))
    var contact = List[Float32](capacity=t * len(feet))
    for f in range(t):
        for b in feet:
            contact.append(con[f * NB + b])
    var fq = open(dst + ".qref", "w")
    var head: List[Int32] = [Int32(t), Int32(nq)]
    fq.write_bytes(Span(unsafe_ptr=head.unsafe_ptr().unsafe_bitcast[UInt8](), length=8))
    write_f32(fq, [Float32(c.hz)])
    write_f32(fq, qref)
    var fm = open(dst + ".mclip", "w")
    var mh: List[Int32] = [Int32(t), Int32(NB), Int32(len(feet))]
    var fidx = List[Int32]()
    for b in feet:
        fidx.append(Int32(b))
    fm.write_bytes(Span(unsafe_ptr=mh.unsafe_ptr().unsafe_bitcast[UInt8](), length=12))
    write_f32(fm, [Float32(c.hz)])
    fm.write_bytes(Span(unsafe_ptr=fidx.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(feet) * 4))
    var p32 = List[Float32](capacity=3 * len(pos))
    for p in pos:
        for k in range(3):
            p32.append(Float32(p[1 + k]))
    write_f32(fm, p32)
    write_f32(fm, quat)
    write_f32(fm, contact)
    print("clip", dst, "frames", t, "hz", c.hz, "bodies", NB, "nq", nq)


def main() raises:
    var args = argv()
    if len(args) == 3 and String(args[1]) == "mjcf":
        write_mjcf(String(args[2]))
    elif len(args) == 4 and String(args[1]) == "clip":
        var src = parse(String(args[2]))
        var fixed = repair(src) if env("REPAIR", 1) > 0 else 0
        var c = retarget(src)
        var wp = List[Q]()
        var wq = List[Q]()
        poses(c, wp, wq)
        var con = contacts(c, wp)
        if env("PLANT", 1) > 0:
            plant(c, con, wp, wq)
        write_clip(c, con, String(args[3]))
        print("repaired", fixed, "spikes")
    else:
        raise Error("usage: bvh mjcf OUT.xml | bvh clip IN.bvh OUT")
