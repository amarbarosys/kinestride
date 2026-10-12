"""Motion metrics between a reference clip and a simulated one, both .mclip (docs/plans/MOTION_DATASET.md).

pos_err      mean |pS - pR| over frames and bodies (m)
rot_err      mean geodesic angle of conj(qR) qS, 2 atan2(|v|, |w|) (rad; q and -q are one rotation)
npss         per body position coordinate: power spectrum over frames without the DC bin, normalised to a
             distribution; earth mover's distance R vs S, weighted by R's share of the total power
foot_skate   mean horizontal (x, y; z is up) foot speed of S over steps with the foot in contact at both ends (m/s)
contact_err  fraction of (frame, foot) where S and R disagree on contact

Float64 throughout: the inputs are float32, the sums are not. Gate: check.sh motion (numpy twin, known answers, mutants).

usage: motion_metrics REF.mclip SIM.mclip [OUT.bin]   (OUT: the five values as float64, in the order above)
build: mojo build -I kinestride kinestride/motion_metrics.mojo
"""
from std.math import atan2, cos, pi, sin, sqrt
from std.sys import argv

from max.algorithm import parallelize


@fieldwise_init
struct Clip(Movable):
    var t: Int
    var nb: Int
    var nf: Int
    var hz: Float64
    var foot: List[Int]
    var pos: List[Float64]  # [t][nb][3]
    var quat: List[Float64]  # [t][nb][4], w x y z
    var contact: List[Float64]  # [t][nf]


def load(path: String) raises -> Clip:
    var raw = open(path, "r").read_bytes()
    if len(raw) < 16:
        raise Error(path + ": shorter than the .mclip header")
    var words = raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var vals = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var t = Int(words.unsafe_offset(0)[])
    var nb = Int(words.unsafe_offset(1)[])
    var nf = Int(words.unsafe_offset(2)[])
    var nwords = 4 + nf + t * nb * 7 + t * nf
    if len(raw) != 4 * nwords:
        raise Error(path + ": " + String(len(raw)) + " bytes, header says " + String(4 * nwords))
    var foot = List[Int](capacity=nf)
    for f in range(nf):
        foot.append(Int(words.unsafe_offset(4 + f)[]))
    var at = 4 + nf
    var pos = List[Float64](capacity=t * nb * 3)
    for i in range(t * nb * 3):
        pos.append(Float64(vals.unsafe_offset(at + i)[]))
    at += t * nb * 3
    var quat = List[Float64](capacity=t * nb * 4)
    for i in range(t * nb * 4):
        quat.append(Float64(vals.unsafe_offset(at + i)[]))
    at += t * nb * 4
    var contact = List[Float64](capacity=t * nf)
    for i in range(t * nf):
        contact.append(Float64(vals.unsafe_offset(at + i)[]))
    return Clip(t, nb, nf, Float64(vals.unsafe_offset(3)[]), foot^, pos^, quat^, contact^)


def pos_err(r: Clip, s: Clip) -> Float64:
    var acc = 0.0
    for i in range(r.t * r.nb):
        var dx = s.pos[3 * i] - r.pos[3 * i]
        var dy = s.pos[3 * i + 1] - r.pos[3 * i + 1]
        var dz = s.pos[3 * i + 2] - r.pos[3 * i + 2]
        acc += sqrt(dx * dx + dy * dy + dz * dz)
    return acc / Float64(max(r.t * r.nb, 1))


def rot_err(r: Clip, s: Clip) -> Float64:
    var acc = 0.0
    for i in range(r.t * r.nb):
        var rw = r.quat[4 * i]
        var rx = r.quat[4 * i + 1]
        var ry = r.quat[4 * i + 2]
        var rz = r.quat[4 * i + 3]
        var sw = s.quat[4 * i]
        var sx = s.quat[4 * i + 1]
        var sy = s.quat[4 * i + 2]
        var sz = s.quat[4 * i + 3]
        # conj(r) s = (rw sw + rv.sv, rw sv - sw rv - rv x sv)
        var w = rw * sw + rx * sx + ry * sy + rz * sz
        var vx = rw * sx - sw * rx - (ry * sz - rz * sy)
        var vy = rw * sy - sw * ry - (rz * sx - rx * sz)
        var vz = rw * sz - sw * rz - (rx * sy - ry * sx)
        acc += 2 * atan2(sqrt(vx * vx + vy * vy + vz * vz), abs(w))
    return acc / Float64(max(r.t * r.nb, 1))


def npss(r: Clip, s: Clip) -> Float64:
    var t = r.t
    var nk = t // 2  # bins 1..t/2: DC (mean position) is not motion
    if nk == 0:
        return 0.0
    var nft = r.nb * 3  # feature f: body f // 3, coordinate f % 3
    var x = List[Float64](length=2 * nft * t, fill=0.0)  # [R|S][feature][frame], contiguous for the DFT
    for i in range(t):
        for f in range(nft):
            x[f * t + i] = r.pos[3 * i * r.nb + f]
            x[(nft + f) * t + i] = s.pos[3 * i * r.nb + f]
    var cs = List[Float64](capacity=2 * t)  # cos, sin of 2 pi m / t, interleaved
    for m in range(t):
        cs.append(cos(2 * pi * Float64(m) / Float64(t)))
        cs.append(sin(2 * pi * Float64(m) / Float64(t)))
    var tr = List[Float64](length=nft, fill=0.0)
    var emd = List[Float64](length=nft, fill=0.0)
    var xp = x.unsafe_ptr()
    var csp = cs.unsafe_ptr()
    var trp = tr.unsafe_ptr()
    var emdp = emd.unsafe_ptr()

    def feature(f: Int) {imm xp, imm csp, imm trp, imm emdp, imm t, imm nk, imm nft}:
        var a = xp.unsafe_offset(f * t)
        var b = xp.unsafe_offset((nft + f) * t)
        var pr = List[Float64](length=nk, fill=0.0)
        var ps = List[Float64](length=nk, fill=0.0)
        var tra = 0.0
        var tsa = 0.0
        for k in range(1, nk + 1):
            var rre = 0.0
            var rim = 0.0
            var sre = 0.0
            var sim = 0.0
            var m = 0
            for i in range(t):
                var cm = csp.unsafe_offset(2 * m)[]
                var sm = csp.unsafe_offset(2 * m + 1)[]
                rre += a.unsafe_offset(i)[] * cm
                rim -= a.unsafe_offset(i)[] * sm
                sre += b.unsafe_offset(i)[] * cm
                sim -= b.unsafe_offset(i)[] * sm
                m += k
                if m >= t:
                    m -= t
            pr[k - 1] = rre * rre + rim * rim
            ps[k - 1] = sre * sre + sim * sim
            tra += pr[k - 1]
            tsa += ps[k - 1]
        var e = 0.0
        var cr = 0.0
        var cq = 0.0
        for k in range(nk):
            cr += pr[k] / tra if tra > 0 else 0.0
            cq += ps[k] / tsa if tsa > 0 else 0.0
            e += abs(cr - cq)
        trp.unsafe_offset(f)[] = tra
        emdp.unsafe_offset(f)[] = e

    parallelize(feature, nft)
    var weighted = 0.0
    var total = 0.0
    for f in range(nft):  # summed in feature order: the result does not depend on the thread count
        weighted += tr[f] * emd[f]
        total += tr[f]
    return weighted / total if total > 0 else 0.0


def foot_skate(s: Clip) -> Float64:
    var acc = 0.0
    var n = 0
    for f in range(s.nf):
        var b = s.foot[f]
        for t in range(s.t - 1):
            if s.contact[t * s.nf + f] > 0.5 and s.contact[(t + 1) * s.nf + f] > 0.5:
                var a = 3 * (t * s.nb + b)
                var e = 3 * ((t + 1) * s.nb + b)
                var dx = s.pos[e] - s.pos[a]
                var dy = s.pos[e + 1] - s.pos[a + 1]
                acc += sqrt(dx * dx + dy * dy) * s.hz
                n += 1
    return acc / Float64(n) if n > 0 else 0.0


def contact_err(r: Clip, s: Clip) -> Float64:
    var n = 0
    for i in range(r.t * r.nf):
        if (r.contact[i] > 0.5) != (s.contact[i] > 0.5):
            n += 1
    return Float64(n) / Float64(r.t * r.nf) if r.nf > 0 else 0.0


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: motion_metrics REF.mclip SIM.mclip [OUT.bin]")
    var r = load(String(args[1]))
    var s = load(String(args[2]))
    if r.t != s.t or r.nb != s.nb or r.nf != s.nf or r.hz != s.hz:
        raise Error("clips differ in shape (frames, bodies, feet or hz)")
    for f in range(r.nf):
        if r.foot[f] != s.foot[f] or r.foot[f] < 0 or r.foot[f] >= r.nb:
            raise Error("foot " + String(f) + ": body index differs between clips or is out of range")
    var out: List[Float64] = [pos_err(r, s), rot_err(r, s), npss(r, s), foot_skate(s), contact_err(r, s)]
    print("pos_err", out[0], "rot_err", out[1], "npss", out[2], "foot_skate", out[3], "contact_err", out[4])
    if len(args) > 3:
        var fh = open(String(args[3]), "w")
        fh.write_bytes(Span(unsafe_ptr=out.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(out) * 8))
