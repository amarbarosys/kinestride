"""Convex-convex narrowphase, ported from MuJoCo 3.9.0 src/engine/engine_collision_gjk.c (Apache-2.0): GJK, EPA and
(box-mesh) multicontact, behind mjc_Convex / mjc_penetration (engine_collision_convex.c:857, :87).

3.9 collides every convex pair that is not a primitive case with this NATIVE GJK + EPA (mjc_ccd), not libccd / MPR
(mjDSBL_NATIVECCD off): ellipsoid-mesh returns one contact, box-mesh up to 4 (5 with a margin) from the multicontact
step inside mjc_ccd. Primitive pairs (sphere, capsule, ellipsoid, cylinder: the flight model) run the same mjc_ccd with its
sphere/capsule core trick (GJK on the point or segment, the radius folded into the margin, EPA only when the cores meet) and,
for capsule/cylinder pairs, the perturbation search (up to 5 contacts); capsule-capsule, sphere-capsule and sphere-cylinder are the analytic primitives
(engine_collision_primitive.c), ported below in the same scalar type.

Scalar type: float32 as the engine, or float64 with -DCCD64=1 (the CPU build that must match MuJoCo's doubles).
Mesh data is the engine's hull tables (model.py feet): hull vertices in the geom frame, per-vertex first edge, edge
lists of hull-local vertex ids ended by -1 (MuJoCo's mesh_graph, so the hill climb visits the same vertices).
Deviation from the C source, all bounded forms of unbounded C: the polytope is a fixed array of 6 * ccd_iterations
faces (as mjc_ccdSize), the EPA horizon recursion is an explicit stack of CCD_STACK frames, horizon edges 24 as the C buffer.
"""
from std.sys import get_defined_int
from std.memory import Pointer
from std.math import sqrt, floor, ceil

comptime CCD_DEBUG = get_defined_int["CCD_DEBUG", 0]() == 1
comptime CCD64 = get_defined_int["CCD64", 0]() == 1
comptime T = DType.float64 if CCD64 else DType.float32
comptime S = Scalar[T]
comptime V = SIMD[T, 4]
comptime NITER = get_defined_int["CCD_ITER", 35]()  # opt.ccd_iterations
comptime MAXF = 6 * NITER
comptime MAXV = 5 + NITER
comptime CCD_STACK = 64
comptime MINVAL = S(1e-15)
comptime MINVAL2 = S(1e-30)
comptime MAXVAL2 = S(1e20)
comptime BIG = S(1.7976931348623157e308) if CCD64 else S(3.4028234e38)

comptime ELLIPSOID = 0
comptime MESH = 1
comptime BOX = 2
comptime PRISM = 3  # one heightfield cell triangle as a prism (mjc_ConvexHField): 6 vertices in the hfield frame
comptime SPHERE = 4  # size0 radius; mjc_ccd shrinks it to its centre (core) with the radius folded into the margin
comptime CAPSULE = 5  # size0 radius, size1 half-length along z; core = the segment
comptime CYLINDER = 6  # size0 radius, size1 half-height along z


@always_inline
def dot(a: V, b: V) -> S:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@always_inline
def cross(a: V, b: V) -> V:
    return V(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0)


@always_inline
def det3(a: V, b: V, c: V) -> S:
    return a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])


@always_inline
def norm(a: V) -> S:
    return sqrt(dot(a, a))


@always_inline
def equal3(a: V, b: V) -> Bool:
    return abs(a[0] - b[0]) < MINVAL and abs(a[1] - b[1]) < MINVAL and abs(a[2] - b[2]) < MINVAL


@always_inline
def normalize3(a: V) -> V:
    """mju_normalize3: (1, 0, 0) below mjMINVAL."""
    var n = norm(a)
    if n < MINVAL:
        return V(1, 0, 0, 0)
    return a * (1 / n)


struct Shape[o: MutOrigin](Copyable):
    """mjCCDObj for the geom kinds ported so far: ellipsoid, hull mesh (hill climb), box."""
    var kind: Int
    var core: Bool  # sphere or capsule shrunk to its centre or segment (mjc_ccd's special handling)
    var pos: V
    var r0: V  # rotation matrix rows (mat[0..2], mat[3..5], mat[6..8])
    var r1: V
    var r2: V
    var size: V
    var margin: S
    var verts: Pointer[Float32, Self.o]  # hull vertices, geom frame
    var first: Pointer[Int32, Self.o]  # per hull vertex: first edge, absolute into edges
    var edges: Pointer[Int32, Self.o]  # hull-local neighbour ids, -1 ends a vertex
    var vertindex: Int
    var meshindex: Int
    var prism: Array[V, 6]

    def __init__(out self, verts: Pointer[Float32, Self.o], first: Pointer[Int32, Self.o], edges: Pointer[Int32, Self.o]):
        self.kind = 0
        self.core = False
        self.pos = V(0)
        self.r0 = V(1, 0, 0, 0)
        self.r1 = V(0, 1, 0, 0)
        self.r2 = V(0, 0, 1, 0)
        self.size = V(0)
        self.margin = 0
        self.verts = verts
        self.first = first
        self.edges = edges
        self.vertindex = -1
        self.meshindex = -1
        self.prism = Array[V, 6](fill=V(0))

    @always_inline
    def local(self, d: V) -> V:
        """mat' d."""
        return self.r0 * d[0] + self.r1 * d[1] + self.r2 * d[2]

    @always_inline
    def world(self, l: V) -> V:
        """mat l + pos."""
        return V(dot(self.r0, l), dot(self.r1, l), dot(self.r2, l), 0) + self.pos

    def center(self) -> V:
        """mjc_center (engine_collision_convex.c:158): the geom position, the mean of the 6 vertices for a prism."""
        if self.kind == PRISM:
            var c = V(0)
            for i in range(6):
                c = c + self.prism[i]
            return c * (S(1) / 6)
        return self.pos

    def support(mut self, d: V) -> V:
        """mjc_ellipsoidSupport (engine_collision_convex.c:256), mjc_hillclimbSupport (:391), mjc_boxSupport (:317)."""
        if self.kind == PRISM:  # mjc_prism_support (:436): best vertex of the bottom (dir z < 0) or top triangle
            var istart = 0 if d[2] < 0 else 3
            var ibest = istart
            var best = dot(self.prism[istart], d)
            for i in range(1, 3):
                var tmp = dot(self.prism[istart + i], d)
                if tmp > best:
                    ibest = istart + i
                    best = tmp
            return self.prism[ibest]
        if self.kind == SPHERE:  # mjc_sphereSupport (:238), mjc_pointSupport for the core; dir is unit
            return self.pos if self.core else self.pos + d * self.size[0]
        var ld = self.local(d)
        if self.kind == CAPSULE:  # mjc_capsuleSupport (:270), mjc_lineSupport for the core
            var end = V(0, 0, self.size[1] if ld[2] >= 0 else -self.size[1], 0)
            return self.world(end if self.core else ld * self.size[0] + end)
        if self.kind == CYLINDER:  # mjc_cylinderSupport (:288)
            var n2 = ld[0] * ld[0] + ld[1] * ld[1]
            var scl = self.size[0] / sqrt(n2) if n2 >= MINVAL2 else S(0)
            return self.world(V(scl * ld[0], scl * ld[1], self.size[1] if ld[2] >= 0 else -self.size[1], 0))
        if self.kind == ELLIPSOID:
            var ls = V(ld[0] * self.size[0], ld[1] * self.size[1], ld[2] * self.size[2], 0)
            var n2 = dot(ls, ls)
            if n2 < MINVAL2:
                return V(self.r0[0] * self.size[0], self.r1[0] * self.size[0], self.r2[0] * self.size[0], 0) + self.pos
            var ni = 1 / sqrt(n2)
            return self.world(V(ls[0] * ni * self.size[0], ls[1] * ni * self.size[1], ls[2] * ni * self.size[2], 0))
        if self.kind == BOX:
            var s = V(self.size[0] if ld[0] >= 0 else -self.size[0], self.size[1] if ld[1] >= 0 else -self.size[1],
                      self.size[2] if ld[2] >= 0 else -self.size[2], 0)
            self.vertindex = (1 if s[0] > 0 else 0) | (2 if s[1] > 0 else 0) | (4 if s[2] > 0 else 0)
            return self.world(s)
        # mesh: hill climb on the hull graph; max starts at -FLT_MAX and is never reset (as the C)
        var mx = S(-3.4028234e38)
        var prev = -1
        var imax = self.meshindex if self.meshindex >= 0 else 0
        while imax != prev:
            prev = imax
            var i = Int(self.first[unsafe_offset=imax])
            while True:
                var sub = Int(self.edges[unsafe_offset=i])
                if sub < 0:
                    break
                var vd = ld[0] * S(self.verts[unsafe_offset=3 * sub]) + ld[1] * S(self.verts[unsafe_offset=3 * sub + 1]) + ld[2] * S(self.verts[unsafe_offset=3 * sub + 2])
                if vd > mx:
                    mx = vd
                    imax = sub
                i += 1
        self.meshindex = imax
        self.vertindex = imax
        return self.world(V(S(self.verts[unsafe_offset=3 * imax]), S(self.verts[unsafe_offset=3 * imax + 1]), S(self.verts[unsafe_offset=3 * imax + 2]), 0))


comptime Sp = Shape[MutAnyOrigin]


struct Vertex(Copyable):
    var vert: V  # v1 - v2, vertex of the Minkowski difference
    var vert1: V
    var vert2: V
    var index1: Int
    var index2: Int

    def __init__(out self):
        self.vert = V(0)
        self.vert1 = V(0)
        self.vert2 = V(0)
        self.index1 = -1
        self.index2 = -1


struct Face(Copyable):
    var a: Int  # polytope vertex ids
    var b: Int
    var c: Int
    var adj0: Int
    var adj1: Int
    var adj2: Int
    var v: V  # witness point: origin projected on the face
    var dist2: S
    var index: Int  # position in the map, -1 not in it, -2 deleted

    def __init__(out self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.adj0 = 0
        self.adj1 = 0
        self.adj2 = 0
        self.v = V(0)
        self.dist2 = 0
        self.index = -1

    @always_inline
    def vid(self, k: Int) -> Int:
        return self.a if k == 0 else (self.b if k == 1 else self.c)

    @always_inline
    def adj(self, k: Int) -> Int:
        return self.adj0 if k == 0 else (self.adj1 if k == 1 else self.adj2)


struct Polytope(Copyable):
    var verts: Array[Vertex, MAXV]
    var faces: Array[Face, MAXF]
    var map: Array[Int32, MAXF]
    var nverts: Int
    var nfaces: Int
    var nmap: Int
    var hedges: Array[Int32, 24]  # horizon: edge of the face, face index
    var hindex: Array[Int32, 24]
    var nedges: Int
    var w: V

    def __init__(out self):
        self.verts = Array[Vertex, MAXV](fill=Vertex())
        self.faces = Array[Face, MAXF](fill=Face())
        self.map = Array[Int32, MAXF](fill=0)
        self.nverts = 0
        self.nfaces = 0
        self.nmap = 0
        self.hedges = Array[Int32, 24](fill=0)
        self.hindex = Array[Int32, 24](fill=0)
        self.nedges = 0
        self.w = V(0)


struct Status(Copyable):
    """mjCCDStatus."""
    var dist: S
    var x1: V
    var x2: V
    var nx: Int
    var max_iterations: Int
    var tolerance: S
    var max_contacts: Int
    var dist_cutoff: S
    var gjk_iterations: Int
    var epa_iterations: Int
    var epa_status: Int
    var simplex: Array[Vertex, 4]
    var nsimplex: Int

    def __init__(out self):
        self.dist = 0
        self.x1 = V(0)
        self.x2 = V(0)
        self.nx = 0
        self.max_iterations = NITER
        self.tolerance = 1e-6
        self.max_contacts = 1
        self.dist_cutoff = 0
        self.gjk_iterations = 0
        self.epa_iterations = 0
        self.epa_status = -1
        self.simplex = Array[Vertex, 4](fill=Vertex())
        self.nsimplex = 0


# ---------------------------------------- GJK (engine_collision_gjk.c:177) --------------------------------


@always_inline
def lincomb(coef: Array[S, 4], n: Int, v1: V, v2: V, v3: V, v4: V) -> V:
    """engine_collision_gjk.c:456."""
    if n == 1:
        return v1 * coef[0]
    if n == 2:
        return v1 * coef[0] + v2 * coef[1]
    if n == 3:
        return v1 * coef[0] + v2 * coef[1] + v3 * coef[2]
    return v1 * coef[0] + v2 * coef[1] + v3 * coef[2] + v4 * coef[3]


@always_inline
def same_sign2(a: S, b: S) -> Int:
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1
    return 0


def project_origin_plane(mut res: V, v1: V, v2: V, v3: V) -> Int:
    """engine_collision_gjk.c:484."""
    var d21 = v2 - v1
    var d31 = v3 - v1
    var d32 = v3 - v2
    var n = cross(d32, d21)
    var nv = dot(n, v2)
    var nn = dot(n, n)
    if nn == 0:
        return 1
    if nv != 0 and nn > MINVAL:
        res = n * (nv / nn)
        return 0
    n = cross(d21, d31)
    nv = dot(n, v1)
    nn = dot(n, n)
    if nn == 0:
        return 1
    if nv != 0 and nn > MINVAL:
        res = n * (nv / nn)
        return 0
    n = cross(d31, d32)
    nv = dot(n, v3)
    nn = dot(n, n)
    res = n * (nv / nn)
    return 0


@always_inline
def project_origin_line(v1: V, v2: V) -> V:
    """engine_collision_gjk.c:521."""
    var diff = v2 - v1
    var scl = -(dot(v2, diff) / dot(diff, diff))
    return v2 + diff * scl


def s1d(mut lam: Array[S, 4], s1: V, s2: V):
    """engine_collision_gjk.c:791: lam[0..1]."""
    var p = project_origin_line(s1, s2)
    var mu = s1[0] - s2[0]
    var mu_max = mu
    var index = 0
    mu = s1[1] - s2[1]
    if abs(mu) >= abs(mu_max):
        mu_max = mu
        index = 1
    mu = s1[2] - s2[2]
    if abs(mu) >= abs(mu_max):
        mu_max = mu
        index = 2
    var c1 = p[index] - s2[index]
    var c2 = s1[index] - p[index]
    var same = same_sign2(mu_max, c1) != 0 and same_sign2(mu_max, c2) != 0
    lam[0] = c1 / mu_max if same else 0
    lam[1] = c2 / mu_max if same else 1


def s2d(mut lam: Array[S, 4], s1: V, s2: V, s3: V):
    """engine_collision_gjk.c:658: lam[0..2]."""
    var po = V(0)
    if project_origin_plane(po, s1, s2, s3) != 0:
        s1d(lam, s1, s2)
        lam[2] = 0
        return
    var m14 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1]
    var m24 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0]
    var m34 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0]
    var mmax: S
    var x: Int
    var y: Int
    var mu1 = abs(m14)
    var mu2 = abs(m24)
    var mu3 = abs(m34)
    if mu1 >= mu2 and mu1 >= mu3:
        mmax = m14
        x = 1
        y = 2
    elif mu2 >= mu3:
        mmax = m24
        x = 0
        y = 2
    else:
        mmax = m34
        x = 0
        y = 1
    var c31 = po[x] * s2[y] + po[y] * s3[x] + s2[x] * s3[y] - po[x] * s3[y] - po[y] * s2[x] - s3[x] * s2[y]
    var c32 = po[x] * s3[y] + po[y] * s1[x] + s3[x] * s1[y] - po[x] * s1[y] - po[y] * s3[x] - s1[x] * s3[y]
    var c33 = po[x] * s1[y] + po[y] * s2[x] + s1[x] * s2[y] - po[x] * s2[y] - po[y] * s1[x] - s2[x] * s1[y]
    var comp1 = same_sign2(mmax, c31)
    var comp2 = same_sign2(mmax, c32)
    var comp3 = same_sign2(mmax, c33)
    if comp1 != 0 and comp2 != 0 and comp3 != 0:
        lam[0] = c31 / mmax
        lam[1] = c32 / mmax
        lam[2] = c33 / mmax
        return
    var dmin = BIG
    var l1 = Array[S, 4](fill=0)
    if comp1 == 0:
        s1d(l1, s2, s3)
        var xx = s2 * l1[0] + s3 * l1[1]
        dmin = dot(xx, xx)
        lam[0] = 0
        lam[1] = l1[0]
        lam[2] = l1[1]
    if comp2 == 0:
        s1d(l1, s1, s3)
        var xx = s1 * l1[0] + s3 * l1[1]
        var d = dot(xx, xx)
        if d < dmin:
            lam[0] = l1[0]
            lam[1] = 0
            lam[2] = l1[1]
            dmin = d
    if comp3 == 0:
        s1d(l1, s1, s2)
        var xx = s1 * l1[0] + s2 * l1[1]
        var d = dot(xx, xx)
        if d < dmin:
            lam[0] = l1[0]
            lam[1] = l1[1]
            lam[2] = 0


def s3d(mut lam: Array[S, 4], s1: V, s2: V, s3: V, s4: V):
    """engine_collision_gjk.c:567: lam[0..3]."""
    var c41 = -det3(s2, s3, s4)
    var c42 = det3(s1, s3, s4)
    var c43 = -det3(s1, s2, s4)
    var c44 = det3(s1, s2, s3)
    var mdet = c41 + c42 + c43 + c44
    var comp1 = same_sign2(mdet, c41)
    var comp2 = same_sign2(mdet, c42)
    var comp3 = same_sign2(mdet, c43)
    var comp4 = same_sign2(mdet, c44)
    if comp1 != 0 and comp2 != 0 and comp3 != 0 and comp4 != 0:
        lam[0] = c41 / mdet
        lam[1] = c42 / mdet
        lam[2] = c43 / mdet
        lam[3] = c44 / mdet
        return
    var dmin = BIG
    var l2 = Array[S, 4](fill=0)
    if comp1 == 0:
        s2d(l2, s2, s3, s4)
        var x = s2 * l2[0] + s3 * l2[1] + s4 * l2[2]
        dmin = dot(x, x)
        lam[0] = 0
        lam[1] = l2[0]
        lam[2] = l2[1]
        lam[3] = l2[2]
    if comp2 == 0:
        s2d(l2, s1, s3, s4)
        var x = s1 * l2[0] + s3 * l2[1] + s4 * l2[2]
        var d = dot(x, x)
        if d < dmin:
            lam[0] = l2[0]
            lam[1] = 0
            lam[2] = l2[1]
            lam[3] = l2[2]
            dmin = d
    if comp3 == 0:
        s2d(l2, s1, s2, s4)
        var x = s1 * l2[0] + s2 * l2[1] + s4 * l2[2]
        var d = dot(x, x)
        if d < dmin:
            lam[0] = l2[0]
            lam[1] = l2[1]
            lam[2] = 0
            lam[3] = l2[2]
            dmin = d
    if comp4 == 0:
        s2d(l2, s1, s2, s3)
        var x = s1 * l2[0] + s2 * l2[1] + s3 * l2[2]
        var d = dot(x, x)
        if d < dmin:
            lam[0] = l2[0]
            lam[1] = l2[1]
            lam[2] = l2[2]


def subdistance(mut lam: Array[S, 4], n: Int, simplex: Array[Vertex, 4]):
    """engine_collision_gjk.c:543 (Montanari et al, ToG 2017)."""
    lam[0] = 0
    lam[1] = 0
    lam[2] = 0
    lam[3] = 0
    if n == 4:
        s3d(lam, simplex[0].vert, simplex[1].vert, simplex[2].vert, simplex[3].vert)
    elif n == 3:
        s2d(lam, simplex[0].vert, simplex[1].vert, simplex[2].vert)
    elif n == 2:
        s1d(lam, simplex[0].vert, simplex[1].vert)
    else:
        lam[0] = 1


def support(mut v: Vertex, mut o1: Sp, mut o2: Sp, dir: V, dir_neg: V):
    """engine_collision_gjk.c:311: S_{A-B}(dir) = S_A(dir) - S_B(-dir), each inflated by margin / 2."""
    v.vert1 = o1.support(dir)
    if o1.margin > 0:
        v.vert1 = v.vert1 + dir * (S(0.5) * o1.margin)
    v.vert2 = o2.support(dir_neg)
    if o2.margin > 0:
        v.vert2 = v.vert2 + dir_neg * (S(0.5) * o2.margin)
    v.vert = v.vert1 - v.vert2
    v.index1 = o1.vertindex
    v.index2 = o2.vertindex


@always_inline
def signed_distance(mut normal: V, v1: Vertex, v2: Vertex, v3: Vertex) -> S:
    """engine_collision_gjk.c:381."""
    var d1 = v3.vert - v1.vert
    var d2 = v2.vert - v1.vert
    normal = cross(d1, d2)
    var n2 = dot(normal, normal)
    if n2 > MINVAL2 and n2 < MAXVAL2:
        normal = normal * (1 / sqrt(n2))
        return dot(normal, v1.vert)
    return BIG


def gjk_intersect(mut st: Status, mut o1: Sp, mut o2: Sp) -> Int:
    """engine_collision_gjk.c:397: 1 in contact, 0 not, -1 inconclusive."""
    var simplex = Array[Vertex, 4](fill=Vertex())
    for i in range(4):
        simplex[i] = st.simplex[i].copy()
    var s = Array[Int, 4](fill=0)
    for q in range(4):
        s[q] = q
    var k = st.gjk_iterations
    var kmax = st.max_iterations
    while k < kmax:
        var n0 = V(0)
        var n1 = V(0)
        var n2 = V(0)
        var n3 = V(0)
        var d0 = signed_distance(n0, simplex[s[2]], simplex[s[1]], simplex[s[3]])
        var d1 = signed_distance(n1, simplex[s[0]], simplex[s[2]], simplex[s[3]])
        var d2 = signed_distance(n2, simplex[s[1]], simplex[s[0]], simplex[s[3]])
        var d3 = signed_distance(n3, simplex[s[0]], simplex[s[1]], simplex[s[2]])
        if d3 == 0 or d2 == 0 or d1 == 0 or d0 == 0:
            st.gjk_iterations = k
            return -1
        var i = 0 if d0 < d1 else 1
        var j = 2 if d2 < d3 else 3
        var di = d0 if i == 0 else d1
        var dj = d2 if j == 2 else d3
        var index = i if di < dj else j
        var dindex = d0 if index == 0 else (d1 if index == 1 else (d2 if index == 2 else d3))
        var nrm = n0 if index == 0 else (n1 if index == 1 else (n2 if index == 2 else n3))
        if dindex > 0:
            st.nsimplex = 4
            for q in range(4):
                st.simplex[q] = simplex[s[q]].copy()
            st.gjk_iterations = k
            return 1
        support(simplex[s[index]], o1, o2, nrm, -nrm)
        if dot(nrm, simplex[s[index]].vert) < 0:
            st.nsimplex = 0
            st.gjk_iterations = k
            return 0
        i = (index + 1) & 3
        j = (index + 2) & 3
        var swap = s[i]
        s[i] = s[j]
        s[j] = swap
        k += 1
    st.gjk_iterations = k
    return -1


def discrete_geoms(o1: Sp, o2: Sp) -> Bool:
    """engine_collision_gjk.c:165: meshes and boxes without margin."""
    if o1.margin != 0 or o2.margin != 0:
        return False
    return (o1.kind == MESH or o1.kind == BOX) and (o2.kind == MESH or o2.kind == BOX)


def gjk(mut st: Status, mut o1: Sp, mut o2: Sp):
    """engine_collision_gjk.c:177 (dist_cutoff 0: no geom distance, early out on a separating hyperplane)."""
    var get_dist = st.dist_cutoff > 0
    var backup_gjk = not get_dist
    var n = 0
    var k = 0
    var kmax = st.max_iterations
    var lam = Array[S, 4](fill=0)
    lam[0] = 1
    var cutoff2 = st.dist_cutoff * st.dist_cutoff
    var tol2 = st.tolerance * st.tolerance
    var discrete = discrete_geoms(o1, o2)
    var epsilon = S(0) if discrete else S(0.5) * tol2
    var min_norm2 = MINVAL2 if discrete else tol2
    var x_norm = S(0)
    var xk = st.x1 - st.x2
    while k < kmax:
        x_norm = dot(xk, xk)
        if x_norm < min_norm2:
            break
        x_norm = sqrt(x_norm)
        var dir_neg = xk * (1 / x_norm)
        var dir = dir_neg * S(-1)
        support(st.simplex[n], o1, o2, dir, dir_neg)
        var sk = st.simplex[n].vert
        var diff = xk - sk
        if dot(xk, diff) < epsilon:
            if k == 0:
                n = 1
            break
        if not get_dist:
            if dot(xk, sk) > 0:
                st.gjk_iterations = k
                st.nsimplex = 0
                st.nx = 0
                st.dist = BIG
                return
        elif st.dist_cutoff < BIG:
            var vs = dot(xk, sk)
            var vv = dot(xk, xk)
            if vs > 0 and (vs * vs / vv) >= cutoff2:
                st.gjk_iterations = k
                st.nsimplex = 0
                st.nx = 0
                st.dist = BIG
                return
        if n == 3 and backup_gjk:
            st.gjk_iterations = k
            var ret = gjk_intersect(st, o1, o2)
            if ret != -1:
                st.nx = 0
                st.dist = S(0) if ret > 0 else BIG
                return
            k = st.gjk_iterations
            backup_gjk = False
        subdistance(lam, n + 1, st.simplex)
        n = 0
        for i in range(4):
            if lam[i] == 0:
                continue
            st.simplex[n] = st.simplex[i].copy()
            lam[n] = lam[i]
            n += 1
        if n < 1:
            st.gjk_iterations = k
            st.nsimplex = 0
            st.nx = 0
            st.dist = BIG
            return
        var xnext = lincomb(lam, n, st.simplex[0].vert, st.simplex[1].vert, st.simplex[2].vert, st.simplex[3].vert)
        if equal3(xnext, xk):
            break
        xk = xnext
        if n == 4:
            x_norm = 0
            break
        k += 1
    st.x1 = lincomb(lam, n, st.simplex[0].vert1, st.simplex[1].vert1, st.simplex[2].vert1, st.simplex[3].vert1)
    st.x2 = lincomb(lam, n, st.simplex[0].vert2, st.simplex[1].vert2, st.simplex[2].vert2, st.simplex[3].vert2)
    st.nx = 1
    st.gjk_iterations = k
    st.nsimplex = n
    st.dist = x_norm


# ---------------------------------------- EPA (engine_collision_gjk.c:823) --------------------------------


def epa_support(mut pt: Polytope, mut o1: Sp, mut o2: Sp, d: V, dnorm: S) -> Int:
    """engine_collision_gjk.c:353."""
    var dir = V(1, 0, 0, 0)
    var dir_neg = V(-1, 0, 0, 0)
    if dnorm > MINVAL:
        dir = d * (1 / dnorm)
        dir_neg = dir * S(-1)
    var n = pt.nverts
    pt.nverts += 1
    support(pt.verts[n], o1, o2, dir, dir_neg)
    return n


@always_inline
def insert_vertex(mut pt: Polytope, v: Vertex) -> Int:
    var n = pt.nverts
    pt.nverts += 1
    pt.verts[n] = v.copy()
    return n


def attach_face(mut pt: Polytope, v1: Int, v2: Int, v3: Int, adj1: Int, adj2: Int, adj3: Int) -> S:
    """engine_collision_gjk.c:1194: squared distance of the face to the origin (0 when it cannot be projected)."""
    var fi = pt.nfaces
    pt.nfaces += 1
    pt.faces[fi].a = v1
    pt.faces[fi].b = v2
    pt.faces[fi].c = v3
    pt.faces[fi].adj0 = adj1
    pt.faces[fi].adj1 = adj2
    pt.faces[fi].adj2 = adj3
    var v = V(0)
    var ret = project_origin_plane(v, pt.verts[v3].vert, pt.verts[v2].vert, pt.verts[v1].vert)
    pt.faces[fi].v = v
    if ret != 0:
        return 0
    pt.faces[fi].dist2 = dot(v, v)
    pt.faces[fi].index = -1
    return pt.faces[fi].dist2


def delete_face(mut pt: Polytope, fi: Int):
    """engine_collision_gjk.c:1178."""
    var idx = pt.faces[fi].index
    if idx >= 0:
        pt.nmap -= 1
        pt.map[idx] = pt.map[pt.nmap]
        pt.faces[Int(pt.map[idx])].index = idx
    pt.faces[fi].index = -2


@always_inline
def add_edge(mut pt: Polytope, index: Int, edge: Int):
    if pt.nedges < 24:
        pt.hedges[pt.nedges] = Int32(edge)
        pt.hindex[pt.nedges] = Int32(index)
    pt.nedges += 1


@always_inline
def get_edge(pt: Polytope, fi: Int, vertex: Int) -> Int:
    if pt.faces[fi].a == vertex:
        return 0
    if pt.faces[fi].b == vertex:
        return 1
    return 2


def horizon_rec(mut pt: Polytope, face0: Int, e0: Int) -> Int:
    """engine_collision_gjk.c:1233 without recursion: the same depth-first order on an explicit stack.
    Returns 1 if face0 is visible from w (deleted), else 0."""
    if dot(pt.faces[face0].v, pt.w) - pt.faces[face0].dist2 <= MINVAL:
        return 0
    var sf = Array[Int, CCD_STACK](fill=0)  # face
    var se = Array[Int, CCD_STACK](fill=0)  # edge it was entered through
    var sk = Array[Int, CCD_STACK](fill=0)  # next neighbour (1, 2, then done)
    var sv0 = Array[Int, CCD_STACK](fill=0)  # its vertices at entry
    var sv1 = Array[Int, CCD_STACK](fill=0)
    var sv2 = Array[Int, CCD_STACK](fill=0)
    var top = 0
    sf[0] = face0
    se[0] = e0
    sk[0] = 1
    sv0[0] = pt.faces[face0].a
    sv1[0] = pt.faces[face0].b
    sv2[0] = pt.faces[face0].c
    delete_face(pt, face0)
    while top >= 0:
        if sk[top] > 2:
            top -= 1
            continue
        var f = sf[top]
        var i = (se[top] + sk[top]) % 3
        sk[top] += 1
        var adj = pt.faces[f].adj(i)
        if pt.faces[adj].index > -2:
            var nv = sv1[top] if i == 0 else (sv2[top] if i == 1 else sv0[top])  # verts[(i + 1) % 3]
            var adj_edge = get_edge(pt, adj, nv)
            if dot(pt.faces[adj].v, pt.w) - pt.faces[adj].dist2 > MINVAL:
                if top + 1 >= CCD_STACK:
                    return -1
                top += 1
                sf[top] = adj
                se[top] = adj_edge
                sk[top] = 1
                sv0[top] = pt.faces[adj].a
                sv1[top] = pt.faces[adj].b
                sv2[top] = pt.faces[adj].c
                delete_face(pt, adj)
            else:
                add_edge(pt, adj, adj_edge)
    return 1


def horizon(mut pt: Polytope, face: Int):
    """engine_collision_gjk.c:1257."""
    delete_face(pt, face)
    var v0 = pt.faces[face].a
    var v1 = pt.faces[face].b
    var v2 = pt.faces[face].c
    var adj = pt.faces[face].adj0
    var adj_edge = get_edge(pt, adj, v1)
    if horizon_rec(pt, adj, adj_edge) == 0:
        add_edge(pt, pt.faces[face].adj0, adj_edge)
    adj = pt.faces[face].adj1
    adj_edge = get_edge(pt, adj, v2)
    if pt.faces[adj].index > -2 and horizon_rec(pt, adj, adj_edge) == 0:
        add_edge(pt, pt.faces[face].adj1, adj_edge)
    adj = pt.faces[face].adj2
    adj_edge = get_edge(pt, adj, v0)
    if pt.faces[adj].index > -2 and horizon_rec(pt, adj, adj_edge) == 0:
        add_edge(pt, pt.faces[face].adj2, adj_edge)


def tri_affine_coord(mut lam: Array[S, 4], v1: V, v2: V, v3: V, p: V):
    """engine_collision_gjk.c:990."""
    var m14 = v2[1] * v3[2] - v2[2] * v3[1] - v1[1] * v3[2] + v1[2] * v3[1] + v1[1] * v2[2] - v1[2] * v2[1]
    var m24 = v2[0] * v3[2] - v2[2] * v3[0] - v1[0] * v3[2] + v1[2] * v3[0] + v1[0] * v2[2] - v1[2] * v2[0]
    var m34 = v2[0] * v3[1] - v2[1] * v3[0] - v1[0] * v3[1] + v1[1] * v3[0] + v1[0] * v2[1] - v1[1] * v2[0]
    var mmax: S
    var x: Int
    var y: Int
    var mu1 = abs(m14)
    var mu2 = abs(m24)
    var mu3 = abs(m34)
    if mu1 >= mu2 and mu1 >= mu3:
        mmax = m14
        x = 1
        y = 2
    elif mu2 >= mu3:
        mmax = m24
        x = 0
        y = 2
    else:
        mmax = m34
        x = 0
        y = 1
    var c31 = p[x] * v2[y] + p[y] * v3[x] + v2[x] * v3[y] - p[x] * v3[y] - p[y] * v2[x] - v3[x] * v2[y]
    var c32 = p[x] * v3[y] + p[y] * v1[x] + v3[x] * v1[y] - p[x] * v1[y] - p[y] * v3[x] - v1[x] * v3[y]
    var c33 = p[x] * v1[y] + p[y] * v2[x] + v1[x] * v2[y] - p[x] * v2[y] - p[y] * v1[x] - v2[x] * v1[y]
    lam[0] = c31 / mmax
    lam[1] = c32 / mmax
    lam[2] = c33 / mmax


def tri_point_intersect(v1: V, v2: V, v3: V, p: V) -> Bool:
    """engine_collision_gjk.c:1035."""
    var lam = Array[S, 4](fill=0)
    tri_affine_coord(lam, v1, v2, v3, p)
    if lam[0] < 0 or lam[1] < 0 or lam[2] < 0:
        return False
    var pr = v1 * lam[0] + v2 * lam[1] + v3 * lam[2]
    return norm(pr - p) < MINVAL


def epa_witness(pt: Polytope, fi: Int, mut x1: V, mut x2: V) -> S:
    """engine_collision_gjk.c:1285: witness points from the face's affine coordinates; signed distance."""
    var lam = Array[S, 4](fill=0)
    var a = pt.faces[fi].a
    var b = pt.faces[fi].b
    var c = pt.faces[fi].c
    tri_affine_coord(lam, pt.verts[a].vert, pt.verts[b].vert, pt.verts[c].vert, pt.faces[fi].v)
    x1 = lincomb(lam, 3, pt.verts[a].vert1, pt.verts[b].vert1, pt.verts[c].vert1, V(0))
    x2 = lincomb(lam, 3, pt.verts[a].vert2, pt.verts[b].vert2, pt.verts[c].vert2, V(0))
    return -sqrt(pt.faces[fi].dist2)


def epa(mut st: Status, mut pt: Polytope, mut o1: Sp, mut o2: Sp) -> Int:
    """engine_collision_gjk.c:1306: index of the face of the expanded polytope that best approximates the penetration
    depth (-1: none); witness points in st.x1, st.x2."""
    var upper = BIG
    var upper2 = BIG
    var lower2 = BIG
    var face = -1
    var pface = -1
    var tolerance = st.tolerance
    var discrete = discrete_geoms(o1, o2)
    comptime if CCD64:
        if discrete:
            tolerance = MINVAL
    var repeat_stops = not CCD64  # float32: the 1e-6 tolerance is below the resolution of mm-sized vertices; a repeated vertex ends EPA as for discrete geoms
    var kmax = st.max_iterations if st.max_iterations < 1000 else 1000
    var k = 0
    while k < kmax:
        pface = face
        lower2 = BIG
        for i in range(pt.nmap):
            var fm = Int(pt.map[i])
            if pt.faces[fm].dist2 < lower2:
                face = fm
                lower2 = pt.faces[fm].dist2
        if lower2 > upper2 or face < 0:
            comptime if CCD_DEBUG:
                print('epa: lower2 > upper2', lower2, upper2, 'k', k, 'pface', pface)
            face = pface
            break
        if lower2 <= 0:
            break
        var lower = sqrt(lower2)
        var wi = epa_support(pt, o1, o2, pt.faces[face].v, lower)
        var upper_k = dot(pt.faces[face].v, pt.verts[wi].vert) / lower
        if upper_k < upper:
            upper = upper_k
            upper2 = upper * upper
        if upper - lower < tolerance:
            break
        if discrete or (repeat_stops and (o1.kind == MESH or o1.kind == BOX) and (o2.kind == MESH or o2.kind == BOX)):
            var i = 0
            var nverts = pt.nverts - 1
            while i < nverts:
                if pt.verts[wi].index1 == pt.verts[i].index1 and pt.verts[wi].index2 == pt.verts[i].index2:
                    break
                i += 1
            if i != nverts:
                break
        pt.w = pt.verts[wi].vert
        horizon(pt, face)
        if pt.nedges < 3:
            comptime if CCD_DEBUG:
                print('epa: horizon nedges', pt.nedges, 'k', k)
            face = -1
            break
        var nfaces = pt.nfaces
        var nedges = pt.nedges
        if nedges > 24 or nedges > MAXF - pt.nfaces:
            comptime if CCD_DEBUG:
                print('epa: out of faces/edges', nedges, pt.nfaces, 'k', k)
            break
        var hi = Int(pt.hindex[0])
        var he = Int(pt.hedges[0])
        var hv1 = pt.faces[hi].vid(he)
        var hv2 = pt.faces[hi].vid((he + 1) % 3)
        if he == 0:
            pt.faces[hi].adj0 = nfaces
        elif he == 1:
            pt.faces[hi].adj1 = nfaces
        else:
            pt.faces[hi].adj2 = nfaces
        var dist2 = attach_face(pt, wi, hv2, hv1, nfaces + nedges - 1, hi, nfaces + 1)
        if dist2 == 0:
            comptime if CCD_DEBUG:
                print('epa: attach dist2 0 (first) k', k)
            face = -1
            break
        if dist2 >= lower2 and dist2 <= upper2:
            var im = pt.nmap
            pt.nmap += 1
            pt.map[im] = Int32(pt.nfaces - 1)
            pt.faces[pt.nfaces - 1].index = im
        for i in range(1, nedges):
            var cur = nfaces + i
            var nxt = nfaces + (i + 1) % nedges
            hi = Int(pt.hindex[i])
            he = Int(pt.hedges[i])
            hv1 = pt.faces[hi].vid(he)
            hv2 = pt.faces[hi].vid((he + 1) % 3)
            if he == 0:
                pt.faces[hi].adj0 = cur
            elif he == 1:
                pt.faces[hi].adj1 = cur
            else:
                pt.faces[hi].adj2 = cur
            dist2 = attach_face(pt, wi, hv2, hv1, cur - 1, hi, nxt)
            if dist2 == 0:
                comptime if CCD_DEBUG:
                    print('epa: attach dist2 0 edge', i, 'k', k)
                face = -1
                break
            if dist2 >= lower2 and dist2 <= upper2:
                var im = pt.nmap
                pt.nmap += 1
                pt.map[im] = Int32(pt.nfaces - 1)
                pt.faces[pt.nfaces - 1].index = im
        pt.nedges = 0
        if pt.nmap == 0 or face < 0:
            break
        k += 1
    st.epa_iterations = k
    if face >= 0:
        var x1 = V(0)
        var x2 = V(0)
        st.dist = epa_witness(pt, face, x1, x2)
        st.x1 = x1
        st.x2 = x2
        st.nx = 1
    else:
        st.nx = 0
        st.dist = 0
    return face


def replace_simplex3(mut pt: Polytope, mut st: Status, v1: Int, v2: Int, v3: Int):
    """engine_collision_gjk.c:826."""
    st.nsimplex = 3
    st.simplex[0] = pt.verts[v1].copy()
    st.simplex[1] = pt.verts[v2].copy()
    st.simplex[2] = pt.verts[v3].copy()
    pt.nfaces = 0
    pt.nverts = 0
    pt.nmap = 0


def same_side(p0: V, p1: V, p2: V, p3: V) -> Bool:
    """engine_collision_gjk.c:841: the origin and p3 on the same side of the plane p0 p1 p2."""
    var n = cross(p1 - p0, p2 - p0)
    var dot1 = dot(n, p3 - p0)
    var dot2 = dot(n, p0 * S(-1))
    if dot1 > 0 and dot2 > 0:
        return True
    if dot1 < 0 and dot2 < 0:
        return True
    return False


def test_tetra(p0: V, p1: V, p2: V, p3: V) -> Bool:
    return same_side(p0, p1, p2, p3) and same_side(p1, p2, p3, p0) and same_side(p2, p3, p0, p1) and same_side(p3, p0, p1, p2)


def rotmat(axis: V) -> Array[S, 9]:
    """engine_collision_gjk.c:870: rotation of 120 degrees around axis."""
    var n = norm(axis)
    var u1 = axis[0] / n
    var u2 = axis[1] / n
    var u3 = axis[2] / n
    var sn = S(0.86602540378)
    var cs = S(-0.5)
    var R = Array[S, 9](fill=0)
    R[0] = cs + u1 * u1 * (1 - cs)
    R[1] = u1 * u2 * (1 - cs) - u3 * sn
    R[2] = u1 * u3 * (1 - cs) + u2 * sn
    R[3] = u2 * u1 * (1 - cs) + u3 * sn
    R[4] = cs + u2 * u2 * (1 - cs)
    R[5] = u2 * u3 * (1 - cs) - u1 * sn
    R[6] = u1 * u3 * (1 - cs) - u2 * sn
    R[7] = u2 * u3 * (1 - cs) + u1 * sn
    R[8] = cs + u3 * u3 * (1 - cs)
    return R^


def ray_triangle(v1: V, v2: V, v3: V, v4: V, v5: V) -> Int:
    """engine_collision_gjk.c:888."""
    var d12 = v2 - v1
    var d13 = v3 - v1
    var d14 = v4 - v1
    var d15 = v5 - v1
    var vol1 = det3(d13, d14, d12)
    var vol2 = det3(d14, d15, d12)
    var vol3 = det3(d15, d13, d12)
    if vol1 >= 0 and vol2 >= 0 and vol3 >= 0:
        return 1
    if vol1 <= 0 and vol2 <= 0 and vol3 <= 0:
        return -1
    return 0


def polytope3(mut pt: Polytope, mut st: Status, mut o1: Sp, mut o2: Sp) -> Int:
    """engine_collision_gjk.c:1052: hexahedron around a 2-simplex (0 on success, else mjEPA_P3_*)."""
    var v1 = st.simplex[0].vert
    var v2 = st.simplex[1].vert
    var v3 = st.simplex[2].vert
    var n = cross(v2 - v1, v3 - v1)
    var n_norm = norm(n)
    if n_norm < MINVAL:
        return 4
    var n_neg = n * S(-1)
    var v1i = insert_vertex(pt, st.simplex[0])
    var v2i = insert_vertex(pt, st.simplex[1])
    var v3i = insert_vertex(pt, st.simplex[2])
    var v5i = epa_support(pt, o1, o2, n_neg, n_norm)
    var v4i = epa_support(pt, o1, o2, n, n_norm)
    var v4 = pt.verts[v4i].vert
    var v5 = pt.verts[v5i].vert
    if tri_point_intersect(v1, v2, v3, v4):
        return 5
    if tri_point_intersect(v1, v2, v3, v5):
        return 6
    if st.dist > 10 * MINVAL and not test_tetra(v1, v2, v3, v4) and not test_tetra(v1, v2, v3, v5):
        return 7
    if attach_face(pt, v4i, v1i, v2i, 1, 3, 2) < MINVAL2:
        return 8
    if attach_face(pt, v4i, v3i, v1i, 2, 4, 0) < MINVAL2:
        return 8
    if attach_face(pt, v4i, v2i, v3i, 0, 5, 1) < MINVAL2:
        return 8
    if attach_face(pt, v5i, v2i, v1i, 5, 0, 4) < MINVAL2:
        return 8
    if attach_face(pt, v5i, v1i, v3i, 3, 1, 5) < MINVAL2:
        return 8
    if attach_face(pt, v5i, v3i, v2i, 4, 2, 3) < MINVAL2:
        return 8
    for i in range(6):
        pt.map[i] = Int32(i)
        pt.faces[i].index = i
    pt.nmap = 6
    return 0


def polytope2(mut pt: Polytope, mut st: Status, mut o1: Sp, mut o2: Sp) -> Int:
    """engine_collision_gjk.c:907: hexahedron around a 1-simplex."""
    var v1 = st.simplex[0].vert
    var v2 = st.simplex[1].vert
    var diff = v2 - v1
    var value = BIG
    var index = 0
    for i in range(3):
        if abs(diff[i]) < value:
            value = abs(diff[i])
            index = i
    var e = V(0)
    e[index] = 1
    var d1 = cross(e, diff)
    var R = rotmat(diff)
    var d2 = V(R[0] * d1[0] + R[1] * d1[1] + R[2] * d1[2], R[3] * d1[0] + R[4] * d1[1] + R[5] * d1[2], R[6] * d1[0] + R[7] * d1[1] + R[8] * d1[2], 0)
    var d3 = V(R[0] * d2[0] + R[1] * d2[1] + R[2] * d2[2], R[3] * d2[0] + R[4] * d2[1] + R[5] * d2[2], R[6] * d2[0] + R[7] * d2[1] + R[8] * d2[2], 0)
    var v1i = insert_vertex(pt, st.simplex[0])
    var v2i = insert_vertex(pt, st.simplex[1])
    var v3i = epa_support(pt, o1, o2, d1, norm(d1))
    var v4i = epa_support(pt, o1, o2, d2, norm(d2))
    var v5i = epa_support(pt, o1, o2, d3, norm(d3))
    var v3 = pt.verts[v3i].vert
    var v4 = pt.verts[v4i].vert
    var v5 = pt.verts[v5i].vert
    if attach_face(pt, v1i, v3i, v4i, 1, 3, 2) < MINVAL2:
        replace_simplex3(pt, st, v1i, v3i, v4i)
        return polytope3(pt, st, o1, o2)
    if attach_face(pt, v1i, v5i, v3i, 2, 4, 0) < MINVAL2:
        replace_simplex3(pt, st, v1i, v5i, v3i)
        return polytope3(pt, st, o1, o2)
    if attach_face(pt, v1i, v4i, v5i, 0, 5, 1) < MINVAL2:
        replace_simplex3(pt, st, v1i, v4i, v5i)
        return polytope3(pt, st, o1, o2)
    if attach_face(pt, v2i, v4i, v3i, 5, 0, 4) < MINVAL2:
        replace_simplex3(pt, st, v2i, v4i, v3i)
        return polytope3(pt, st, o1, o2)
    if attach_face(pt, v2i, v3i, v5i, 3, 1, 5) < MINVAL2:
        replace_simplex3(pt, st, v2i, v3i, v5i)
        return polytope3(pt, st, o1, o2)
    if attach_face(pt, v2i, v5i, v4i, 4, 2, 3) < MINVAL2:
        replace_simplex3(pt, st, v2i, v5i, v4i)
        return polytope3(pt, st, o1, o2)
    if ray_triangle(v1, v2, v3, v4, v5) == 0:
        return 2
    for i in range(6):
        pt.map[i] = Int32(i)
        pt.faces[i].index = i
    pt.nmap = 6
    return 0


def polytope4(mut pt: Polytope, mut st: Status, mut o1: Sp, mut o2: Sp) -> Int:
    """engine_collision_gjk.c:1131: the 3-simplex as the polytope."""
    var v1 = insert_vertex(pt, st.simplex[0])
    var v2 = insert_vertex(pt, st.simplex[1])
    var v3 = insert_vertex(pt, st.simplex[2])
    var v4 = insert_vertex(pt, st.simplex[3])
    if attach_face(pt, v1, v2, v3, 1, 3, 2) < MINVAL2:
        replace_simplex3(pt, st, v1, v2, v3)
        return polytope3(pt, st, o1, o2)
    if attach_face(pt, v1, v4, v2, 2, 3, 0) < MINVAL2:
        replace_simplex3(pt, st, v1, v4, v2)
        return polytope3(pt, st, o1, o2)
    if attach_face(pt, v1, v3, v4, 0, 3, 1) < MINVAL2:
        replace_simplex3(pt, st, v1, v3, v4)
        return polytope3(pt, st, o1, o2)
    if attach_face(pt, v4, v3, v2, 2, 0, 1) < MINVAL2:
        replace_simplex3(pt, st, v4, v3, v2)
        return polytope3(pt, st, o1, o2)
    if not test_tetra(pt.verts[v1].vert, pt.verts[v2].vert, pt.verts[v3].vert, pt.verts[v4].vert):
        return 9
    for i in range(4):
        pt.map[i] = Int32(i)
        pt.faces[i].index = i
    pt.nmap = 4
    return 0


def ccd(mut st: Status, mut o1: Sp, mut o2: Sp) -> S:
    """mjc_ccd (engine_collision_gjk.c:2220) for geoms that are neither sphere nor capsule, penetration contacts only
    (dist_cutoff 0, contacts wanted). Returns the signed distance (< 0: penetrating); witness points in st.x1, st.x2."""
    st.x1 = o1.center()
    st.x2 = o2.center()
    st.gjk_iterations = 0
    st.epa_iterations = 0
    st.epa_status = -1
    gjk(st, o1, o2)
    if st.dist <= st.tolerance and st.nsimplex > 1:
        st.dist = 0
        var pt = Polytope()
        var ret = 0
        if st.nsimplex == 2:
            ret = polytope2(pt, st, o1, o2)
        elif st.nsimplex == 3:
            ret = polytope3(pt, st, o1, o2)
        else:
            ret = polytope4(pt, st, o1, o2)
        st.epa_status = ret
        if ret == 0:
            _ = epa(st, pt, o1, o2)
    return st.dist


def contact_normal(st: Status) -> V:
    """mjc_penetration (engine_collision_convex.c:87): contact normal = normalized x1 - x2 of the witness points."""
    return normalize3(st.x1 - st.x2)


struct Con(Copyable):
    """One contact: n = 0 or 1, dist, pos, normal."""
    var n: Int
    var dist: S
    var pos: V
    var normal: V

    def __init__(out self):
        self.n = 0
        self.dist = 0
        self.pos = V(0)
        self.normal = V(0)


def penetration(mut o1: Sp, mut o2: Sp, margin: S, tol: S, max_contacts: Int) -> Con:
    """mjc_penetration (engine_collision_convex.c:87), native CCD, one contact: both geoms inflated by margin / 2 (their
    margin field), mjc_ccd, a contact where the inflated geoms penetrate (dist < 0): dist margin + depth, pos the witness
    midpoint, normal x1 - x2."""
    o1.margin = margin
    o2.margin = margin
    var st = Status()
    st.tolerance = tol
    st.max_contacts = max_contacts
    var d: S = 0
    var round1 = o1.kind == SPHERE or o1.kind == CAPSULE
    var round2 = o2.kind == SPHERE or o2.kind == CAPSULE
    var cored = False
    if round1 or round2:  # mjc_ccd (engine_collision_gjk.c:2233): GJK on the cores (point, segment) with the radius and half the
        # margin as a distance cutoff; apart: the witness points inflated back (inflate, :2192); cores penetrating: full shapes
        var fm1 = o1.size[0] + S(0.5) * margin if round1 else S(0)
        var fm2 = o2.size[0] + S(0.5) * margin if round2 else S(0)
        o1.core = round1
        o2.core = round2
        if round1:
            o1.margin = 0
        if round2:
            o2.margin = 0
        st.dist_cutoff = fm1 + fm2
        st.x1 = o1.center()
        st.x2 = o2.center()
        st.gjk_iterations = 0
        st.epa_iterations = 0
        st.epa_status = -1
        gjk(st, o1, o2)
        st.dist_cutoff = 0
        o1.core = False
        o2.core = False
        o1.margin = margin
        o2.margin = margin
        if st.dist > st.tolerance:
            var nrm = normalize3(st.x2 - st.x1)
            st.x1 = st.x1 + nrm * fm1
            st.x2 = st.x2 - nrm * fm2
            d = st.dist - (fm1 + fm2)
            if d > 0:
                d = BIG
            cored = True
    if not cored:
        d = ccd(st, o1, o2)
    comptime if CCD_DEBUG:
        print('ccd: gjk_it', st.gjk_iterations, 'nsimplex', st.nsimplex, 'dist', d, 'epa_status', st.epa_status, 'epa_it', st.epa_iterations)
    var c = Con()
    if d < 0:
        c.n = 1
        c.dist = margin + d
        c.pos = (st.x1 + st.x2) * S(0.5)
        c.normal = contact_normal(st)
    return c^


struct ConSet(Copyable):
    """The contacts of a pair, at most 5 (the first plus four perturbed)."""
    var n: Int
    var dist: Array[S, 5]
    var pos: Array[V, 5]
    var normal: Array[V, 5]

    def __init__(out self):
        self.n = 0
        self.dist = Array[S, 5](fill=0)
        self.pos = Array[V, 5](fill=V(0))
        self.normal = Array[V, 5](fill=V(0))


@always_inline
def rotate_frame(origin: V, rot: Array[S, 9], mut o: Sp):
    """mju_rotateFrame (engine_collision_convex.c:818): mat = rot mat, pos -= rot (origin - pos) - (origin - pos)."""
    var a = o.r0
    var b = o.r1
    var c = o.r2
    o.r0 = a * rot[0] + b * rot[1] + c * rot[2]
    o.r1 = a * rot[3] + b * rot[4] + c * rot[5]
    o.r2 = a * rot[6] + b * rot[7] + c * rot[8]
    var rel = origin - o.pos
    var vec = V(rot[0] * rel[0] + rot[1] * rel[1] + rot[2] * rel[2], rot[3] * rel[0] + rot[4] * rel[1] + rot[5] * rel[2],
                rot[6] * rel[0] + rot[7] * rel[1] + rot[8] * rel[2], 0) - rel
    o.pos = o.pos - vec


def tilt(axis: V, sgn: S) -> Array[S, 9]:
    """mji_axisAngle2Quat + mju_quat2Mat for an angle of sgn * 1e-3 rad."""
    var w = S(0.99999987500000260416)  # cos(5e-4)
    var sn = sgn * S(4.9999997916666693e-4)  # sin(+-5e-4)
    var x = axis[0] * sn
    var y = axis[1] * sn
    var z = axis[2] * sn
    var m = Array[S, 9](fill=0)
    m[0] = w * w + x * x - y * y - z * z
    m[1] = 2 * (x * y - w * z)
    m[2] = 2 * (x * z + w * y)
    m[3] = 2 * (x * y + w * z)
    m[4] = w * w - x * x + y * y - z * z
    m[5] = 2 * (y * z - w * x)
    m[6] = 2 * (x * z - w * y)
    m[7] = 2 * (y * z + w * x)
    m[8] = w * w - x * x - y * y + z * z
    return m^


@always_inline
def clip1(x: S) -> S:
    return S(1) if x > 1 else (S(-1) if x < -1 else x)


@always_inline
def raw_sphere_sphere(mut cs: ConSet, margin: S, pos1: V, ax1: V, r1: S, pos2: V, ax2: V, r2: S):
    """mjraw_SphereSphere (engine_collision_primitive.c:262): one contact on the centre line appended to cs, none beyond the margin."""
    var dif = pos1 - pos2
    var cd2 = dot(dif, dif)
    var mind = margin + r1 + r2
    if cd2 > mind * mind or cs.n >= 5:
        return
    var dist = sqrt(cd2) - r1 - r2
    var normal = pos2 - pos1
    var len = norm(normal)
    if len < MINVAL:
        normal = cross(ax1, ax2)
        normal = normalize3(normal)
    else:
        normal = normal / len
    cs.dist[cs.n] = dist
    cs.normal[cs.n] = normal
    cs.pos[cs.n] = (pos1 + normal * (r1 + dist / 2))
    cs.n += 1


def primitive_pair(kind: Int, pos1: V, ax1: V, sz1: V, pos2: V, ax2: V, sz2: V, margin: S) -> ConSet:
    """The analytic primitive pairs: mjraw_CapsuleCapsule (engine_collision_primitive.c:425, kind 5: closest points of the two
    segments, two contacts when parallel), mjraw_SphereCapsule (:313, kind 6), mjc_SphereCylinder (:345, kind 7: side, cap or
    corner). Contacts as the convex set (dist, pos, normal from geom1 to geom2)."""
    var cs = ConSet()
    if kind == 5:
        var a1 = ax1 * sz1[1]; var a2 = ax2 * sz2[1]
        var dif = pos1 - pos2
        var ma = dot(a1, a1); var mb = -dot(a1, a2); var mc = dot(a2, a2)
        var u = -dot(a1, dif); var v = dot(a2, dif)
        var det = ma * mc - mb * mb
        if abs(det) >= MINVAL:
            var x1 = (mc * u - mb * v) / det; var x2 = (ma * v - mb * u) / det
            if x1 > 1:
                x1 = 1; x2 = (v - mb) / mc
            elif x1 < -1:
                x1 = -1; x2 = (v + mb) / mc
            if x2 > 1:
                x2 = 1; x1 = clip1((u - mb) / ma)
            elif x2 < -1:
                x2 = -1; x1 = clip1((u + mb) / ma)
            raw_sphere_sphere(cs, margin, pos1 + a1 * x1, ax1, sz1[0], pos2 + a2 * x2, ax2, sz2[0])
        else:  # parallel: both ends of each segment against the other
            var x2 = clip1((v - mb) / mc)
            raw_sphere_sphere(cs, margin, pos1 + a1, ax1, sz1[0], pos2 + a2 * x2, ax2, sz2[0])
            x2 = clip1((v + mb) / mc)
            raw_sphere_sphere(cs, margin, pos1 - a1, ax1, sz1[0], pos2 + a2 * x2, ax2, sz2[0])
            if cs.n < 2:
                var x1 = clip1((u - mb) / ma)
                raw_sphere_sphere(cs, margin, pos1 + a1 * x1, ax1, sz1[0], pos2 + a2, ax2, sz2[0])
            if cs.n < 2:
                var x1 = clip1((u + mb) / ma)
                raw_sphere_sphere(cs, margin, pos1 + a1 * x1, ax1, sz1[0], pos2 - a2, ax2, sz2[0])
    elif kind == 6:
        var vec = pos1 - pos2
        var x = min(sz2[1], max(-sz2[1], dot(ax2, vec)))
        raw_sphere_sphere(cs, margin, pos1, ax1, sz1[0], pos2 + ax2 * x, ax2, sz2[0])
    else:  # kind 7
        var radius = sz2[0]; var height = sz2[1]
        var vec = pos1 - pos2
        var x = dot(ax2, vec)
        var a_proj = ax2 * x
        var p_proj = vec - a_proj
        var pp2 = dot(p_proj, p_proj)
        var side = abs(x) < height
        var cap = pp2 < radius * radius
        if side and cap:  # sphere centre inside the cylinder: the nearer exit
            if height - abs(x) < radius - sqrt(pp2):
                side = False
            else:
                cap = False
        if side:
            raw_sphere_sphere(cs, margin, pos1, ax1, sz1[0], a_proj + pos2, ax2, sz2[0])
        elif cap:  # mjraw_PlaneSphere on the cap, normal flipped (plane < sphere < cylinder in the type order)
            var n = ax2 if x > 0 else -ax2
            var pos_cap = pos2 + ax2 * (height if x > 0 else -height)
            var cdist = dot(pos1 - pos_cap, n)
            if cdist <= margin + sz1[0]:
                var dist = cdist - sz1[0]
                cs.dist[0] = dist
                cs.normal[0] = (-n)
                cs.pos[0] = (pos1 + n * (-dist / 2 - sz1[0]))
                cs.n = 1
        else:  # corner: a point sphere on the rim
            var rim = p_proj * (radius / sqrt(pp2)) + ax2 * (height if x > 0 else -height) + pos2
            raw_sphere_sphere(cs, margin, pos1, ax1, sz1[0], rim, ax2, S(0))
    return cs^

def collide(mut o1: Sp, mut o2: Sp, margin: S, tol: S, rbound: S, multi: Bool = True) -> ConSet:
    """mjc_Convex (engine_collision_convex.c:857) for a pair with margin > 0, where maxContacts is 1: the first contact,
    then (not for sphere/ellipsoid) the perturbation search of :874-927: both geoms rotated by -+1e-3 rad about the two
    tangent axes of the first contact, around its point, a new contact kept when it is farther than 1e-3 rbound (rbound =
    min of the two geoms' bounding radii) from the earlier ones, its depth set to the first's. The solve runs
    relative to geom 2's centre (translation invariant; keeps float32 coordinates mm-sized so the 1e-6 tolerance is
    resolvable); positions are returned in the caller's frame. Mesh hill-climb state persists across the calls."""
    var org = o2.pos
    o1.pos = o1.pos - org
    o2.pos = V(0)
    var cs = ConSet()
    var c0 = penetration(o1, o2, margin, tol, 1)
    if c0.n > 0:
        cs.n = 1
        cs.dist[0] = c0.dist
        cs.pos[0] = c0.pos
        cs.normal[0] = c0.normal
        if multi and o1.kind != ELLIPSOID and o2.kind != ELLIPSOID and o1.kind != SPHERE and o2.kind != SPHERE:  # mjc_Convex :870
            var p1 = o1.pos
            var q1 = (o1.r0, o1.r1, o1.r2)
            var p2 = o2.pos
            var q2 = (o2.r0, o2.r1, o2.r2)
            var tol_d = S(1e-3) * rbound
            # frame of the first contact: mju_makeFrame(normal)
            var nrm = c0.normal
            var yax = V(0, 1, 0, 0) if (nrm[1] < 0.5 and nrm[1] > -0.5) else V(0, 0, 1, 0)
            yax = yax - nrm * dot(nrm, yax)
            yax = normalize3(yax)
            var zax = cross(nrm, yax)
            for axis_id in range(2):
                for angle_id in range(2):
                    var axis = yax if axis_id == 0 else zax
                    var sgn = S(-1) if angle_id == 0 else S(1)
                    var rot = tilt(axis, sgn)
                    var inv = Array[S, 9](fill=0)  # transpose
                    for a in range(3):
                        for b in range(3):
                            inv[3 * a + b] = rot[3 * b + a]
                    rotate_frame(c0.pos, rot, o1)
                    rotate_frame(c0.pos, inv, o2)
                    var c = penetration(o1, o2, margin, tol, 1)
                    if c.n > 0:
                        var distinct = True  # mjc_isDistinctContact: the new one against every earlier
                        for i in range(cs.n):
                            var dd = c.pos - cs.pos[i]
                            if sqrt(dot(dd, dd)) <= tol_d:
                                distinct = False
                        if distinct:
                            cs.dist[cs.n] = c0.dist
                            cs.pos[cs.n] = c.pos
                            cs.normal[cs.n] = c.normal
                            cs.n += 1
                    o1.pos = p1
                    o1.r0 = q1[0]
                    o1.r1 = q1[1]
                    o1.r2 = q1[2]
                    o2.pos = p2
                    o2.r0 = q2[0]
                    o2.r1 = q2[1]
                    o2.r2 = q2[2]
    for i in range(cs.n):
        cs.pos[i] = cs.pos[i] + org
    o1.pos = o1.pos + org
    o2.pos = org
    return cs^


# ---------------------------------------- heightfield (engine_collision_convex.c:1100) ------------------------

comptime HF_SLOTS = get_defined_int["HF_SLOTS", 50]()  # contacts kept per heightfield pair (mjMAXCONPAIR 50)


trait Height:
    def h(self, r: Int, c: Int) -> S:
        """Height of grid node (row r, column c) above the hfield frame's origin: hfield_data * size[2]."""
        ...


struct HCons(Copyable):
    """Contacts of one heightfield pair in the hfield frame, in prism order."""
    var n: Int
    var dist: Array[S, HF_SLOTS]
    var pos: Array[V, HF_SLOTS]
    var normal: Array[V, HF_SLOTS]

    def __init__(out self):
        self.n = 0
        self.dist = Array[S, HF_SLOTS](fill=0)
        self.pos = Array[V, HF_SLOTS](fill=V(0))
        self.normal = Array[V, HF_SLOTS](fill=V(0))


@always_inline
def add_prism_vert(mut o: Sp, hf: Some[Height], r: Int, c: Int, i: Int, dx: S, dy: S, size0: S, size1: S, margin: S):
    """addPrismVert (engine_collision_convex.c:1100): slide the prism by one vertex, the new one at the last position."""
    o.prism[0] = o.prism[1]
    o.prism[1] = o.prism[2]
    o.prism[3] = o.prism[4]
    o.prism[4] = o.prism[5]
    var dr = 1 - i
    var x = dx * S(c) - size0
    var y = dy * S(r + dr) - size1
    o.prism[2] = V(x, y, o.prism[2][2], 0)
    o.prism[5] = V(x, y, hf.h(r + dr, c) + margin, 0)


def collide_hfield(hf: Some[Height], mut mesh: Sp, nrow: Int, ncol: Int, size0: S, size1: S, size2: S, size3: S, margin: S,
                   tol: S, rb2: S, endless: Bool, bottom_prism: V) -> HCons:
    """mjc_ConvexHField (engine_collision_convex.c:1120) with native CCD, in the hfield frame (the mesh's pose is already
    expressed in it: pos, rotation rows): the sphere and AABB early returns, the sub-grid under the mesh's AABB, one
    GJK + EPA per cell-triangle prism whose top is not below the mesh, the prism's top raised by the margin and the mesh
    inflated by it. Contacts in prism order, at most HF_SLOTS. endless: no clamp to the grid (the engine's terrain is a
    function; nodes outside nrow x ncol exist). bottom_prism is unused padding for call symmetry."""
    var cs = HCons()
    var lp = mesh.pos
    var radius = rb2 + margin
    if size0 < lp[0] - radius or -size0 > lp[0] + radius or size1 < lp[1] - radius or -size1 > lp[1] + radius:
        if not endless:
            return cs^
    if size2 < lp[2] - radius or -size3 > lp[2] + radius:
        if not endless:
            return cs^
    mesh.margin = 0
    var xmax = mesh.support(V(1, 0, 0, 0))[0]
    var xmin = mesh.support(V(-1, 0, 0, 0))[0]
    var ymax = mesh.support(V(0, 1, 0, 0))[1]
    var ymin = mesh.support(V(0, -1, 0, 0))[1]
    var zmax = mesh.support(V(0, 0, 1, 0))[2]
    var zmin = mesh.support(V(0, 0, -1, 0))[2]
    if not endless:
        if xmin - margin > size0 or xmax + margin < -size0 or ymin - margin > size1 or ymax + margin < -size1 or zmin - margin > size2 or zmax + margin < -size3:
            return cs^
    var dx = (2 * size0) / S(ncol - 1)
    var dy = (2 * size1) / S(nrow - 1)
    var cmin = Int(floor((xmin + size0) / (2 * size0) * S(ncol - 1)))
    var cmax = Int(ceil((xmax + size0) / (2 * size0) * S(ncol - 1)))
    var rmin = Int(floor((ymin + size1) / (2 * size1) * S(nrow - 1)))
    var rmax = Int(ceil((ymax + size1) / (2 * size1) * S(nrow - 1)))
    if not endless:
        cmin = max(0, cmin)
        cmax = min(ncol - 1, cmax)
        rmin = max(0, rmin)
        rmax = min(nrow - 1, rmax)
    mesh.margin = margin
    var o1 = Sp(mesh.verts, mesh.first, mesh.edges)
    o1.kind = PRISM
    o1.prism[0] = V(0, 0, -size3, 0)
    o1.prism[1] = V(0, 0, -size3, 0)
    o1.prism[2] = V(0, 0, -size3, 0)
    var r = rmin
    while r < rmax:
        add_prism_vert(o1, hf, r, cmin, 0, dx, dy, size0, size1, margin)
        add_prism_vert(o1, hf, r, cmin, 1, dx, dy, size0, size1, margin)
        var c = cmin + 1
        while c <= cmax:
            for i in range(2):
                add_prism_vert(o1, hf, r, c, i, dx, dy, size0, size1, margin)
                if o1.prism[3][2] < zmin and o1.prism[4][2] < zmin and o1.prism[5][2] < zmin:
                    continue
                var st = Status()
                st.tolerance = tol
                st.max_contacts = 1
                o1.margin = 0
                var d = ccd(st, o1, mesh)
                if d < 0:
                    cs.dist[cs.n] = d
                    cs.pos[cs.n] = (st.x1 + st.x2) * S(0.5)
                    cs.normal[cs.n] = contact_normal(st)
                    cs.n += 1
                    if cs.n >= HF_SLOTS:
                        return cs^
            c += 1
        r += 1
    return cs^
