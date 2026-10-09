"""Camera fitting helpers for the body engine spectator.

No MuJoCo function is ported here: this is view framing, not engine physics.
The camera format is eye xyz, target xyz, and vertical field of view in degrees,
matching render.mojo's existing camera arguments.
"""

from std.math import atan, cos, sin, sqrt, tan

@fieldwise_init
struct Camera:
    var ex: Float32
    var ey: Float32
    var ez: Float32
    var tx: Float32
    var ty: Float32
    var tz: Float32
    var fovy: Float32


comptime DIR_X: Float32 = 0.655  # eye offset direction: (DIR_X, -DIR_X, DIR_Z), unit length
comptime DIR_Z: Float32 = 0.376


# Distance at which a sphere of `radius` around the target fits the limiting (smaller) field of view with 5 % margin.
# Shared by fit_camera (host) and spectate.follow (device), so one formula frames both.
@always_inline
def fit_distance(radius: Float32, width: Int, height: Int, fovy: Float32) -> Float32:
    # sin(half of the limiting angle) without atan/tan: libm is CPU-only, and spectate.follow runs this on the card
    var half = fovy * 0.00872664626
    var t = sin(half) / cos(half)
    var tl = min(t, t * Float32(width) / Float32(height))
    return 1.05 * radius * sqrt(1.0 + tl * tl) / tl


# `roots` is xyz triples and `radii` has one conservative world-space radius
# per root. The target is the mean root position; distance includes root spread.
def fit_camera(roots: List[Float32], radii: List[Float32], width: Int,
               height: Int, fovy: Float32) -> Camera:
    var n = len(radii)
    var tx: Float32 = 0.0
    var ty: Float32 = 0.0
    var tz: Float32 = 0.0
    for i in range(n):
        tx += roots[3 * i]
        ty += roots[3 * i + 1]
        tz += roots[3 * i + 2]
    if n > 0:
        tx /= Float32(n)
        ty /= Float32(n)
        tz /= Float32(n)
    var radius: Float32 = 0.0
    for i in range(n):
        var dx = roots[3 * i] - tx
        var dy = roots[3 * i + 1] - ty
        var dz = roots[3 * i + 2] - tz
        var d = sqrt(dx * dx + dy * dy + dz * dz) + radii[i]
        if d > radius:
            radius = d
    var distance = fit_distance(radius, width, height, fovy)
    # Fixed elevated oblique view. The target remains the exact fitted center.
    var ex = tx + distance * DIR_X
    var ey = ty - distance * DIR_X
    var ez = tz + distance * DIR_Z
    return Camera(ex, ey, ez, tx, ty, tz, fovy)


def main() raises:
    var roots: List[Float32] = [0.0, 0.0, 0.0, 12.0, -4.0, 2.0]
    var radii: List[Float32] = [3.0, 5.0]
    for dims in [(960, 540), (540, 960)]:
        var cam = fit_camera(roots, radii, dims[0], dims[1], 45.0)
        if abs(cam.tx - 6.0) > 1e-5 or abs(cam.ty + 2.0) > 1e-5 or abs(cam.tz - 1.0) > 1e-5:
            raise Error("camera target missed the mean root position")
        if cam.ez <= cam.tz:
            raise Error("camera must look down toward the framed scene")
        var dx = cam.ex - cam.tx
        var dy = cam.ey - cam.ty
        var dz = cam.ez - cam.tz
        var distance = sqrt(dx * dx + dy * dy + dz * dz)
        var aspect = Float32(dims[0]) / Float32(dims[1])
        var half_v = cam.fovy * 0.00872664626  # half angles in radians
        var half_h = atan(tan(half_v) * aspect)
        var limiting = half_v
        if half_h < limiting:
            limiting = half_h
        var required = sin(limiting) * distance
        var needed: Float32 = 0.0
        for i in range(len(radii)):
            var rx = roots[3 * i] - cam.tx
            var ry = roots[3 * i + 1] - cam.ty
            var rz = roots[3 * i + 2] - cam.tz
            var bound = sqrt(rx * rx + ry * ry + rz * rz) + radii[i]
            if bound > needed:
                needed = bound
        if required + 1e-4 < needed:
            raise Error("root sphere falls outside the limiting FOV")
        print("PASS camera fixture", dims[0], dims[1], cam.tx, cam.ty, cam.tz, cam.fovy)
