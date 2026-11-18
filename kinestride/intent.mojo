"""Grid intent -> gait drives for a body that carries an npc-mind brain (private note BRAIN B3). Host only: a few numbers
per animal per control step.

The npc-mind world says where its NPC should be after a tick: a tile (x east, y south) and a facing (0 N, 1 E, 2 S, 3 W).
The body walks there: `drive` steers the spider gait (fly-connectome spider_brain.gait: speed in [0, 1], turn in [-1, 1])
toward the tile centre, then turns in place to the facing. Measured in this engine (npc_live calib, 1.6 s): speed 1 walks
30.7 mm along +x; turn +1 at speed 0.6 turns -296 degrees, clockwise seen from above (Z up). Spin (0.8 s): speed 0.6
turn 1 turns 148 degrees and drifts 8.4 mm; speed 0.5 turn 2 (the sides sweep opposite ways) 257 degrees, 4.0 mm. Engine frame:
X = east, Y = north, mm; yaw = the body's +x axis (the spider walks along it) counter-clockwise from +X.
"""
from std.math import atan2, cos, floor, sin

comptime TILE = Float32(10)  # mm per tile: a spider walks ~18 mm/s at speed 1 (spider_gait.py), ~1.4 tiles per tick
comptime PI = Float32(3.14159265)
comptime ARRIVE = Float32(0.2)  # tiles: closer than this to the target centre counts as there; an arrived body walks
# again only when it leaves the target tile (turning in place drifts it a few mm)
comptime FACE_TOL = Float32(0.2)  # rad: closer than this to the target facing stops turning
comptime KTURN = Float32(2)  # turn drive per rad of heading error
comptime WIDE = Float32(1.0)  # rad: a heading error above this spins in place
comptime CCW = Float32(-1)  # the turn drive that turns the body counter-clockwise (calib: +1 turns clockwise)
comptime SPIN = Float32(0.5)  # spin in place: speed SPIN, turn +-TURN (sweep = (1 + turn side) speed, so |turn| > 1
comptime TURN = Float32(2)  # sweeps the two sides in opposite directions; the gait cannot turn at speed 0)


def wrap(a: Float32) -> Float32:
    """a in (-pi, pi]."""
    var t = a - 2 * PI * floor((a + PI) / (2 * PI))
    return t


def yaw_of(qw: Float32, qx: Float32, qy: Float32, qz: Float32) -> Float32:
    """Yaw of the body's +x axis from its orientation quaternion."""
    return atan2(2 * (qx * qy + qw * qz), 1 - 2 * (qy * qy + qz * qz))


def face_yaw(face: Int) -> Float32:
    """N (0) is +Y, E (1) is +X."""
    return wrap(PI / 2 - Float32(face) * PI / 2)


def face_of(yaw: Float32) -> Int:
    """The facing nearest to a yaw."""
    var k = Int(floor((PI / 2 - yaw) / (PI / 2) + Float32(0.5)))
    return ((k % 4) + 4) % 4


def centre(x: Int, y: Int) -> Tuple[Float32, Float32]:
    """Engine X, Y (mm) of a tile centre."""
    return ((Float32(x) + Float32(0.5)) * TILE, -(Float32(y) + Float32(0.5)) * TILE)


def tile_of(px: Float32, py: Float32) -> Tuple[Int, Int]:
    return (Int(floor(px / TILE)), Int(floor(-py / TILE)))


def drive(px: Float32, py: Float32, yaw: Float32, tx: Int, ty: Int, tface: Int, mut arrived: Bool) -> Tuple[Float32, Float32]:
    """(speed, turn) toward tile (tx, ty), then to facing tface. arrived: the body reached the centre, or started the
    tick in the target tile (the caller sets it per tick); it stays set while the body is in the target tile."""
    var c = centre(tx, ty)
    var ex = c[0] - px
    var ey = c[1] - py
    var tl = tile_of(px, py)
    if arrived:
        arrived = tl[0] == tx and tl[1] == ty
    else:
        arrived = ex * ex + ey * ey <= (ARRIVE * TILE) * (ARRIVE * TILE)
    var err: Float32
    if not arrived:
        err = wrap(atan2(ey, ex) - yaw)
        if abs(err) > WIDE:
            return (SPIN, TURN * (CCW if err > 0 else -CCW))
        return (Float32(1), CCW * max(Float32(-1), min(Float32(1), KTURN * err)))
    err = wrap(face_yaw(tface) - yaw)
    if abs(err) < FACE_TOL:
        return (Float32(0), Float32(0))
    return (SPIN, TURN * (CCW if err > 0 else -CCW))
