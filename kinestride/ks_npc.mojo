"""npc-mind NPCs as a C shared library for Godot (docs/plans/GODOT_SPIKE.md N1): the base brain arm on the GPU and
npc-mind's world, ticked exactly as npc_live's `base grid` run (same seed calls, same per-tick u draws), so a C or
Godot host reproduces that trace byte for byte. ks_npc_think runs the brains alone on observations a game supplies.

ks_npc_create(dir, seed) -> handle (0 on error; DIR holds wout_<seed>.bin from tools/npc_imitate), ks_npc_tick(h, n),
ks_npc_state(h, int32* dst): per NPC x, y, face, last body action; per food x, y, wait; per forge x, y; per wolf x, y;
the G x G tree map (ks_npc_nstate() ints). ks_npc_think(h, float* obs N x O, int32* acts N x NH). ks_npc_destroy(h).
build: mojo build --emit shared-lib -I <npc-mind> -I <inc with npc_world.mojo -> npc-mind world.mojo> kinestride/ks_npc.mojo"""
from std.memory.alloc import alloc, Layout
from std.random import random_float64, seed
from mind import NH, O
from mind_gpu import Brains
from npc_world import G, N, NFOOD, NFORGE, NWOLF, World, make_base, make_minds

comptime NSTATE = 4 * N + 3 * NFOOD + 2 * NFORGE + 2 * NWOLF + G * G
comptime U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]


struct Npcs(Movable):
    var g: Brains
    var w: World

    def __init__(out self, dir: String, sd: Int) raises:  # npc_live main: base arm seeding, then the test world
        self.g = Brains(N, 1)
        seed(sd * 7919 + 1)
        var base = make_base()
        var raw = open(dir + "wout_" + String(sd) + ".bin", "r").read_bytes()
        if len(raw) != 8 * len(base.wout):
            raise Error("wout size")
        for k in range(len(base.wout)):
            base.wout[k] = raw.unsafe_ptr().unsafe_bitcast[Float64]().unsafe_offset(k)[]
        self.g.set_base(0, base)
        seed(sd * 7919 + 3)
        self.g.set_minds(0, make_minds(True, False), 0, 0)
        seed(sd * 104729 + 2)
        self.w = World()

    def tick(mut self) raises:  # npc_live tick, base arm
        var o = self.w.all_obs()
        for i in range(N * O):
            self.g.obs_h[i] = Float32(o[i])
        for i in range(N * NH):
            self.g.u_h[i] = Float32(random_float64())
        self.g.step()
        var acts = List[Int](capacity=N * NH)
        for i in range(N * NH):
            acts.append(Int(self.g.acts_h[i]))
        _ = self.w.step(acts)


def npcs(h: Int) -> MutPointer[Npcs, MutUntrackedOrigin]:
    return MutPointer[Npcs, MutUntrackedOrigin](unsafe_from_address=h)


@export
def ks_npc_nstate() abi("C") -> Int:
    return NSTATE


@export
def ks_npc_create(dir: U8P, sd: Int) abi("C") -> Int:
    var n = 0
    while dir[unsafe_offset=n] != 0:
        n += 1
    var buf = List[UInt8](length=n, fill=0)
    for i in range(n):
        buf[i] = dir[unsafe_offset=i]
    try:
        var v = Npcs(String(StringSlice(unsafe_from_utf8=Span(buf))), sd)  # may raise: before the allocation exists
        var a = alloc(Layout[Npcs](count=1))
        var p = a.unsafe_ptr()
        p.unsafe_write(v^)
        var h = Int(p)
        _ = a^.unsafe_leak()
        return h
    except e:
        print("FAIL ks_npc_create:", e)
        return 0


@export
def ks_npc_tick(h: Int, n: Int) abi("C") -> Int:
    try:
        ref s = npcs(h)[]
        for _ in range(n):
            s.tick()
        return 0
    except:
        return -1


@export
def ks_npc_state(h: Int, dst: I32P) abi("C"):
    ref w = npcs(h)[].w
    var k = 0
    for i in range(N):
        dst[unsafe_offset=k] = Int32(w.x[i]); dst[unsafe_offset=k + 1] = Int32(w.y[i])
        dst[unsafe_offset=k + 2] = Int32(w.face[i]); dst[unsafe_offset=k + 3] = Int32(w.last[3 * i])
        k += 4
    for i in range(NFOOD):
        dst[unsafe_offset=k] = Int32(w.fx[i]); dst[unsafe_offset=k + 1] = Int32(w.fy[i]); dst[unsafe_offset=k + 2] = Int32(w.fwait[i])
        k += 3
    for i in range(NFORGE):
        dst[unsafe_offset=k] = Int32(w.gx[i]); dst[unsafe_offset=k + 1] = Int32(w.gy[i])
        k += 2
    for i in range(NWOLF):
        dst[unsafe_offset=k] = Int32(w.wx[i]); dst[unsafe_offset=k + 1] = Int32(w.wy[i])
        k += 2
    for i in range(G * G):
        dst[unsafe_offset=k + i] = 1 if w.tree[i] else 0


@export
def ks_npc_think(h: Int, obs: F32P, acts: I32P) abi("C") -> Int:
    try:
        ref g = npcs(h)[].g
        for i in range(N * O):
            g.obs_h[i] = obs[unsafe_offset=i]
        for i in range(N * NH):
            g.u_h[i] = Float32(random_float64())
        g.step()
        for i in range(N * NH):
            acts[unsafe_offset=i] = g.acts_h[i]
        return 0
    except:
        return -1


@export
def ks_npc_destroy(h: Int) abi("C"):
    var p = npcs(h)
    _ = p.take_pointee()
    p.unsafe_free()
