"""brain_view.mojo check (GPU, under gpu-wait): the FlyWire layout drawn at rest and with the DNp09 group (walk) lit.
Gates: every drawn DNp09 neuron's pixel is brighter lit than at rest, a pixel far from every neuron is the same in both,
the layout loads all neurons; draw (activity upload + splat + tone) timed. OUT (.ppm) gets rest | lit | everything lit.

usage: gpu-wait run --priority 90 --vram 1 --timeout 300 -- BINARY [LAYOUT_DIR/] [OUT.ppm]
       (build: mojo build -I kinestride kinestride/brain_view_test.mojo)
"""
from std.math import cos, sin
from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from brain_view import BH, BW, BrainView, project


def pixels(ctx: DeviceContext, bv: BrainView) raises -> List[UInt32]:
    var out = List[UInt32](length=BW * BH, fill=0)
    ctx.enqueue_copy(out.unsafe_ptr(), bv.img)
    ctx.synchronize()
    return out^


def lum(p: UInt32) -> Int:
    return Int(p & 255) + Int((p >> 8) & 255) + Int((p >> 16) & 255)


def main() raises:
    var dir = String(argv()[1]) if len(argv()) > 1 else getenv("HOME") + "/Projects/science/fly-connectome/.work/agent/body_engine/brain-layout/flywire/"
    var ctx = DeviceContext()
    var bv = BrainView(ctx, dir)
    var gb = open(dir + "groups.bin", "r").read_bytes()
    var g = gb.unsafe_ptr().unsafe_bitcast[Int32]()
    var ids = List[Int]()  # group 0 of groups.txt: DNp09
    for k in range(Int(g[1])):
        ids.append(Int(g[2 + k]))
    var cls = open(dir + "class.bin", "r").read_bytes()
    var yaw, pitch, zoom = Float32(0.35), Float32(0.25), Float32(150)
    var act = List[UInt8](length=bv.n, fill=0)
    bv.draw(ctx, Int(act.unsafe_ptr()), yaw, pitch, zoom)
    var rest = pixels(ctx, bv)
    for i in ids:
        act[i] = 255
    bv.draw(ctx, Int(act.unsafe_ptr()), yaw, pitch, zoom)
    var lit = pixels(ctx, bv)
    var fails = 0
    var brighter = 0
    for i in ids:
        var p = project(bv.host_xyz[3 * i], bv.host_xyz[3 * i + 1], bv.host_xyz[3 * i + 2], cos(yaw), sin(yaw), cos(pitch), sin(pitch), zoom)
        var o = p[1] * BW + p[0]
        print("DNp09 neuron", i, "class", Int(cls[i]), "pixel", p[0], p[1], "luminance rest", lum(rest[o]), "lit", lum(lit[o]))
        if lum(lit[o]) >= lum(rest[o]) + 60:
            brighter += 1
    if brighter != len(ids):
        fails += 1
    if rest[2 * BW + 2] != lit[2 * BW + 2]:  # the corner: no neuron there
        fails += 1
    var t0 = perf_counter_ns()
    for _ in range(100):
        bv.draw(ctx, Int(act.unsafe_ptr()), yaw, pitch, zoom)
    var us = Float64(perf_counter_ns() - t0) / 100 / 1000
    print("neurons", bv.n, "; DNp09 brighter", brighter, "of", len(ids), "; draw", us, "us per frame at", BW, "x", BH)
    if len(argv()) > 2:
        for i in range(bv.n):
            act[i] = 255
        bv.draw(ctx, Int(act.unsafe_ptr()), yaw, pitch, zoom)
        var all = pixels(ctx, bv)
        var ppm = List[UInt8](capacity=3 * BW * BH * 3)
        for y in range(BH):
            for panel in range(3):
                for x in range(BW):
                    var p = rest[y * BW + x] if panel == 0 else (lit[y * BW + x] if panel == 1 else all[y * BW + x])
                    ppm.append(UInt8(p & 255))
                    ppm.append(UInt8((p >> 8) & 255))
                    ppm.append(UInt8((p >> 16) & 255))
        with open(String(argv()[2]), "w") as o:
            o.write("P6\n" + String(3 * BW) + " " + String(BH) + "\n255\n")
            o.write_bytes(Span(ppm))
    if fails:
        raise Error("FAIL " + String(fails) + " brain view checks")
    print("PASS brain_view_test")
