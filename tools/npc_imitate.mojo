# The imitated base readout of npc-mind seeds 0..S-1 (sweep_gpu.imitate_all: scripted schedule -> SGD fit), for
# npc_live's base arm (private note BRAIN). Built against npc-mind alone: its world.mojo shares a module name with
# kinestride's. usage: npc_imitate S OUT  -> OUT/wout_<s>.bin (float64 A x H per seed); prints the train accuracy.
# build: mojo build -I <npc-mind> tools/npc_imitate.mojo
from sweep_gpu import imitate_all
from std.sys import argv


def main() raises:
    var s = Int(String(argv()[1]))
    var out = String(argv()[2])
    var im = imitate_all(s)
    for k in range(s):
        var w = im[0][k].copy()
        var fh = open(out + "/wout_" + String(k) + ".bin", "w")
        fh.write_bytes(Span(unsafe_ptr=w.unsafe_ptr().unsafe_bitcast[UInt8](), length=8 * len(w)))
        fh.close()
        print("seed", k, "imitation acc", im[1][k], "weights", len(w))
