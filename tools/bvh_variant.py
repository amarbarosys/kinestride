"""Rewrite a CMU BVH into a second topology with the same motion, for check.sh bvh's retarget round trip.

Changes, all motion-preserving: names Mixamo-style with the "mixamorig:" prefix; units in centimetres (run bvh with
SCALE=0.01); the spine bone Spine -> Spine1 split by an inserted joint at its midpoint (identity rotation); a twist leaf
joint with its own random rotation under each forearm and upper leg (leaves never carry the chain); the root's children
listed in reverse order. The canonical retarget of this file must equal the original's.

usage: python tools/bvh_variant.py IN.bvh OUT.bvh
"""

from __future__ import annotations

import sys

import numpy as np

RENAME = {"Spine": "Spine", "Spine1": "Spine2", "LeftUpLeg": "LeftUpLeg", "LeftLeg": "LeftLeg", "LeftToeBase": "LeftToeBase"}
CM = 0.056444 / 0.01


def parse(path):
    """Joints as dicts (name, offset, channels, children, end, col) in file order, the frame time and the motion."""
    tok = open(path).read().split()
    i = 1  # after HIERARCHY
    joints = []

    def joint():
        nonlocal i
        j = {"name": tok[i + 1], "children": [], "end": None}  # ROOT|JOINT name {
        joints.append(j)
        j["offset"] = [float(v) for v in tok[i + 4:i + 7]]  # OFFSET x y z
        n = int(tok[i + 8])
        j["channels"] = tok[i + 9:i + 9 + n]
        i += 9 + n
        while tok[i] != "}":
            if tok[i] == "End":  # End Site { OFFSET x y z }
                j["end"] = [float(v) for v in tok[i + 4:i + 7]]
                i += 8
            else:
                j["children"].append(joint())
        i += 1
        return j

    root = joint()
    t, dt = int(tok[i + 2]), tok[i + 5]  # MOTION Frames: T Frame Time: dt
    motion = np.array(tok[i + 6:], float).reshape(t, -1)
    col = 0
    for j in joints:
        j["col"] = col
        col += len(j["channels"])
    return root, joints, dt, motion


def main() -> None:
    root, joints, dt, motion = parse(sys.argv[1])
    rng = np.random.default_rng(3)
    cols = []  # output motion columns: (source column or None for zeros, or a fixed random triple)

    def name(n):
        return "mixamorig:" + RENAME.get(n, n)

    def emit(j, depth, out):
        pad = "  " * depth
        kw = "ROOT" if j is root else "JOINT"
        out.append(f"{pad}{kw} {name(j['name'])}\n{pad}{{\n{pad}  OFFSET " + " ".join(f"{v * CM:.6f}" for v in j["offset"]))
        out.append(f"{pad}  CHANNELS {len(j['channels'])} " + " ".join(j["channels"]))
        cols.extend(("src", j["col"] + k) for k in range(len(j["channels"])))
        kids = list(reversed(j["children"])) if j is root else j["children"]
        for c in kids:
            if j["name"] == "Spine" and c["name"] == "Spine1":  # split the bone: a joint at its midpoint, identity rotation
                half = [v / 2 for v in c["offset"]]
                out.append(f"{pad}  JOINT mixamorig:Spine1\n{pad}  {{\n{pad}    OFFSET " + " ".join(f"{v * CM:.6f}" for v in half))
                out.append(f"{pad}    CHANNELS 3 Zrotation Yrotation Xrotation")
                cols.extend(("zero", 0) for _ in range(3))
                emit(dict(c, offset=half), depth + 2, out)
                out.append(f"{pad}  }}")
            else:
                emit(c, depth + 1, out)
        if j["name"] in ("LeftForeArm", "RightForeArm", "LeftUpLeg", "RightUpLeg"):  # a twist leaf, its own rotation
            out.append(f"{pad}  JOINT {name(j['name'])}Twist\n{pad}  {{\n{pad}    OFFSET 0 0 0")
            out.append(f"{pad}    CHANNELS 3 Zrotation Yrotation Xrotation")
            cols.extend(("rand", k) for k in range(3))
            out.append(f"{pad}    End Site\n{pad}    {{\n{pad}      OFFSET 0 {2 * CM:.6f} 0\n{pad}    }}\n{pad}  }}")
        if j["end"] is not None:
            out.append(f"{pad}  End Site\n{pad}  {{\n{pad}    OFFSET " + " ".join(f"{v * CM:.6f}" for v in j["end"]) + f"\n{pad}  }}")
        out.append(f"{pad}}}")

    out = ["HIERARCHY"]
    emit(root, 0, out)
    rand = rng.uniform(-60, 60, (len(motion), 3))
    data = np.zeros((len(motion), len(cols)))
    for i, (kind, k) in enumerate(cols):
        if kind == "src":
            data[:, i] = motion[:, k] * (CM if k < 3 else 1)  # root position channels in centimetres
        elif kind == "rand":
            data[:, i] = rand[:, k]
    out.append(f"MOTION\nFrames: {len(motion)}\nFrame Time: {dt}")
    out.extend(" ".join(f"{v:.6f}" for v in r) for r in data)
    open(sys.argv[2], "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
