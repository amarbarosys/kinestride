"""Gate for bvh.mojo, three ways: Mojo's forward kinematics (.mclip, quaternions), an independent numpy one (rotation
matrices, its own BVH parser), and MuJoCo's mj_kinematics on the generated MJCF driven by the .qref rows.

1. .mclip positions within 1e-5 m and orientation matrices within 1e-5 of numpy, every body, every frame
2. MuJoCo xpos / xmat from .qref within the same bars of numpy
3. contacts equal to numpy's (four foot bodies: within 3 cm of their 10th-percentile z, horizontal speed under 0.5 m/s)
4. the body stands: head over the toes by more than 1.2 m on average, toes' lowest z within [-0.1, 0.15] m,
   every body has mass, 70 kg in all (MASS default; catches an upside-down or mis-scaled character, which a
   shared convention would hide)

usage: python -m kinestride.bvh_check BINARY CLIP.bvh
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work/bvh"
SCALE = 0.056444
C = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # y-up BVH -> z-up MuJoCo: (x, y, z) -> (x, -z, y)
FEET = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
TOL = 1e-5


def parse(path):
    names, parents, offsets, orders = [], [], [], []
    stack, lines = [], iter(Path(path).read_text().split("\n"))
    kind = None
    for line in lines:
        w = line.split()
        if not w:
            continue
        if w[0] in ("ROOT", "JOINT"):
            parents.append(next((s for s in reversed(stack) if s is not None), -1))
            names.append(w[1])
            kind = "joint"
        elif w[0] == "End":
            kind = "end"
        elif w[0] == "{":
            stack.append(len(names) - 1 if kind == "joint" else None)
        elif w[0] == "}":
            stack.pop()
        elif w[0] == "OFFSET" and kind == "joint":
            offsets.append([float(v) for v in w[1:4]])
        elif w[0] == "CHANNELS":
            orders.append([c[0].lower() for c in w[2:] if c.endswith("rotation")])
        elif w[0] == "MOTION":
            break
    t = int(next(lines).split()[1])
    dt = float(next(lines).split()[2])
    motion = np.array([[float(v) for v in next(lines).split()] for _ in range(t)])
    return names, parents, np.array(offsets), orders, dt, motion[int(os.environ.get("FIRST", 1)):][:int(os.environ.get("FRAMES", t))]  # bvh.mojo's FIRST, FRAMES


def rot(axis, deg):
    a = np.radians(deg)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    m = {"x": [[o, z, z], [z, c, -s], [z, s, c]], "y": [[c, z, s], [z, o, z], [-s, z, c]],
         "z": [[c, -s, z], [s, c, z], [z, z, o]]}[axis]
    return np.moveaxis(np.array(m), -1, 0)


def fk(names, parents, offsets, orders, motion):
    t, nj = len(motion), len(names)
    pos, wr = np.zeros((t, nj, 3)), np.zeros((t, nj, 3, 3))
    for j in range(nj):
        r = np.broadcast_to(np.eye(3), (t, 3, 3))
        for c, axis in enumerate(orders[j]):
            r = r @ rot(axis, motion[:, 3 + 3 * j + c])
        if j == 0:
            pos[:, 0], wr[:, 0] = motion[:, :3] * SCALE, r
        else:
            p = parents[j]
            pos[:, j] = pos[:, p] + wr[:, p] @ (offsets[j] * SCALE)
            wr[:, j] = wr[:, p] @ r
    return pos @ C.T, C @ wr @ C.T


def qmat(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def main() -> None:
    binary, clip = sys.argv[1], sys.argv[2]
    WORK.mkdir(parents=True, exist_ok=True)
    stem = WORK / Path(clip).stem
    for cmd in (["mjcf", clip, f"{stem}.xml"], ["clip", clip, str(stem)]):
        subprocess.run([binary, *cmd], check=True, stdout=subprocess.DEVNULL)
    names, parents, offsets, orders, dt, motion = parse(clip)
    pos, wr = fk(names, parents, offsets, orders, motion)
    t, nj = pos.shape[:2]
    fails = []

    def bar(what, err):
        ok = err <= TOL
        print(f"{'ok  ' if ok else 'FAIL'} {what}: max err {err:.3g}")
        if not ok:
            fails.append(what)

    raw = Path(f"{stem}.mclip").read_bytes()
    mt, mnb, mnf = struct.unpack_from("<3i", raw)
    feet = np.frombuffer(raw, "<i4", mnf, 16)
    vals = np.frombuffer(raw, "<f4", offset=16 + 4 * mnf).astype(np.float64)
    if (mt, mnb) != (t, nj) or [names[b] for b in feet] != FEET:
        print(f"FAIL bvh mclip shape: {mt}x{mnb}, feet {[names[b] for b in feet]}")
        sys.exit(1)
    mpos = vals[:t * nj * 3].reshape(t, nj, 3)
    mquat = vals[t * nj * 3:t * nj * 7].reshape(t, nj, 4)
    mcon = vals[t * nj * 7:].reshape(t, mnf)
    bar("mclip positions vs numpy (m)", np.abs(mpos - pos).max())
    bar("mclip orientations vs numpy", np.abs(qmat(mquat) - wr).max())

    model = mujoco.MjModel.from_xml_path(f"{stem}.xml")
    data = mujoco.MjData(model)
    qraw = Path(f"{stem}.qref").read_bytes()
    qt, nq = struct.unpack_from("<2i", qraw)
    hz = struct.unpack_from("<f", qraw, 8)[0]
    qref = np.frombuffer(qraw, "<f4", offset=12).reshape(qt, nq).astype(np.float64)
    if (qt, nq) != (t, model.nq) or abs(hz - np.float32(1 / dt)) > 1e-3:
        print(f"FAIL bvh qref shape: {qt}x{nq} hz {hz}, model nq {model.nq}, clip {t} frames at {1 / dt}")
        sys.exit(1)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    xpos, xmat = np.zeros_like(pos), np.zeros_like(wr)
    for f in range(t):
        data.qpos[:] = qref[f]
        mujoco.mj_kinematics(model, data)
        xpos[f] = data.xpos[ids]
        xmat[f] = data.xmat[ids].reshape(nj, 3, 3)
    bar("MuJoCo(qref) positions vs numpy (m)", np.abs(xpos - pos).max())
    bar("MuJoCo(qref) orientations vs numpy", np.abs(xmat - wr).max())

    fi = [names.index(n) for n in FEET]
    z = pos[:, fi, 2]
    nxt = np.r_[np.arange(1, t), t - 2]
    speed = np.linalg.norm(pos[nxt][:, fi, :2] - pos[:, fi, :2], axis=-1) / dt
    con = (z < np.percentile(z, 10, axis=0, method="lower") + 0.03) & (speed < 0.5)
    miss = int(((mcon > 0.5) != con).sum())
    print(f"{'ok  ' if miss == 0 else 'FAIL'} contacts vs numpy: {miss} of {con.size} differ, {int(con.sum())} in contact")
    if miss:
        fails.append("contacts")

    toes = pos[:, [names.index("LeftToeBase"), names.index("RightToeBase")], 2]
    rise = (pos[:, names.index("Head"), 2] - toes.mean(1)).mean()
    low = toes.min()
    stands = rise > 1.2 and -0.1 <= low <= 0.15 and (model.body_mass[ids] > 0).all() and abs(model.body_mass.sum() - 70) < 0.1
    print(f"{'ok  ' if stands else 'FAIL'} stands: head over toes {rise:.3f} m, lowest toe {low:.3f} m, "
          f"lightest body {model.body_mass[ids].min():.3g} kg, total {model.body_mass.sum():.1f} kg")
    if not stands:
        fails.append("stands")

    if fails:
        print(f"FAIL {len(fails)}/7 bvh checks: {', '.join(fails)}")
        sys.exit(1)
    print(f"PASS bvh {Path(clip).name}: {t} frames x {nj} bodies, three-way kinematics within {TOL}, contacts equal, stands")


if __name__ == "__main__":
    main()
