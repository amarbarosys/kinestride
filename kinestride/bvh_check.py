"""Gate for bvh.mojo: the canonical retarget against an independent numpy twin (matrices, its own CMU alias table, the
canonical rest read back from the generated MJCF), and MuJoCo's forward kinematics of the .qref.

1. .mclip positions within 1e-5 m and orientations within 1e-5 of MuJoCo(qref) on the character, every body and frame
2. MuJoCo(qref) bodies vs the twin's retarget on frames no planting window touches (each body: the source joint's world rotation, rest-corrected onto the
   source bone, re-aimed along the source chain when unmapped joints sit inside it; root scaled by hip height,
   then grounded: the lower toe's 10th-percentile height = the 3.5 cm foot radius)
3. contacts equal to the twin's (four foot bodies: within 3 cm of their 10th-percentile z, horizontal speed < 0.5 m/s)
4. the body stands (head over the toes by more than 1.2 m on average, toes' lowest z within [-0.1, 0.15] m), 70 kg,
   every hinge's middle angle under 80 deg (away from gimbal lock)
5. spike repair: bvh's count equals the twin's, no isolated spike left in the repaired source, every frame no spike
   touched equal to a REPAIR=0 run within 1e-5
6. limb planting (legs; arms with HANDZ=2): the end does not move inside a contact run (< 1 mm/frame), no blend frame
   pops (step <= 2x the unplanted step + reach / (BLEND + 1)), the knee or elbow keeps its side of the limb, the end body's
   world orientation and every frame outside the windows equal a PLANT=0 run
7. round trip: tools/bvh_variant.py's Mixamo-style rewrite of the clip (cm, split spine, twist leaves, reordered) gives
   the same .qref within 1e-5

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
FEET = ["LeftFoot", "LeftToes", "RightFoot", "RightToes"]
CMU = {"Hips": "Hips", "Spine": "Spine", "Chest": "Spine1", "Neck": "Neck1", "Head": "Head",
       "LeftShoulder": "LeftShoulder", "LeftUpperArm": "LeftArm", "LeftLowerArm": "LeftForeArm", "LeftHand": "LeftHand",
       "LeftUpperLeg": "LeftUpLeg", "LeftLowerLeg": "LeftLeg", "LeftFoot": "LeftFoot", "LeftToes": "LeftToeBase"}
CMU |= {k.replace("Left", "Right"): v.replace("Left", "Right") for k, v in CMU.items() if k.startswith("Left")}
MAIN = {"Hips": "Spine", "Spine": "Chest", "Chest": "Neck", "Neck": "Head", "LeftShoulder": "LeftUpperArm",
        "LeftUpperArm": "LeftLowerArm", "LeftLowerArm": "LeftHand", "LeftUpperLeg": "LeftLowerLeg",
        "LeftLowerLeg": "LeftFoot", "LeftFoot": "LeftToes"}
MAIN |= {k.replace("Left", "Right"): v.replace("Left", "Right") for k, v in MAIN.items() if k.startswith("Left")}
TOL = 1e-5


def parse(path):
    """BVH -> names, parents, offsets (BVH units), per-joint channel axes, end-site offsets, dt, motion."""
    tok = Path(path).read_text().split()
    names, parents, offsets, orders, ends = [], [], [], [], {}
    i = 1

    def joint(parent):
        nonlocal i
        j = len(names)
        names.append(tok[i + 1].replace("mixamorig:", ""))  # ROOT|JOINT name { OFFSET x y z CHANNELS n ...
        parents.append(parent)
        offsets.append([float(v) for v in tok[i + 4:i + 7]])
        n = int(tok[i + 8])
        orders.append([c[0].lower() for c in tok[i + 9:i + 9 + n] if c.endswith("rotation")])
        i += 9 + n
        while tok[i] != "}":
            if tok[i] == "End":  # End Site { OFFSET x y z }
                ends[j] = [float(v) for v in tok[i + 4:i + 7]]
                i += 8
            else:
                joint(j)
        i += 1

    joint(-1)
    t, dt = int(tok[i + 2]), float(tok[i + 5])
    motion = np.array(tok[i + 6:], float).reshape(t, -1)
    first = int(os.environ.get("FIRST", 1))
    return names, parents, np.array(offsets), orders, ends, dt, motion[first:first + int(os.environ.get("FRAMES", t))]


def qmul(a, b):
    w = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], -1, keepdims=True)
    return np.concatenate([w, a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])], -1)


def angle(a, b):
    d = qmul(a * [1, -1, -1, -1], b)
    return 2 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0]))


def local_quats(orders, motion):
    """[t][joint] local rotation quaternions, channels applied in order (BVH frame)."""
    q = np.zeros((len(motion), len(orders), 4))
    q[..., 0] = 1
    for j, order in enumerate(orders):
        for c, axis in enumerate(order):
            h = np.radians(motion[:, 3 + 3 * j + c]) / 2
            a = np.zeros((len(motion), 4))
            a[:, 0], a[:, 1 + "xyz".index(axis)] = np.cos(h), np.sin(h)
            q[:, j] = qmul(q[:, j], a)
    return q


def spikes(q, hz, cap=30.0):
    """Frames 1..t-2 where a joint is more than cap/hz from both neighbours while they agree within it."""
    a, m, b = q[:-2], q[1:-1], q[2:]
    lim = cap / hz
    return (angle(a, m) > lim) & (angle(m, b) > lim) & (angle(a, b) <= lim)


def repair(q, hz):
    """bvh.mojo's spike repair, judged on the unrepaired clip: a spike becomes its neighbours' midpoint."""
    flag = spikes(q, hz)
    a, b = q[:-2], q[2:]
    mid = a + np.where(np.sum(a * b, -1, keepdims=True) >= 0, 1, -1) * b
    out = q.copy()
    out[1:-1][flag] = (mid / np.linalg.norm(mid, axis=-1, keepdims=True))[flag]
    return out, int(flag.sum())


def qmat(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def fk(names, parents, offsets, quats, motion):
    """World positions (m) and rotations of every source joint, MuJoCo frame."""
    t, nj = len(motion), len(names)
    pos, wr = np.zeros((t, nj, 3)), np.zeros((t, nj, 3, 3))
    for j in range(nj):
        r = qmat(quats[:, j])
        if j == 0:
            pos[:, 0], wr[:, 0] = motion[:, :3] * SCALE, r
        else:
            p = parents[j]
            pos[:, j] = pos[:, p] + wr[:, p] @ (offsets[j] * SCALE)
            wr[:, j] = wr[:, p] @ r
    return pos @ C.T, C @ wr @ C.T


def rmin(a, b):
    """Matrix of the smallest rotation taking unit a to unit b (Rodrigues), batched over leading axes."""
    v, c = np.cross(a, b), np.sum(a * b, -1)[..., None, None]
    k = np.zeros(v.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2], k[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    k -= np.swapaxes(k, -1, -2)
    return np.eye(3) + k + k @ k / (1 + c)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def twin(model, names, parents, offsets, quats, ends, motion):
    """Expected world positions and rotations of the canonical bodies, in MuJoCo body order."""
    spos, swr = fk(names, parents, offsets, quats, motion)
    srest = np.zeros((len(names), 3))
    for j in range(1, len(names)):
        srest[j] = srest[parents[j]] + C @ offsets[j] * SCALE
    stip = {}  # rest vector to the end of each joint's first-child chain
    for j in reversed(range(len(names))):
        kids = [k for k in range(len(names)) if parents[k] == j]
        stip[j] = C @ np.array(ends[j]) * SCALE if j in ends else C @ offsets[kids[0]] * SCALE + stip[kids[0]]
    bname = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    canon = [n for n in bname if n in CMU]
    body = {n: bname.index(n) for n in canon}
    tip = {}
    for n in canon:
        if n in MAIN:
            tip[n] = model.body_pos[body[MAIN[n]]].copy()
        else:  # a leaf: the bone runs between its mesh's two poles, the farthest vertex pair, outward from the joint
            g = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body[n])
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]].astype(float)
            gm = np.zeros(9)
            mujoco.mju_quat2Mat(gm, model.geom_quat[g])
            v = v @ gm.reshape(3, 3).T + model.geom_pos[g]
            far = np.linalg.norm(v[:, None] - v[None], axis=-1)
            i, k = np.unravel_index(far.argmax(), far.shape)
            tip[n] = v[i] - v[k] if np.linalg.norm(v[i]) > np.linalg.norm(v[k]) else v[k] - v[i]
    src = {n: names.index(CMU[n]) for n in canon}
    crest = {}
    for n in canon:
        crest[n] = np.zeros(3) if n == "Hips" else crest[bname[model.body_parentid[body[n]]]] + model.body_pos[body[n]]
    low_c = min(min(crest[n][2], crest[n][2] + tip[n][2]) for n in canon)
    low_s = min(min(srest[j][2], srest[j][2] + stip[j][2]) for j in range(len(names)))
    ratio = -low_c / (srest[src["Hips"]][2] - low_s)
    w, pos = {}, {}
    for n in canon:
        d_rest = srest[src[MAIN[n]]] - srest[src[n]] if n in MAIN else stip[src[n]]
        wb = swr[:, src[n]] @ rmin(unit(tip[n]), unit(d_rest))
        if n != "Hips" and n in MAIN:
            wb = rmin(unit(wb @ tip[n]), unit(spos[:, src[MAIN[n]]] - spos[:, src[n]])) @ wb
        w[n] = wb
        par = bname[model.body_parentid[body[n]]]
        pos[n] = spos[:, src[n]] * ratio if n == "Hips" else pos[par] + w[par] @ model.body_pos[body[n]]
    pos = np.stack([pos[n] for n in canon], 1)
    low = np.minimum(pos[:, canon.index("LeftToes"), 2], pos[:, canon.index("RightToes"), 2])
    radius = 0.035  # the foot capsule: grounded when the lower toe's 10th-percentile height equals it
    pos[..., 2] += radius - np.percentile(low, 10, method="lower")
    return canon, body, pos, np.stack([w[n] for n in canon], 1)


BLEND = 6


def runs(c):
    """Contact runs as (start, end) frame pairs, end exclusive, joined across gaps of up to 2 BLEND frames."""
    out, f = [], 0
    while f < len(c):
        if c[f]:
            s0 = f
            while f < len(c) and c[f]:
                f += 1
            if out and s0 - out[-1][1] <= 2 * BLEND:
                s0 = out.pop()[0]
            out.append((s0, f))
        else:
            f += 1
    return out


def planted(limbs, t):
    """Frames inside any limb's contact run or its BLEND frames on either side."""
    mask = np.zeros(t, bool)
    for *_, c in limbs:
        for s0, e0 in runs(c):
            mask[max(0, s0 - BLEND):min(t, e0 + BLEND)] = True
    return mask


def read_qref(path):
    raw = Path(path).read_bytes()
    t, nq = struct.unpack_from("<2i", raw)
    return struct.unpack_from("<f", raw, 8)[0], np.frombuffer(raw, "<f4", offset=12).reshape(t, nq).astype(np.float64)


def main() -> None:
    binary, clip = sys.argv[1], sys.argv[2]
    WORK.mkdir(parents=True, exist_ok=True)
    stem = WORK / Path(clip).stem
    subprocess.run([binary, "mjcf", f"{WORK}/character.xml"], check=True, stdout=subprocess.DEVNULL)
    said = subprocess.run([binary, "clip", clip, str(stem)], check=True, capture_output=True, text=True).stdout
    mojo_fix = int(said.split("repaired ")[1].split()[0])
    model = mujoco.MjModel.from_xml_path(f"{WORK}/character.xml")
    data = mujoco.MjData(model)
    names, parents, offsets, orders, ends, dt, motion = parse(clip)
    raw_q = local_quats(orders, motion)
    quats, nfix = repair(raw_q, 1 / dt)
    canon, body, tpos, twr = twin(model, names, parents, offsets, quats, ends, motion)
    t, nb = tpos.shape[:2]
    fails = []

    def bar(what, err):
        print(f"{'ok  ' if err <= TOL else 'FAIL'} {what}: max err {err:.3g}")
        if not err <= TOL:
            fails.append(what)

    hz, qref = read_qref(f"{stem}.qref")
    ids = [body[n] for n in canon]
    xpos, xmat = np.zeros_like(tpos), np.zeros_like(twr)
    for f in range(t):
        data.qpos[:] = qref[f]
        mujoco.mj_kinematics(model, data)
        xpos[f], xmat[f] = data.xpos[ids], data.xmat[ids].reshape(nb, 3, 3)
    raw = Path(f"{stem}.mclip").read_bytes()
    mt, mnb, mnf = struct.unpack_from("<3i", raw)
    feet = np.frombuffer(raw, "<i4", mnf, 16)
    vals = np.frombuffer(raw, "<f4", offset=16 + 4 * mnf).astype(np.float64)
    if (mt, mnb) != (t, nb) or [canon[b] for b in feet] != FEET:
        print(f"FAIL bvh mclip shape: {mt}x{mnb}, feet {[canon[b] for b in feet]}")
        sys.exit(1)
    bar("mclip positions vs MuJoCo(qref) (m)", np.abs(vals[:t * nb * 3].reshape(t, nb, 3) - xpos).max())
    bar("mclip orientations vs MuJoCo(qref)", np.abs(qmat(vals[t * nb * 3:t * nb * 7].reshape(t, nb, 4)) - xmat).max())
    fi = [canon.index(n) for n in FEET]
    z = tpos[:, fi, 2]
    nxt = np.r_[np.arange(1, t), t - 2]
    speed = np.linalg.norm(tpos[nxt][:, fi, :2] - tpos[:, fi, :2], axis=-1) / dt
    con = (z < np.percentile(z, 10, axis=0, method="lower") + 0.03) & (speed < 0.5)
    legs = [("LeftUpperLeg", "LeftLowerLeg", "LeftFoot", con[:, 0]), ("RightUpperLeg", "RightLowerLeg", "RightFoot", con[:, 2])]
    free = ~planted(legs, t)
    bar(f"MuJoCo(qref) positions vs twin retarget, {free.sum()} unplanted frames (m)", np.abs(xpos - tpos)[free].max(initial=0))
    bar("MuJoCo(qref) orientations vs twin retarget, unplanted frames", np.abs(xmat - twr)[free].max(initial=0))
    miss = int(((vals[t * nb * 7:].reshape(t, mnf) > 0.5) != con).sum())
    print(f"{'ok  ' if miss == 0 else 'FAIL'} contacts vs twin: {miss} of {con.size} differ, {int(con.sum())} in contact")
    if miss:
        fails.append("contacts")

    toes = tpos[:, [canon.index("LeftToes"), canon.index("RightToes")], 2]
    rise = (tpos[:, canon.index("Head"), 2] - toes.mean(1)).mean()
    middle = np.abs((qref[:, 8::3] + np.pi / 2) % np.pi - np.pi / 2).max() * 180 / np.pi  # principal |b| (b and pi - b)
    stands = rise > 1.2 and -0.1 <= toes.min() <= 0.15 and abs(model.body_mass.sum() - 70) < 0.1 and middle < 80
    print(f"{'ok  ' if stands else 'FAIL'} stands: head over toes {rise:.3f} m, lowest toe {toes.min():.3f} m, "
          f"{model.body_mass.sum():.1f} kg, worst middle hinge angle {middle:.1f} deg")
    if not stands:
        fails.append("stands")

    left = int(spikes(quats, 1 / dt).sum())
    ok = mojo_fix == nfix and left == 0
    print(f"{'ok  ' if ok else 'FAIL'} spike repair: bvh {mojo_fix}, twin {nfix}, isolated spikes left {left}")
    if not ok:
        fails.append("spike repair")
    plain = WORK / f"{Path(clip).stem}_norepair"
    subprocess.run([binary, "clip", clip, str(plain)], check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, REPAIR="0"))
    touched = np.zeros(t, bool)
    touched[1:-1] = spikes(raw_q, 1 / dt).any(1)
    pmat, pref = np.zeros_like(xmat), read_qref(f"{plain}.qref")[1]
    for f in range(t):
        data.qpos[:] = pref[f]
        mujoco.mj_kinematics(model, data)
        pmat[f] = data.xmat[ids].reshape(nb, 3, 3)
    bar(f"untouched frames ({t - touched.sum()} of {t}) vs REPAIR=0", np.abs(pmat - xmat)[~touched].max())

    def kin(path):
        qs, xp, xm = read_qref(path)[1], np.zeros_like(xpos), np.zeros_like(xmat)
        for f in range(t):
            data.qpos[:] = qs[f]
            mujoco.mj_kinematics(model, data)
            xp[f], xm[f] = data.xpos[ids], data.xmat[ids].reshape(nb, 3, 3)
        return xp, xm

    unplanted = WORK / f"{Path(clip).stem}_noplant"
    subprocess.run([binary, "clip", clip, str(unplanted)], check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, PLANT="0"))
    x0p, x0m = kin(f"{unplanted}.qref")
    hands = WORK / f"{Path(clip).stem}_hands"
    subprocess.run([binary, "clip", clip, str(hands)], check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, HANDZ="2"))
    hi = [canon.index("LeftHand"), canon.index("RightHand")]
    hcon = (tpos[:, hi, 2] < 2) & (np.linalg.norm(tpos[nxt][:, hi, :2] - tpos[:, hi, :2], axis=-1) / dt < 0.5)
    arms = [("LeftUpperArm", "LeftLowerArm", "LeftHand", hcon[:, 0]), ("RightUpperArm", "RightLowerArm", "RightHand", hcon[:, 1])]
    for label, limbs, (x1p, x1m) in (("legs", legs, (xpos, xmat)), ("arms, HANDZ=2", arms, kin(f"{hands}.qref"))):
        skate, pop, side, turn, far = 0.0, -1.0, 1.0, 0.0, 0
        for up, lo, en, c in limbs:
            iu, il, ie = canon.index(up), canon.index(lo), canon.index(en)
            mask = planted([(up, lo, en, c)], t)

            def perp(x):
                u = unit(x[:, ie] - x[:, iu])
                v = x[:, il] - x[:, iu]
                return v - u * np.sum(v * u, -1, keepdims=True)

            p1, p0 = perp(x1p), perp(x0p)
            bent = mask & (np.linalg.norm(p1, axis=-1) > 0.01) & (np.linalg.norm(p0, axis=-1) > 0.01)  # straight: no side
            side = min(side, np.sum(unit(p1) * unit(p0), -1)[bent].min(initial=1))
            turn = max(turn, np.abs(x1m[:, ie] - x0m[:, ie]).max())
            reach = np.linalg.norm(x0p[0, il] - x0p[0, iu]) + np.linalg.norm(x0p[0, ie] - x0p[0, il])
            for s0, e0 in runs(c):
                goal = x0p[s0:e0, ie].mean(0)
                ok_reach = np.linalg.norm(goal - x0p[s0:e0, iu], axis=-1) < reach - 1e-4  # frames the solve can hold
                far += int((~ok_reach).sum())
                both = ok_reach[1:] & ok_reach[:-1]
                if both.any():
                    skate = max(skate, np.linalg.norm(np.diff(x1p[s0:e0, ie], axis=0), axis=-1)[both].max())
                for g in [*range(max(1, s0 - BLEND), s0 + 1), *range(e0, min(t, e0 + BLEND + 1))]:
                    d0 = np.linalg.norm(x0p[g, ie] - x0p[g - 1, ie])
                    gap = max(np.linalg.norm(goal - x0p[g, ie]), np.linalg.norm(goal - x0p[g - 1, ie]))
                    pop = max(pop, np.linalg.norm(x1p[g, ie] - x1p[g - 1, ie]) - (2 * d0 + 2 * gap / (BLEND + 1) + 1e-4))
        held = planted(limbs + legs, t)
        ok = skate < 1e-3 and pop <= 0 and side > 0 and turn <= TOL and np.abs(x1m - x0m)[~held].max(initial=0) <= TOL
        print(f"{'ok  ' if ok else 'FAIL'} planting {label}: {sum(len(runs(c)) for *_, c in limbs)} runs, {held.sum()} frames held, "
              f"end moves {skate * 1000:.4f} mm/frame in runs where reachable ({far} frames out of reach), worst pop margin {pop * 1000:.3f} mm, bend side cos {side:.3g} > 0, "
              f"end orientation {turn:.2g}, untouched frames {np.abs(x1m - x0m)[~held].max(initial=0):.2g}")
        if not ok:
            fails.append(f"planting {label}")

    variant = WORK / f"{Path(clip).stem}_variant"
    subprocess.run([sys.executable, "-I", str(ROOT / "tools/bvh_variant.py"), clip, f"{variant}.bvh"], check=True)
    subprocess.run([binary, "clip", f"{variant}.bvh", str(variant)], check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, SCALE="0.01"))
    bar("round trip: Mixamo-style variant .qref vs original", np.abs(read_qref(f"{variant}.qref")[1] - qref).max())

    if fails:
        print(f"FAIL {len(fails)}/12 bvh checks: {', '.join(fails)}")
        sys.exit(1)
    print(f"PASS bvh {Path(clip).name}: {t} frames x {nb} canonical bodies, twin retarget and MuJoCo within {TOL}, round trip, stands")


if __name__ == "__main__":
    main()
