"""Per-contact comparison of the engine's solver rows with MuJoCo's, after `DUMP=1 ... model.py check ... contact`:
for every contact slot the engine filled, MuJoCo's constraint rows (D, aref) of the same pair and contact index. Localises a
qacc mismatch to a contact (depth, velocity or impedance) before the solver. Reads DIR/dump.bin, dump.txt, mi.bin, qpos.bin,
qvel.bin written by the DUMP build.

usage: python -m kinestride.row_diag MJCF DIR [N_WORST]   (env CONVEX / HFEXACT as for the check that wrote DIR)
"""
import sys, os
import numpy as np
import mujoco
from pathlib import Path
from kinestride.model import load, pairs, FEET


def main(mjcf, out, worst=10):
    out = Path(out)
    s = load(mjcf)
    convex, hfexact = bool(os.environ.get('CONVEX')), bool(os.environ.get('HFEXACT'))
    on = set(pairs(s, FEET, convex)[0])
    for i in range(s.npair):
        if i not in on:
            s.pair_margin[i] = -1e10
    s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    nrow, F, ovalid, od, oaref, oj, nslot, cph, ncp, ckind, osd = map(int, (out / 'dump.txt').read_text().split())
    dump = np.fromfile(out / 'dump.bin', np.float32).reshape(nrow, F)
    mi = np.fromfile(out / 'mi.bin', np.int32)
    qpos = np.fromfile(out / 'qpos.bin', np.float32).reshape(-1, F)
    qvel = np.fromfile(out / 'qvel.bin', np.float32).reshape(-1, F)
    head = mi[cph:cph + 11 * ncp].reshape(ncp, 11)
    pair_of = {(int(h[0]), int(h[1])): p for p, h in enumerate(head)}
    d = mujoco.MjData(s)
    rows = []
    for n in range(F):
        d.qpos[:], d.qvel[:] = qpos[:, n], qvel[:, n]
        mujoco.mj_forward(s, d)
        seen = {}
        for c in d.contact[:d.ncon]:
            key = (int(c.geom1), int(c.geom2))
            if c.efc_address < 0 or key not in pair_of:
                continue
            k = seen.get(key, 0)
            seen[key] = k + 1
            p = pair_of[key]
            sbase, ns = int(head[p][8]), int(head[p][9])
            if ckind >= 0 and mi[ckind + p] > 0:  # pooled pair: this step's allocation
                sbase, ns = int(dump[osd + 3 * p, n]), int(dump[osd + 3 * p + 1, n])
            if k >= ns:
                rows.append((n, p, k, 'TRUNCATED', 0, 0, 0))
                continue
            slot = sbase + k
            valid = dump[ovalid + slot, n] != 0
            if not valid:
                rows.append((n, p, k, 'ENGINE-MISSING', 0, 0, 0))
                continue
            ed = dump[od + 4 * slot:od + 4 * slot + 4, n]
            ea = dump[oaref + 4 * slot:oaref + 4 * slot + 4, n]
            wd = d.efc_D[c.efc_address:c.efc_address + 4]
            wa = d.efc_aref[c.efc_address:c.efc_address + 4]
            rows.append((n, p, k, 'ok', float(np.abs(ed - wd).max() / np.abs(wd).max()),
                         float(np.abs(ea - wa).max() / max(np.abs(d.efc_aref).max(), 1e-30)), float(c.dist), float(np.abs(wa).max()), float(np.abs(ed - wd).max() / np.abs(d.efc_D).max())))
        for p in range(ncp):  # engine slots valid but MuJoCo has no such contact
            key = (int(head[p][0]), int(head[p][1]))
            sb, ns = int(head[p][8]), int(head[p][9])
            if ckind >= 0 and mi[ckind + p] > 0:
                sb, ns = int(dump[osd + 3 * p, n]), int(dump[osd + 3 * p + 1, n])
            for k in range(seen.get(key, 0), ns):
                if dump[ovalid + sb + k, n] != 0:
                    rows.append((n, p, k, 'ENGINE-EXTRA', 0, 0, 0))
    bad = [r for r in rows if r[3] != 'ok']
    ok = [r for r in rows if r[3] == 'ok']
    print(f'{len(ok)} contacts compared, {len(bad)} missing/extra/truncated: {bad[:8]}')
    for label, pick in (('convex (box, ellipsoid vs mesh)', lambda g: s.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID)),
                        ('hfield', lambda g: s.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD),
                        ('plane and sphere', lambda g: s.geom_type[g] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE))):
        sub = [r for r in ok if pick(int(head[r[1]][0]))]
        if not sub:
            continue
        print(f'{label}: {len(sub)} contacts; worst aref error (of the state\'s largest aref) {max(r[5] for r in sub):.1e}, worst D error (of the state\'s largest D) {max(r[7] for r in sub if len(r) > 7):.1e}')
        for r in sorted(sub, key=lambda r: -r[5])[:worst]:
            g1, g2 = head[r[1]][0], head[r[1]][1]
            print(f'  state {r[0]} pair {r[1]} (geoms {g1},{g2}, {mujoco.mj_id2name(s, mujoco.mjtObj.mjOBJ_GEOM, int(g1))}) contact {r[2]}: aref err {r[5]:.1e} D err {r[7]:.1e} dist {r[6]:.5f} |aref| {r[7-1]:.2e}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
