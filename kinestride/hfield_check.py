"""Narrowphase parity gate of convex.mojo's heightfield contact (mjc_ConvexHField) against MuJoCo 3.9 mj_collision on a bumpy
seeded terrain (model.terrain with octaves > 0: MuJoCo's own hfield geom, not the plane twin).

Cases: a mesh part of the fly at a random pose over the terrain, its centre between 0.4 radii under and 0.8 radii over the
surface; MuJoCo's contacts for the (hfield, mesh) pair are the oracle. Compared: contact count, then per contact (prism order)
dist, pos, normal. float64 build: bar 1e-9 (a few EPA-tolerance flips allowed as in convex_check); float32: the F32 bars.

usage: python -m kinestride.hfield_check N [f64|f32] [SEED]
"""
import os, subprocess, sys
import numpy as np
import mujoco
from pathlib import Path
from kinestride.convex_check import hull_tables, randrot, F64, F32
from kinestride.model import terrain, world_full as full

BODY = str(__import__('pathlib').Path(__file__).resolve().parents[1] / '.work/assets/fly/body.xml')
PARAMS = (0, 0, 0.6, 8.0, 4, 1, 2.0, 0.5)  # slope x y, amplitude mm, wavelength mm, octaves, seed, lacunarity, gain
HW, HO = 20, 1 + 7 * 50
G = mujoco.mjtGeom


def run(n, prec='f64', seed=0, bar=1e-9, verbose=True, func=0):
    path = terrain(BODY, PARAMS)[0]
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    sel = [i for i in range(m.npair) if m.geom_type[m.pair_geom1[i]] == G.mjGEOM_HFIELD]
    g1 = int(m.pair_geom1[sel[0]])
    hid = int(m.geom_dataid[g1])
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    size = m.hfield_size[hid].copy()
    data = m.hfield_data[m.hfield_adr[hid]:m.hfield_adr[hid] + nrow * ncol].astype(np.float32)
    pos1 = d.geom_xpos[g1].copy()
    meshes = sorted({int(m.geom_dataid[m.pair_geom2[i]]) for i in sel})
    hv, hf, he, hm, hidx = hull_tables(m, meshes)
    rng = np.random.default_rng(seed)
    grid = data.reshape(nrow, ncol).astype(np.float64) * size[2] + pos1[2]  # world z of the nodes
    work = Path('.work/convex') / f'hfield-{prec}'
    work.mkdir(parents=True, exist_ok=True)
    cases = np.zeros((n, HW))
    keys = []
    for c in range(n):
        i = sel[rng.integers(len(sel))]
        g2 = int(m.pair_geom2[i])
        x, y = rng.uniform(-30, 30, 2)
        ci = int(round((x + size[0]) / (2 * size[0]) * (ncol - 1)))
        ri = int(round((y + size[1]) / (2 * size[1]) * (nrow - 1)))
        rb = m.geom_rbound[g2]
        pos2 = np.array([x, y, grid[ri, ci] + rng.uniform(-0.4, 0.8) * rb])
        mat2 = randrot(rng)
        keys.append((i, g2, pos2, mat2))
        row = cases[c]
        row[0:3] = pos2 - pos1
        row[3:12] = mat2
        row[12] = hidx[int(m.geom_dataid[g2])]
        row[13] = m.pair_margin[i] + m.pair_gap[i]
        row[14] = m.opt.ccd_tolerance
        row[15] = rb
    cases.tofile(work / 'hcases.f64')
    np.array([nrow, ncol, size[0], size[1], size[2], size[3], pos1[2]]).tofile(work / 'hmeta2.f64')
    data.tofile(work / 'hgrid.f32')
    np.array([*full(PARAMS), pos1[2]], np.float32).tofile(work / 'hparams.f32')
    hv.tofile(work / 'hverts.f32'); hf.tofile(work / 'hfirst.i32'); he.tofile(work / 'hedges.i32'); hm.tofile(work / 'hmeta.i32')
    exe = work / 'ht'
    p = subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools/bin/mojo-cached'), 'build', *(['-DCCD64=1'] if prec == 'f64' else []), *([f'-DHF_FUNC={func}'] if func else []), 'kinestride/hfield_test.mojo', '-o', str(exe)], capture_output=True, text=True)
    if p.returncode or subprocess.run([str(exe), str(work) + '/']).returncode:
        print('FAIL build or run hfield_test', p.stderr[-1500:])
        return False
    out = np.fromfile(work / 'hout.f64').reshape(n, HO)
    errs, counts, nbad, worst_n = [], {}, 0, 0
    for c in range(n):
        i, g2, pos2, mat2 = keys[c]
        d.geom_xpos[g2] = pos2
        d.geom_xmat[g2] = mat2
        mujoco.mj_collision(m, d)
        cs = [k for k in range(d.ncon) if d.contact[k].geom1 == g1 and d.contact[k].geom2 == g2]
        mine = int(out[c, 0])
        counts[(len(cs) > 0, mine > 0)] = counts.get((len(cs) > 0, mine > 0), 0) + 1
        worst_n = max(worst_n, len(cs))
        if len(cs) != mine:
            nbad += 1
            continue
        for k, kc in enumerate(cs):
            con = d.contact[kc]
            errs.append((abs(out[c, 1 + 7 * k] - con.dist), np.abs(out[c, 2 + 7 * k:5 + 7 * k] + pos1 - con.pos).max(),
                         np.abs(out[c, 5 + 7 * k:8 + 7 * k] - con.frame[:3]).max(), cases[c][13] - con.dist))
    a = np.array(errs) if errs else np.zeros((1, 4))
    if prec == 'f64':
        over = float((a[:, :3] > bar).mean())
        ok = nbad == 0 and over <= F64['frac'] and a[:, :3].max() <= F64['max']
        msg = f'{100 * over:.3f} % of values above {bar:g} (bar {100 * F64["frac"]:.1f} %), max {a[:, :3].max():.1e} (bar {F64["max"]:g})'
    else:
        deep = a[a[:, 3] > F32['shallow']][:, :3] if (a[:, 3] > F32['shallow']).any() else a[:, :3]
        pb = float((deep[:, 1] > F32['pos_bar']).mean())
        nb = float((deep[:, 2] > F32['normal_bar']).mean())
        ok = nbad <= F32['count_frac'] * n and a[:, 0].max() <= F32['dist_max'] and pb <= F32['frac'] and nb <= F32['frac']
        msg = (f'max dist {a[:, 0].max():.1e}, above pos {F32["pos_bar"]:g} {100 * pb:.2f} %, above normal {F32["normal_bar"]:g} {100 * nb:.2f} % '
               f'(bar {100 * F32["frac"]:.1f} %), count mismatches {nbad} of {n} (bar {100 * F32["count_frac"]:.0f} %)')
    if verbose:
        print(f"{'PASS' if ok else 'FAIL'} hfield {prec}: {n} cases, {len(errs)} contacts compared, count mismatches {nbad}, "
              f"max contacts per pair {worst_n}, with/without contact oracle/engine {dict(sorted(counts.items()))}; {msg}")
    return ok


if __name__ == '__main__':
    # 4th argument `func`: heights from world.mojo's float32 function (the engine's source) instead of MuJoCo's table; `quant`: the same rounded to MuJoCo's float32 elevation
    sys.exit(0 if run(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'f64', int(sys.argv[3]) if len(sys.argv) > 3 else 0, func={'func': 1, 'quant': 2}.get(sys.argv[4], 0) if len(sys.argv) > 4 else 0) else 1)
