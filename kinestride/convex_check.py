"""Narrowphase parity gate of convex.mojo (GJK + EPA + multicontact) against MuJoCo 3.9 mj_collision on the fly MJCF.

Cases: a pair's two geoms at random relative poses (centres within a ball of their summed bounding radii, so a mix of
separated, shallow, touching and deep), MuJoCo's contact for exactly that pair as the oracle. Compared: contact
count, dist, pos, normal. The CPU driver convex_test.mojo is built float64 (parity with MuJoCo's doubles, bar 1e-9)
or float32 (the engine; bar measured, stated by the caller).

usage: python -m kinestride.convex_check KIND N [f64|f32] [BAR] [SEED]   KIND: ellipsoid | box (vs mesh, the fly), or a primitive pair of
the flight twin (FLIGHT_PAIRS, model.with_auto_pairs): capsule-cylinder, capsule-ellipsoid, ellipsoid-ellipsoid, ellipsoid-cylinder, sphere-ellipsoid
(GJK + EPA, one contact in the driver), capsule-capsule, sphere-capsule, sphere-cylinder (analytic)
"""
import subprocess, sys, os
import numpy as np
import mujoco
from pathlib import Path

BODY = str(__import__('pathlib').Path(__file__).resolve().parents[1] / '.work/assets/fly/body.xml')
FLIGHT_PAIRS = os.environ.get('FLIGHT_PAIRS', '.work/fc/flight_pairs.xml')
NAMES = {'sphere': mujoco.mjtGeom.mjGEOM_SPHERE, 'capsule': mujoco.mjtGeom.mjGEOM_CAPSULE, 'ellipsoid': mujoco.mjtGeom.mjGEOM_ELLIPSOID,
         'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER}
ANALYTIC = {'capsule-capsule': 5, 'sphere-capsule': 6, 'sphere-cylinder': 7}
CW, OW = 40, 36
G = mujoco.mjtGeom
# float32 bars (measured: `floor` = MuJoCo f64 vs MuJoCo on float32-rounded poses gives dist 2e-7, pos 3e-4, normal 2e-3 max)
F64 = dict(frac=0.002, max=2e-5)  # fraction of values above the bar, and the worst (EPA tolerance 1e-6 scaled by the lever arm)
F32 = dict(dist_max=3e-4, pos_bar=1e-3, normal_bar=4e-3, frac=0.005, shallow=1e-4, count_frac=0.02)


def hull_tables(m, meshes):
    """Hull vertices (geom frame), first edges (absolute into edges), edge lists, per-hull offsets: model.py feet()'s layout."""
    verts, first, edges, meta, index = [], [], [], [], {}
    for mesh in meshes:
        ga = m.mesh_graphadr[mesh]
        nv = m.mesh_graph[ga]
        edgeadr = m.mesh_graph[ga + 2:ga + 2 + nv]
        gid = m.mesh_graph[ga + 2 + nv:ga + 2 + 2 * nv]
        local = m.mesh_graph[ga + 2 + 2 * nv:]
        local = local[:edgeadr.max() + list(local[edgeadr.max():]).index(-1) + 1]
        index[mesh] = len(meta)
        meta.append((len(verts) // 3, len(first), len(edges)))
        first += list(len(edges) + edgeadr)
        edges += list(local)
        verts += list(m.mesh_vert[m.mesh_vertadr[mesh] + gid].ravel())
    return (np.array(verts, np.float32), np.array(first, np.int32), np.array(edges, np.int32),
            np.array(meta, np.int32), index)


def degenerate(m, g1, g2, row):
    """One geom's centre inside the other: the penetration direction is tolerance-arbitrary (EPA stops on any of many equally deep
    faces; MuJoCo's own answer depends on its iteration order), so such cases are counted and left out of the error statistics."""
    for a, b, pa, pb, mb in ((g1, g2, row[5:8], row[17:20], row[20:29]), (g2, g1, row[17:20], row[5:8], row[8:17])):
        p = mb.reshape(3, 3).T @ (pa - pb)
        sz, ty = m.geom_size[b], m.geom_type[b]
        if ty == G.mjGEOM_SPHERE and np.linalg.norm(p) < sz[0]:
            return True
        if ty == G.mjGEOM_ELLIPSOID and ((p / sz) ** 2).sum() < 1:
            return True
        if ty == G.mjGEOM_CAPSULE and np.linalg.norm(p - np.array([0, 0, np.clip(p[2], -sz[1], sz[1])])) < sz[0]:
            return True
        if ty == G.mjGEOM_CYLINDER and abs(p[2]) < sz[1] and np.hypot(p[0], p[1]) < sz[0]:
            return True
    return False


def randrot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat


def run(kind, n, prec='f64', bar=1e-9, seed=0, mutant=None, verbose=True):
    from kinestride.model import pair_geoms, SHAPE_KIND
    prim = '-' in kind
    m = mujoco.MjModel.from_xml_path(FLIGHT_PAIRS if prim else BODY)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if prim:
        ta, tb = (NAMES[x] for x in kind.split('-'))
        sel = [i for i in range(m.npair) if tuple(m.geom_type[g] for g in pair_geoms(m, i)) == (ta, tb)]
    else:
        want = G.mjGEOM_BOX if kind == 'box' else G.mjGEOM_ELLIPSOID
        sel = [i for i in range(m.npair) if m.geom_type[m.pair_geom1[i]] == want and m.geom_type[m.pair_geom2[i]] == G.mjGEOM_MESH]
    if not sel:
        print(f'FAIL convex {kind}: no such pairs'); return False
    meshes = sorted({int(m.geom_dataid[m.pair_geom2[i]]) for i in sel if m.geom_type[m.pair_geom2[i]] == G.mjGEOM_MESH})
    hv, hf, he, hm, hidx = hull_tables(m, meshes)
    rng = np.random.default_rng(seed)
    work = Path('.work/convex') / f'{kind}-{prec}'
    work.mkdir(parents=True, exist_ok=True)
    cases = np.zeros((n, CW))
    pairs = []
    for c in range(n):
        i = sel[rng.integers(len(sel))]
        g1, g2 = pair_geoms(m, i)
        pairs.append((i, g1, g2))
        r = m.geom_rbound[g1] + m.geom_rbound[g2]
        dirn = rng.normal(size=3)
        dirn /= np.linalg.norm(dirn)
        pos2 = d.geom_xpos[g2].copy()
        mat1, mat2 = randrot(rng), randrot(rng)
        if kind == 'box':  # the mesh centre inside the box grown by the mesh's radius (a box dwarfs a ball of radius r)
            ext = m.geom_size[g1] + 1.2 * m.geom_rbound[g2]
            pos1 = pos2 - mat1.reshape(3, 3) @ rng.uniform(-ext, ext)
        else:
            pos1 = pos2 + dirn * r * rng.uniform(0.0, 0.9)
        row = cases[c]
        row[0] = SHAPE_KIND[G(m.geom_type[g1])] if prim else (0 if kind == 'ellipsoid' else 2)
        row[1] = SHAPE_KIND[G(m.geom_type[g2])] if prim else 1
        row[2:5] = m.geom_size[g1]
        row[33:36] = m.geom_size[g2]
        row[36] = 1  # mjc_Convex's perturbation search (multiccd on), as the engine
        row[37] = ANALYTIC.get(kind, 0)
        row[5:8] = pos1
        row[8:17] = mat1
        row[17:20] = pos2
        row[20:29] = mat2
        row[29] = hidx[int(m.geom_dataid[g2])] if not prim else 0
        row[30] = m.pair_margin[i] + m.pair_gap[i]
        row[31] = m.opt.ccd_tolerance
        row[32] = min(m.geom_rbound[g1], m.geom_rbound[g2])
    cases.tofile(work / 'cases.f64')
    np.save(work / 'pairs.npy', np.array(pairs))
    hv.tofile(work / 'hverts.f32'); hf.tofile(work / 'hfirst.i32'); he.tofile(work / 'hedges.i32'); hm.tofile(work / 'hmeta.i32')
    exe = work / 'ct'
    flags = ['-DCCD64=1'] if prec == 'f64' else []
    p = subprocess.run([os.path.expanduser('~/iTools/bin/mojo-cached'), 'build', *flags, 'kinestride/convex_test.mojo', '-o', str(exe)], capture_output=True, text=True)
    if p.returncode:
        print('FAIL build convex_test:', p.stderr[-2000:])
        return False
    p = subprocess.run([str(exe), str(work) + '/'], capture_output=True, text=True)
    if p.returncode:
        print('FAIL run convex_test:', p.stdout[-500:], p.stderr[-500:])
        return False
    out = np.fromfile(work / 'out.f64').reshape(n, OW)
    bad = {'ncon': 0, 'dist': 0, 'pos': 0, 'normal': 0}
    worst = {'dist': 0.0, 'pos': 0.0, 'normal': 0.0}
    contact_cases = 0
    errs = []
    counts = {}
    first_bad = []
    ndeg = 0
    for c in range(n):
        i, g1, g2 = pairs[c]
        cs = oracle(m, d, g1, g2, cases[c])
        contact_cases += bool(cs)
        mine = int(out[c, 0])
        if prim and cs and degenerate(m, g1, g2, cases[c]):
            ndeg += 1
            continue
        counts[(len(cs), mine)] = counts.get((len(cs), mine), 0) + 1
        if len(cs) != mine:
            bad['ncon'] += 1
            if len(first_bad) < 5:
                first_bad.append((c, 'ncon', len(cs), mine))
            continue
        for k, (cd, cp, cn) in enumerate(cs):
            ed = abs(out[c, 1 + 7 * k] - cd)
            ep = np.abs(out[c, 2 + 7 * k:5 + 7 * k] - cp).max()
            en = np.abs(out[c, 5 + 7 * k:8 + 7 * k] - cn).max()
            errs.append((ed, ep, en, cases[c][30] - cd))
            worst['dist'], worst['pos'], worst['normal'] = max(worst['dist'], ed), max(worst['pos'], ep), max(worst['normal'], en)
            for key, e in (('dist', ed), ('pos', ep), ('normal', en)):
                if e > bar:
                    bad[key] += 1
                    if len(first_bad) < 5:
                        first_bad.append((c, key, k, e))
    deep = np.array([e[:3] for e in errs if e[3] > F32['shallow']]) if errs else np.zeros((1, 3))  # force-significant contacts
    errs = [e[:3] for e in errs]
    if prec == 'f64' and prim and kind not in ANALYTIC:  # the sphere/capsule core trick: GJK on the cores stops at its 1e-6 tolerance on both sides, the
        # inflated witness points inherit it (measured worst: dist 1e-9, pos 7e-7, normal 1.5e-4 on 1500 cases per kind)
        a = np.array(errs) if errs else np.zeros((1, 3))
        bad = {'ncon': bad['ncon'], 'dist': int(a[:, 0].max() > 1e-8), 'pos': int(a[:, 1].max() > 1e-5), 'normal': int(a[:, 2].max() > 1e-3)}
        print(f'  f64 (cored GJK): max dist {a[:, 0].max():.1e} pos {a[:, 1].max():.1e} normal {a[:, 2].max():.1e} (bars 1e-8, 1e-5, 1e-3)')
    elif prec == 'f64':  # exact except path flips: a comparison decided by 1e-16 changes where EPA stops, within its 1e-6 tolerance
        a = np.array(errs) if errs else np.zeros((1, 3))
        over = float((a > bar).mean())
        bad = {'ncon': bad['ncon'], 'dist': int(over > F64['frac'] or a[:, 0].max() > F64['max']),
               'pos': int(a[:, 1].max() > F64['max']), 'normal': int(a[:, 2].max() > F64['max'])}
        print(f'  f64: {100 * over:.3f} % of dist/pos/normal values above {bar:g} (bar {100 * F64["frac"]:.1f} %), max {a.max():.1e} (bar {F64["max"]:g})')
    counts_bad = bad['ncon']
    if prec == 'f32':  # per-quantity bars instead of one bar on the worst case
        a = deep
        p99 = np.percentile(a, 99, axis=0)
        bad = {'ncon': int(bad['ncon'] > F32['count_frac'] * n), 'dist': int(np.array(errs)[:, 0].max() > F32['dist_max']),
               'pos': int((a[:, 1] > F32['pos_bar']).mean() > F32['frac']), 'normal': int((a[:, 2] > F32['normal_bar']).mean() > F32['frac'])}
        print(f"  f32 (contacts deeper than {F32['shallow']:g} inside the margin: {len(a)} of {len(errs)}): p99 pos {p99[1]:.1e} normal {p99[2]:.1e}, max pos {a[:, 1].max():.1e} normal {a[:, 2].max():.1e}; "
              f"above the bars (pos {F32['pos_bar']:g}, normal {F32['normal_bar']:g}): {100 * (a[:, 1] > F32['pos_bar']).mean():.3f} % / {100 * (a[:, 2] > F32['normal_bar']).mean():.3f} % (bar {100 * F32['frac']:.1f} %); "
              f"count mismatches {counts_bad} of {n} cases (bar {100 * F32['count_frac']:.0f} %)")
    ok = sum(bad.values()) == 0 and contact_cases > 0
    if ndeg:
        print(f'  {ndeg} degenerate cases (a centre inside the other geom) left out')
    if verbose:
        print(f"{'PASS' if ok else 'FAIL'} convex {kind} {prec}: {n} cases, {contact_cases} with a contact, bar {bar:g}; "
              f"worst dist {worst['dist']:.2e} pos {worst['pos']:.2e} normal {worst['normal']:.2e}; mismatches {bad}; contacts oracle/engine {dict(sorted(counts.items()))}")
        if errs:
            q = np.percentile(np.array(errs), [50, 99], axis=0)
            print(f'  p50 / p99 of dist, pos, normal: {q[0][0]:.1e} {q[0][1]:.1e} {q[0][2]:.1e} / {q[1][0]:.1e} {q[1][1]:.1e} {q[1][2]:.1e}')
        if first_bad and not ok:
            print('  first:', first_bad)
    return ok


def oracle(m, d, g1, g2, row):
    """MuJoCo's contact for the pair at the case's poses: (dist, pos, normal) or None."""
    d.geom_xpos[g1] = row[5:8]; d.geom_xmat[g1] = row[8:17]
    d.geom_xpos[g2] = row[17:20]; d.geom_xmat[g2] = row[20:29]
    mujoco.mj_collision(m, d)
    cs = [k for k in range(d.ncon) if d.contact[k].geom1 == g1 and d.contact[k].geom2 == g2]
    return [(d.contact[k].dist, d.contact[k].pos.copy(), d.contact[k].frame[:3].copy()) for k in cs]


def floor(kind, n):
    """The float32 floor of the algorithm itself: MuJoCo (double) on the cases of the last run() against MuJoCo on the same
    cases with every pose rounded to float32 (what any float32 port receives). Prints the worst dist, pos, normal difference."""
    m = mujoco.MjModel.from_xml_path(BODY)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    work = Path('.work/convex') / f'{kind}-f64'
    cases = np.fromfile(work / 'cases.f64').reshape(-1, CW)[:n]
    pairs = np.load(work / 'pairs.npy')
    worst, flips = np.zeros(3), 0
    for row, (i, g1, g2) in zip(cases, pairs):
        a = oracle(m, d, g1, g2, row)
        r32 = row.copy()
        r32[2:29] = row[2:29].astype(np.float32)
        b = oracle(m, d, g1, g2, r32)
        if len(a) != len(b):
            flips += 1
        else:
            for x, y in zip(a, b):
                worst = np.maximum(worst, [abs(x[0] - y[0]), np.abs(x[1] - y[1]).max(), np.abs(x[2] - y[2]).max()])
    print(f'floor {kind}: MuJoCo f64 vs MuJoCo on float32-rounded poses, {len(cases)} cases: count flips {flips}, '
          f'worst dist {worst[0]:.2e} pos {worst[1]:.2e} normal {worst[2]:.2e}')


def arena_poses(count, prec='f32'):
    """The narrowphase on the arena states of `check.sh arena` (standing fly moved into walls and stones, model.arena): every
    box-mesh and ellipsoid-mesh pair at its real poses (rounded to float32 as the engine receives them), against MuJoCo's
    contacts of the same state. Prints, over the contact-bearing pairs, count mismatches and the dist / pos / normal errors."""
    from kinestride.model import standing, arena
    m = mujoco.MjModel.from_xml_path(BODY)
    d = mujoco.MjData(m)
    q, v = arena(m, *standing(m, count))
    sel = [i for i in range(m.npair) if m.geom_type[m.pair_geom1[i]] in (G.mjGEOM_BOX, G.mjGEOM_ELLIPSOID)]
    meshes = sorted({int(m.geom_dataid[m.pair_geom2[i]]) for i in sel})
    hv, hf, he, hm, hidx = hull_tables(m, meshes)
    rows, keys, want = [], [], []
    for n in range(count):
        d.qpos[:], d.qvel[:] = q[n], v[n]
        mujoco.mj_forward(m, d)
        got = {(c.geom1, c.geom2): [] for c in d.contact[:d.ncon]}
        for c in d.contact[:d.ncon]:
            got[(c.geom1, c.geom2)].append((c.dist, c.pos.copy(), c.frame[:3].copy()))
        for i in sel:
            g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
            row = np.zeros(CW)
            row[0] = 2 if m.geom_type[g1] == G.mjGEOM_BOX else 0
            row[1] = 1
            row[2:5] = m.geom_size[g1]
            row[5:8] = d.geom_xpos[g1]; row[8:17] = d.geom_xmat[g1]
            row[17:20] = d.geom_xpos[g2]; row[20:29] = d.geom_xmat[g2]
            row[29] = hidx[int(m.geom_dataid[g2])]
            row[30] = m.pair_margin[i] + m.pair_gap[i]
            row[31] = m.opt.ccd_tolerance
            row[32] = min(m.geom_rbound[g1], m.geom_rbound[g2])
            row[2:29] = row[2:29].astype(np.float32)
            rows.append(row); keys.append((n, i, g1)); want.append(got.get((g1, g2), []))
    work = Path('.work/convex') / f'arena-{prec}'
    work.mkdir(parents=True, exist_ok=True)
    np.array(rows).tofile(work / 'cases.f64')
    hv.tofile(work / 'hverts.f32'); hf.tofile(work / 'hfirst.i32'); he.tofile(work / 'hedges.i32'); hm.tofile(work / 'hmeta.i32')
    exe = work / 'ct'
    p = subprocess.run([os.path.expanduser('~/iTools/bin/mojo-cached'), 'build', *(['-DCCD64=1'] if prec == 'f64' else []), 'kinestride/convex_test.mojo', '-o', str(exe)], capture_output=True, text=True)
    if p.returncode or subprocess.run([str(exe), str(work) + '/']).returncode:
        print('FAIL build or run', p.stderr[-800:])
        return False
    out = np.fromfile(work / 'out.f64').reshape(len(rows), OW)
    nbad, errs, bystate = 0, [], {}
    for r, (n, i, g1) in enumerate(keys):
        w = want[r]
        mine = int(out[r, 0])
        if len(w) != mine:
            nbad += 1
            bystate.setdefault(n, []).append((i, len(w), mine))
            continue
        for k, (cd, cp, cn) in enumerate(w):
            errs.append((n, abs(out[r, 1 + 7 * k] - cd), np.abs(out[r, 2 + 7 * k:5 + 7 * k] - cp).max(), np.abs(out[r, 5 + 7 * k:8 + 7 * k] - cn).max()))
    a = np.array(errs)
    print(f'arena poses {prec}: {len(rows)} pair-states, {len(a)} contacts compared, count mismatches {nbad} {dict(list(bystate.items())[:6])}')
    print(f'  max dist {a[:, 1].max():.1e} pos {a[:, 2].max():.1e} normal {a[:, 3].max():.1e}; p99 pos {np.percentile(a[:, 2], 99):.1e} normal {np.percentile(a[:, 3], 99):.1e}')
    for n in (32, 16, 4, 53, 12, 9, 5):
        b = a[a[:, 0] == n]
        print(f'  state {n}: {len(b)} contacts, max dist {b[:, 1].max():.1e} pos {b[:, 2].max():.1e} normal {b[:, 3].max():.1e}') if len(b) else None
    worst = np.argsort(-a[:, 3])[:5]
    print('  worst normal (state, dist err, pos err, normal err):', [tuple(float(f'{x:.2g}') for x in a[k]) for k in worst])
    return nbad == 0


def flight_poses(count, prec='f32', seed=0):
    """Every contact MuJoCo reports on `count` near-rest states of the flight twin (model.nearby), as a narrowphase case at the geoms'
    world poses: the per-contact dist/pos/normal error of the engine's kinds on the poses the whole-engine gate meets."""
    from kinestride.model import load, nearby, pair_geoms, SHAPE_KIND
    m = load(FLIGHT_PAIRS)
    d = mujoco.MjData(m)
    q, v = nearby(m, count, seed)
    rows, want, kinds = [], [], []
    for n in range(count):
        d.qpos[:], d.qvel[:] = q[n], v[n]
        mujoco.mj_forward(m, d)
        seen = set()
        for c in d.contact[:d.ncon]:
            i = [j for j in range(m.npair) if set(pair_geoms(m, j)) == {int(c.geom1), int(c.geom2)}][0]
            if i in seen:  # the perturbation search's extra contacts of a pair: the first contact is the one compared
                continue
            seen.add(i)
            g1, g2 = pair_geoms(m, i)
            row = np.zeros(CW)
            row[0], row[1] = SHAPE_KIND[G(m.geom_type[g1])], SHAPE_KIND[G(m.geom_type[g2])]
            row[2:5], row[33:36] = m.geom_size[g1], m.geom_size[g2]
            row[5:8], row[8:17] = d.geom_xpos[g1], d.geom_xmat[g1]
            row[17:20], row[20:29] = d.geom_xpos[g2], d.geom_xmat[g2]
            row[30], row[31], row[32] = m.pair_margin[i] + m.pair_gap[i], m.opt.ccd_tolerance, min(m.geom_rbound[g1], m.geom_rbound[g2])
            row[36] = 1
            name = G(m.geom_type[g1]).name[7:].lower() + '-' + G(m.geom_type[g2]).name[7:].lower()
            row[37] = ANALYTIC.get(name, 0)
            rows.append(row); want.append((float(c.dist), c.pos.copy(), c.frame[:3].copy())); kinds.append(name)
    work = Path('.work/convex') / f'flight-{prec}'
    work.mkdir(parents=True, exist_ok=True)
    np.array(rows).tofile(work / 'cases.f64')
    for nm in ('hverts.f32', 'hfirst.i32', 'hedges.i32', 'hmeta.i32'):
        (work / nm).write_bytes(b'')
    exe = work / 'ct'
    p = subprocess.run([os.path.expanduser('~/iTools/bin/mojo-cached'), 'build', *(['-DCCD64=1'] if prec == 'f64' else []), 'kinestride/convex_test.mojo', '-o', str(exe)], capture_output=True, text=True)
    if p.returncode or subprocess.run([str(exe), str(work) + '/']).returncode:
        print('FAIL build/run convex_test:', p.stderr[-1500:]); return False
    out = np.fromfile(work / 'out.f64').reshape(len(rows), OW)
    by = {}
    for k, (row, (cd, cp, cn), name) in enumerate(zip(rows, want, kinds)):
        e = by.setdefault(name, [])
        if int(out[k, 0]) < 1:
            e.append((abs(cd), 1.0, 2.0)); continue
        e.append((abs(out[k, 1] - cd), np.abs(out[k, 2:5] - cp).max(), np.abs(out[k, 5:8] - cn).max()))
    print(f'flight poses {prec}: {len(rows)} contacts on {count} near states')
    for name, e in sorted(by.items()):
        a = np.array(e)
        print(f'  {name:20s} n {len(a):4d}  missed {int((a[:, 1] == 1.0).sum())}  max dist {a[:, 0].max():.1e} pos {a[:, 1].max():.1e} normal {a[:, 2].max():.1e}  p50 dist {np.median(a[:, 0]):.1e}')
    return True




if __name__ == '__main__':
    if sys.argv[1] == 'flight':
        sys.exit(0 if flight_poses(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'f32') else 1)
    if sys.argv[1] == 'arena':
        sys.exit(0 if arena_poses(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'f32') else 1)
    if sys.argv[1] == 'floor':
        floor(sys.argv[2], int(sys.argv[3]))
        sys.exit(0)
    kind = sys.argv[1]
    n = int(sys.argv[2])
    prec = sys.argv[3] if len(sys.argv) > 3 else 'f64'
    bar = float(sys.argv[4]) if len(sys.argv) > 4 else (1e-9 if prec == 'f64' else 1e-4)
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    sys.exit(0 if run(kind, n, prec, bar, seed) else 1)
