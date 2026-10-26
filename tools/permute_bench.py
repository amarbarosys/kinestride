"""Bench input in another lane order, for bench/LANE_UNION_PREREG.md.
usage: tools/permute_bench.py make MJCF SRC OUT random|sorted   OUT gets the columns of SRC's qpos/qvel/ctrl.bin in that
         order (random: seed 0; sorted: by MuJoCo contact pair set), perm.npy (OUT column j = SRC column perm[j]) and
         symlinks to SRC's step, mi.bin, mf.bin, sizes.txt
       tools/permute_bench.py check SRC OUT   OUT/smooth.bin un-permuted must equal SRC/smooth.bin byte for byte
Exit 1 with FAIL on any difference."""
import os, sys
import numpy as np


def cols(path, n):
    return np.fromfile(path, dtype=np.float32).reshape(-1, n)


def make(mjcf, src, out, order):
    sys.path.insert(0, '.')
    from kinestride.model import load
    m = load(mjcf)
    n = np.fromfile(f'{src}/qpos.bin', dtype=np.float32).size // m.nq
    if order == 'random':
        perm = np.random.default_rng(0).permutation(n)
    elif order == 'sorted':
        import mujoco
        d = mujoco.MjData(m)
        q, v = cols(f'{src}/qpos.bin', n), cols(f'{src}/qvel.bin', n)
        key = []
        for i in range(n):
            d.qpos[:] = q[:, i]; d.qvel[:] = v[:, i]
            mujoco.mj_forward(m, d)
            key.append(tuple(sorted({(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in d.contact[:d.ncon]})))
        perm = np.array(sorted(range(n), key=lambda i: key[i]))
    else:
        sys.exit(f'FAIL order: {order}')
    os.makedirs(out, exist_ok=True)
    for f in ('qpos.bin', 'qvel.bin', 'ctrl.bin'):
        cols(f'{src}/{f}', n)[:, perm].copy().tofile(f'{out}/{f}')
    np.save(f'{out}/perm.npy', perm)
    for f in ('step', 'mi.bin', 'mf.bin', 'sizes.txt'):
        dst = f'{out}/{f}'
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(f'{src}/{f}'), dst)
    print(f'{out}: {n} animals, {order}')


def check(src, out):
    perm = np.load(f'{out}/perm.npy')
    a = cols(f'{src}/smooth.bin', len(perm))
    b = cols(f'{out}/smooth.bin', len(perm))
    bad = int((a[:, perm].view(np.uint32) != b.view(np.uint32)).any(axis=0).sum())
    if bad:
        sys.exit(f'FAIL check {out}: {bad} of {len(perm)} animals differ')
    print(f'PASS {out}: {a.shape[0]} rows x {len(perm)} animals byte-identical after un-permute')


if sys.argv[1] == 'make':
    make(*sys.argv[2:6])
elif sys.argv[1] == 'check':
    check(*sys.argv[2:4])
else:
    sys.exit(f'FAIL mode: {sys.argv[1]}')
