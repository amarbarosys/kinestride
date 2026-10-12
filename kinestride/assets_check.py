"""assets.mojo against its oracle assets.py: every builder, same vertices (1e-12), same faces in the same order.

usage: python -m kinestride.assets_check   (CPU; builds .work/assets/assets with mojo-cached)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from kinestride import assets

OUT = Path('.work/assets-mojo')


def read(path):
    b = path.read_bytes()
    nv, nf = np.frombuffer(b[:8], '<u4')
    v = np.frombuffer(b[8:8 + 24 * nv], '<f8').reshape(nv, 3)
    return v, np.frombuffer(b[8 + 24 * nv:], '<i4').reshape(nf, 3)


def write(path, v, f):
    v, f = np.asarray(v, '<f8'), np.asarray(f, '<i4')
    path.write_bytes(np.array([len(v), len(f)], '<u4').tobytes() + v.tobytes() + f.tobytes())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    binary = OUT / 'assets'
    subprocess.run([str(Path.home() / 'iTools/bin/mojo-cached'), 'build', str(Path(__file__).with_name('assets.mojo')), '-o', str(binary)], check=True)
    cases = [(f'sphere {n}', ['sphere', str(n)], assets.sphere(n)) for n in range(4)]
    for box, size in ((1, (0.5, 1.5, 0.25)), (0, (1.2, 1.2, 0.5)), (1, (20.0, 0.5, 3.0))):
        cases.append((f'primitive {box} {size}', ['primitive', str(box), *map(repr, size)], assets.primitive(box, np.array(size))))
    long = assets.primitive(1, np.array([6.0, 0.3, 1.0]))
    write(OUT / 'long.bin', *long)
    for edge in (0.4, 0.1, 2.5):
        cases.append((f'split long box {edge}', ['split', str(OUT / 'long.bin'), None, repr(edge)], assets.split(*long, edge)))
    bad = []
    for i, (name, args, (wv, wf)) in enumerate(cases):
        out = OUT / f'case{i}.bin'
        cmd = [str(binary), *[str(out) if a is None else a for a in args]] + ([] if None in args else [str(out)])
        subprocess.run(cmd, check=True)
        v, f = read(out)
        ok = v.shape == np.shape(wv) and f.shape == np.shape(wf) and np.abs(v - wv).max() <= 1e-12 and (f == wf).all()
        print(f'{"PASS" if ok else "FAIL"} {name}: {len(v)} vertices, {len(f)} faces')
        bad += [] if ok else [name]
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(cases)} {", ".join(bad)}')
    print(f'PASS assets {len(cases)}/{len(cases)}')


if __name__ == '__main__':
    main()
