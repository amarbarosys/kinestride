"""UNION masks (docs/plans/UNION_WALK.md) against the lists they mirror: for every animal, bit i of the OUPM rows is set
exactly when pair i is in OPA, and bit s of the OUSM rows exactly when slot s is in OACT.
usage: tools/union_check.py DIR   (DIR holds dump.bin and union.txt of a DUMP=1 UNION=1 build). Exit 1 on any difference."""
import sys
import numpy as np

d = sys.argv[1]
nrow, F, oact, opa, oupm, ousm, nslot, ncp, union = map(int, open(f'{d}/union.txt').read().split())
if not union:
    sys.exit('FAIL union.txt: built without UNION=1')
r = np.fromfile(f'{d}/dump.bin', dtype=np.float32).reshape(nrow, F)


def bits(base, n):
    w = r[base:base + (n + 23) // 24].astype(np.int64)
    return np.array([(w[i // 24] >> (i % 24)) & 1 for i in range(n)], dtype=bool)


def listed(base, n):
    out = np.zeros((n, F), dtype=bool)
    for f in range(F):
        out[r[base + 1:base + 1 + int(r[base, f]), f].astype(int), f] = True
    return out


bad = 0
for name, mbase, lbase, n in (('pairs', oupm, opa, ncp), ('slots', ousm, oact, nslot)):
    m, l = bits(mbase, n), listed(lbase, n)
    diff = int((m != l).any(axis=0).sum())
    bad += diff
    print(f"{name}: {n} ids, {int(l.sum())} listed over {F} animals, mask differs on {diff} animals")
if bad:
    sys.exit(f'FAIL union masks differ on {bad} animal lists')
print('PASS union masks equal the lists')
