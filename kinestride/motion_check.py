"""Gate for motion_metrics.mojo: a float64 numpy twin (rfft, not the Mojo DFT) on random clips, plus known answers.

Random clips: the simulated quaternions get random sign flips (q and -q are one rotation). Known answers: identical
clips give 0 everywhere; a constant offset d gives pos_err |d| and npss 0 (DC only); every orientation turned by
theta gives rot_err theta; a foot sliding at v with vertical bob in contact gives foot_skate |v_xy|.

usage: python -m kinestride.motion_check BINARY
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work/motion"
TOL = 1e-9


def write(path: Path, hz, foot, pos, quat, contact) -> None:
    t, nb, _ = pos.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<3if", t, nb, len(foot), hz))
        f.write(np.asarray(foot, "<i4").tobytes())
        for a in (pos, quat, contact):
            f.write(np.asarray(a, "<f4").tobytes())


def f32(a):
    return np.asarray(a, np.float32).astype(np.float64)


def twin(hz, foot, rp, rq, rc, sp, sq, sc) -> list[float]:
    rp, rq, rc, sp, sq, sc = map(f32, (rp, rq, rc, sp, sq, sc))
    hz = float(np.float32(hz))
    t, nb, _ = rp.shape
    pos_err = np.linalg.norm(sp - rp, axis=-1).mean()
    w = (rq * sq).sum(-1)
    v = rq[..., :1] * sq[..., 1:] - sq[..., :1] * rq[..., 1:] - np.cross(rq[..., 1:], sq[..., 1:])
    rot_err = (2 * np.arctan2(np.linalg.norm(v, axis=-1), np.abs(w))).mean()
    nk = t // 2
    npss = 0.0
    if nk:
        pr = np.abs(np.fft.rfft(rp.reshape(t, -1), axis=0)[1:nk + 1]) ** 2
        ps = np.abs(np.fft.rfft(sp.reshape(t, -1), axis=0)[1:nk + 1]) ** 2
        tr, ts = pr.sum(0), ps.sum(0)
        cr = np.cumsum(np.divide(pr, tr, out=np.zeros_like(pr), where=tr > 0), 0)
        cs = np.cumsum(np.divide(ps, ts, out=np.zeros_like(ps), where=ts > 0), 0)
        emd = np.abs(cr - cs).sum(0)
        npss = float((tr * emd).sum() / tr.sum()) if tr.sum() > 0 else 0.0
    speeds = []
    for f, b in enumerate(foot):
        on = (sc[:-1, f] > 0.5) & (sc[1:, f] > 0.5)
        d = np.linalg.norm(sp[1:, b, :2] - sp[:-1, b, :2], axis=-1) * hz
        speeds.extend(d[on])
    foot_skate = float(np.mean(speeds)) if speeds else 0.0
    contact_err = float(((rc > 0.5) != (sc > 0.5)).mean()) if len(foot) else 0.0
    return [float(pos_err), float(rot_err), npss, foot_skate, contact_err]


def run(binary: str, name: str, hz, foot, rp, rq, rc, sp, sq, sc) -> list[float]:
    r, s, o = WORK / f"{name}.ref.mclip", WORK / f"{name}.sim.mclip", WORK / f"{name}.out"
    write(r, hz, foot, rp, rq, rc)
    write(s, hz, foot, sp, sq, sc)
    subprocess.run([binary, str(r), str(s), str(o)], check=True, stdout=subprocess.DEVNULL)
    return list(struct.unpack("<5d", o.read_bytes()))


def unit(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qmul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdims=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate([w, v], -1)


def random_case(rng, t, nb, nf):
    foot = rng.choice(nb, nf, replace=False) if nf else np.zeros(0, int)
    rp = np.cumsum(rng.normal(0, 0.01, (t, nb, 3)), 0) + rng.normal(0, 1, (1, nb, 3))
    sp = rp + rng.normal(0, 0.02, (t, nb, 3)) + np.sin(np.arange(t) / 7.0)[:, None, None] * 0.05
    rq = unit(rng.normal(size=(t, nb, 4)))
    sq = qmul(rq, unit(np.concatenate([np.ones((t, nb, 1)) * 8, rng.normal(size=(t, nb, 3))], -1)))
    sq *= rng.choice([-1.0, 1.0], (t, nb, 1))
    rc = rng.random((t, nf)) < 0.6
    sc = np.where(rng.random((t, nf)) < 0.8, rc, ~rc)
    return 120.0, foot, rp, rq, rc, sp, sq, sc


def main() -> None:
    binary = sys.argv[1]
    WORK.mkdir(parents=True, exist_ok=True)
    names = ["pos_err", "rot_err", "npss", "foot_skate", "contact_err"]
    fails = []

    def compare(case, got, want):
        for n, g, w in zip(names, got, want):
            if not abs(g - w) <= TOL * max(1.0, abs(w)):
                fails.append(case)
                print(f"FAIL motion {case} {n}: mojo {g!r} twin {w!r}")

    rng = np.random.default_rng(7)
    shapes = [(240, 20, 2), (97, 5, 1), (2, 3, 1), (1, 4, 2), (64, 7, 0)]
    for t, nb, nf in shapes:
        case = random_case(rng, t, nb, nf)
        compare(f"random {t}x{nb}x{nf}", run(binary, f"r{t}", *case), twin(*case))

    hz, foot, rp, rq, rc, _, _, _ = random_case(rng, 120, 6, 2)
    known = []
    known.append(("identical", (hz, foot, rp, rq, rc, rp, rq, rc), [0, 0, 0, None, 0]))
    d = np.array([0.03, -0.04, 0.12])
    known.append(("offset", (hz, foot, rp, rq, rc, rp + d, rq, rc), [np.linalg.norm(d), 0, 0, None, 0]))
    th = 0.3
    turn = np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])
    known.append(("turn", (hz, foot, rp, rq, rc, rp, qmul(turn, rq), rc), [0, th, 0, None, 0]))
    slide = rp.copy()
    v = np.array([0.2, -0.1])
    slide[:, foot[0], :2] = slide[0, foot[0], :2] + np.arange(120)[:, None] * v / hz
    slide[:, foot[0], 2] = 0.05 * np.sin(np.arange(120) / 3.0)
    on = np.zeros_like(rc)
    on[:, 0] = True
    known.append(("slide", (hz, foot, slide, rq, on, slide, rq, on), [0, 0, 0, np.linalg.norm(v), 0]))
    for name, case, want in known:
        got = run(binary, name, *case)
        compare(name, got, twin(*case))
        for n, g, w in zip(names, got, want):
            if w is not None and not abs(g - w) <= 1e-6 * max(1.0, abs(w)):  # float32 inputs: 1e-6, not TOL
                fails.append(name)
                print(f"FAIL motion {name} {n}: mojo {g!r} expected {w!r}")

    n = len(shapes) + len(known)
    if fails:
        print(f"FAIL {len(set(fails))}/{n} motion cases: {' '.join(sorted(set(fails)))}")
        sys.exit(1)
    print(f"PASS motion: {len(shapes)} random clips within {TOL} of the numpy twin, {len(known)} known answers")


if __name__ == "__main__":
    main()
