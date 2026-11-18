# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved.
"""MuJoCo as the oracle for a port of its own functions: random states, reference outputs, per-field compare.

A port (GPU engine, another language) is checked one MuJoCo function at a time: draw random states,
let MuJoCo run the function in float64, run the port on the same states, compare every output field
relative to the field's own scale (max |value|: a field of 1 mg masses in grams gets the same
relative bar as positions). Random states, not rest: at rest velocity and orientation terms are zero
and hide their bugs.

  states(m, count, seed)          (count, nq) qpos: free joints anywhere within 5 model units at a random
                                  unit quaternion, ball joints random, hinges and slides inside their range
  velocities(m, count, seed)      (count, nv) qvel uniform in +-1 per dof
  reference(m, fn, fields, qpos, qvel=None, ctrl=None)  {field: (count, ...) array} after mujoco.<fn>(m, d) per state; fn may
                                  be a comma list run in order (a stage that needs earlier stages)
  compare(pairs, rel)             pairs {field: (want, got)}; max |got - want| / max |want| per field;
                                  prints PASS/FAIL per field, returns failed names

CLI: mj-oracle ref MJCF FUNC FIELD[,FIELD...] COUNT OUT.npz [SEED]   (writes qpos and every field)
     mj-oracle compare REF.npz GOT.npz [REL]                       (exit 1 on any failed field)
"""
import os
import sys

try:
    import mujoco
except ImportError:  # the shim runs python3; MJ_PYTHON names an interpreter that has mujoco
    if os.environ.get("MJ_PYTHON") and os.path.realpath(sys.executable) != os.path.realpath(os.environ["MJ_PYTHON"]):
        os.execv(os.environ["MJ_PYTHON"], [os.environ["MJ_PYTHON"], os.path.abspath(__file__), *sys.argv[1:]])
    sys.exit("FAIL mj-oracle: no mujoco in this python; set MJ_PYTHON to one that has it")
import numpy as np


def states(m, count, seed=0):
    rng, q = np.random.default_rng(seed), np.tile(m.qpos0, (count, 1))
    for j in range(m.njnt):
        a, t = m.jnt_qposadr[j], m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            q[:, a:a + 3] = rng.uniform(-5, 5, (count, 3))
            a += 3
        if t in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            r = rng.normal(size=(count, 4))
            q[:, a:a + 4] = r / np.linalg.norm(r, axis=1, keepdims=True)
        else:
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
            q[:, a] = rng.uniform(lo, hi, count)
    return q


def velocities(m, count, seed=1, scale=1.0):
    """(count, nv) qvel uniform in +-scale (model units per second; rad/s for rotational dofs)."""
    return np.random.default_rng(seed).uniform(-scale, scale, (count, m.nv))


def reference(m, fn, fields, qpos, qvel=None, ctrl=None):
    d, out = mujoco.MjData(m), {f: [] for f in fields}
    calls = [getattr(mujoco, name) for name in fn.split(',')]  # 'mj_kinematics,mj_comPos,mj_crb' runs each in order
    for n, row in enumerate(qpos):
        d.qpos[:] = row
        d.qvel[:] = 0 if qvel is None else qvel[n]
        d.ctrl[:] = 0 if ctrl is None else ctrl[n]
        for call in calls:
            call(m, d)
        for f in fields:
            out[f].append(np.array(getattr(d, f)))
    return {f: np.array(v) for f, v in out.items()}


def compare(pairs, rel=1e-5):
    bad = []
    for f, (want, got) in pairs.items():
        want, got = np.asarray(want, np.float64), np.asarray(got, np.float64)
        if want.shape != got.shape:
            print(f'FAIL {f:12s} shape {got.shape}, reference {want.shape}')
            bad.append(f)
            continue
        scale = float(np.abs(want).max()) if want.size else 0.
        e = float(np.abs(got - want).max()) / scale if scale > 0 else float(np.abs(got).max(initial=0.))
        print(f'{"PASS" if e <= rel else "FAIL"} {f:12s} max rel err {e:.2e}')
        bad += [] if e <= rel else [f]
    return bad


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'ref':
        m = mujoco.MjModel.from_xml_path(a[0])
        q = states(m, int(a[3]), int(a[5]) if len(a) > 5 else 0)
        np.savez(a[4], qpos=q, **reference(m, a[1], a[2].split(','), q))
    elif cmd == 'compare':
        ref, got = np.load(a[0]), np.load(a[1])
        bad = compare({f: (ref[f], got[f]) for f in got.files if f != 'qpos'}, float(a[2]) if len(a) > 2 else 1e-5)
        if bad:
            print(f'FAIL {len(bad)}/{len(got.files)} fields: {" ".join(bad)}')
            sys.exit(1)
    else:
        sys.exit(__doc__)
