"""auto_pairs vs MuJoCo's own dynamic pairs: on random states of a model, the explicit-pair twin (model.with_auto_pairs) must produce the
same contact set (geom pairs) and the same constraint forces. Usage: auto_pairs_check.py MJCF OUT_XML [COUNT]. Exit 1 on any difference."""
import sys
import mujoco
import numpy as np
from kinestride.model import load, states, velocities, with_auto_pairs

mjcf, out = sys.argv[1], sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 64
t = load(with_auto_pairs(mjcf, out))
m = load(with_auto_pairs(mjcf, out.replace('.xml', '_dynamic.xml'), explicit=False))  # the same writer: the MjSpec XML round trip rounds poses by ~3e-7, enough to move stiff contact forces by 1e-2
q, v = states(m, count), velocities(m, count, scale=float(__import__('os').environ.get('VSCALE', 100)))
dm, dt = mujoco.MjData(m), mujoco.MjData(t)
bad, ncon, worst = 0, [], 0.0
for n in range(count):
    for d, mm in ((dm, m), (dt, t)):
        d.qpos[:], d.qvel[:] = q[n], v[n]
        mujoco.mj_forward(mm, d)
    a = sorted((int(c.geom1), int(c.geom2)) for c in dm.contact[:dm.ncon])
    b = sorted((int(c.geom1), int(c.geom2)) for c in dt.contact[:dt.ncon])
    e = float(np.abs(dm.qfrc_constraint - dt.qfrc_constraint).max() / max(np.abs(dm.qfrc_constraint).max(), 1e-30))
    worst = max(worst, e)
    ncon.append(dm.ncon)
    if a != b or e > 1e-9:
        bad += 1
        if bad <= 3:
            print(f'state {n}: dynamic {len(a)} contacts, explicit {len(b)}, missing {sorted(set(a) - set(b))[:4]}, extra {sorted(set(b) - set(a))[:4]}, qfrc_constraint rel diff {e:.1e}')
print(f'{"PASS" if not bad else "FAIL"} auto_pairs: {t.npair} explicit pairs reproduce the dynamic contacts on {count - bad}/{count} states '
      f'(contacts per state mean {np.mean(ncon):.1f}, max {max(ncon)}; qfrc_constraint worst rel diff {worst:.1e})')
sys.exit(1 if bad else 0)
