"""Engine files for F spiders standing at the origin facing +x, servos holding the stance (docs/plans/BRAIN.md B4).
npc_live places and turns each one onto its NPC's tile. Same steps as fly-connectome live.prepare, without the meshes.
usage: tools/npc_bodies.py DIR F"""
import sys
import mujoco
import numpy as np

sys.path.insert(0, '.')
from kinestride.model import FEET, export, load, standing
from kinestride.models.spider import build

out, n = sys.argv[1], int(sys.argv[2])
mjcf = str(build())
m = load(mjcf)
m.opt.noslip_iterations = 0  # live.prepare: noslip 5 makes the standing body tremble, in MuJoCo too
m = export(m, out, FEET, True)
q, v = standing(load(mjcf), 2)
q, v = q[:1], v[:1]
u = np.where(m.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY, 1.0, q[0][m.jnt_qposadr[np.minimum(m.actuator_trnid[:, 0], m.njnt - 1)]])
np.tile(q.astype(np.float32).T, (1, n)).tofile(f'{out}/qpos.bin')
np.tile(v.astype(np.float32).T, (1, n)).tofile(f'{out}/qvel.bin')
np.tile(u.astype(np.float32).reshape(-1, 1), (1, n)).tofile(f'{out}/ctrl.bin')
print(f'{out}: {n} spiders, nq {m.nq}, nu {m.nu}, root quat {q[0][3:7].round(4).tolist()}')
