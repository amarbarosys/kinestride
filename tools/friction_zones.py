# Friction-loss rows per zone (jar <= -R floss / quadratic / jar >= R floss) from the final jar = J qacc - aref after mj_forward:
# MuJoCo efc_state is stale on friction rows, so the check.sh frictionloss and tendon_friction coverage numbers come from here.
# usage: .venv/bin/python tools/friction_zones.py dof|tendon MODEL.xml ...
import sys, mujoco, numpy as np
from kinestride.model import nearby
def zones(path, kind):
    m = mujoco.MjModel.from_xml_path(path); m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    q, v = nearby(m, 64); d = mujoco.MjData(m); z = [0, 0, 0]
    for n in range(64):
        d.qpos[:], d.qvel[:] = q[n], v[n]; mujoco.mj_forward(m, d)
        J = d.efc_J[:d.nefc * m.nv].reshape(d.nefc, m.nv) if not mujoco.mj_isSparse(m) else None
        if J is None:
            J = np.zeros((d.nefc, m.nv)); mujoco.mju_sparse2dense(J, d.efc_J, d.efc_J_rownnz, d.efc_J_rowadr, d.efc_J_colind)
        jar = J @ d.qacc - d.efc_aref[:d.nefc]
        for r in range(d.nefc):
            if d.efc_type[r] == kind:
                Rf = d.efc_R[r] * d.efc_frictionloss[r]
                z[0 if jar[r] <= -Rf else (2 if jar[r] >= Rf else 1)] += 1
    return z
if __name__ == '__main__':
    k = {'dof': mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF, 'tendon': mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON}[sys.argv[1]]
    for p in sys.argv[2:]:
        print(p, 'neg/quad/pos', zones(p, k))
