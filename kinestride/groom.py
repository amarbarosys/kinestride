"""64 fly states where a grooming sphere-sphere pair carries a constraint row (efc_address >= 0).
Base = model.standing frames (actuated rollouts); front legs are then driven by damped-least-squares IK until the
chosen pair's rub sphere sits at a chosen signed distance from its target sphere (-0.03..0.0005 mm, margin 0.001),
then 30 servo steps (ctrl = target joint angles) so qvel is a rollout velocity. The frame after the rollout is kept
if the pair still touches, else the IK pose. check.sh groom feeds them to model.check (env STATES). From the repo root:
  .venv/bin/python kinestride/groom.py OUT.npz
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, '.')
from kinestride.model import load, standing  # noqa: E402

XML = str(__import__('pathlib').Path(__file__).resolve().parents[1] / '.work/assets/fly/body.xml')
OUT = Path(sys.argv[1])
COUNT, STEPS = 64, 30
m = load(XML)
G = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
pairs = [i for i in range(m.npair) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[i]).endswith('_rub')
         and m.geom_type[m.pair_geom1[i]] == m.geom_type[m.pair_geom2[i]] == mujoco.mjtGeom.mjGEOM_SPHERE]
assert len(pairs) == 7, pairs
pname = lambda i: f'{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom1[i])}|{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[i])}'


def legdofs(side):  # qpos == dof addresses for hinges here: coxa yaw .. tarsus5
    js = [j for j in range(m.njnt) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(f'fly/c_thorax-{side}f_coxa') or
          f'{side}f_' in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split('-')[0] + '_']
    return [j for j in js if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


LEG = {'lf': legdofs('l'), 'rf': legdofs('r')}
assert all(len(v) == 11 for v in LEG.values()), {k: len(v) for k, v in LEG.items()}
qa = {k: np.array([m.jnt_qposadr[j] for j in v]) for k, v in LEG.items()}
da = {k: np.array([m.jnt_dofadr[j] for j in v]) for k, v in LEG.items()}


def ik(d, rng, side, rub, target, dist):
    """move leg `side` so |c_rub - c_target| - r_rub - r_target == dist"""
    jp = np.zeros((3, m.nv))
    want = m.geom_size[rub, 0] + m.geom_size[target, 0] + dist
    d.qpos[qa[side]] += rng.uniform(-0.3, 0.3, 11)
    for _ in range(300):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        c, t = d.geom_xpos[rub], d.geom_xpos[target]
        gap = np.linalg.norm(c - t)
        goal = t + (c - t) / max(gap, 1e-9) * want
        err = goal - c
        if np.linalg.norm(err) < 1e-5:
            break
        mujoco.mj_jacGeom(m, d, jp, None, rub)
        J = jp[:, da[side]]
        d.qpos[qa[side]] += np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-2 * np.eye(3), err), -0.2, 0.2)


def touching(d):
    n = {}
    for c in d.contact[:d.ncon]:
        if c.efc_address >= 0 and c.geom[0] in P1 and (c.geom[0], c.geom[1]) in PAIRG:
            n[PAIRG[(c.geom[0], c.geom[1])]] = n.get(PAIRG[(c.geom[0], c.geom[1])], 0) + 1
    return n


PAIRG = {(m.pair_geom1[i], m.pair_geom2[i]): i for i in pairs}
P1 = {g for g, _ in PAIRG}
q0, v0 = standing(m, COUNT)
rng = np.random.default_rng(7)
Q, V, U, WHICH, KIND = [], [], [], [], []
d = mujoco.MjData(m)
tries = 0
while len(Q) < COUNT:
    tries += 1
    assert tries < 40 * COUNT, f'only {len(Q)} states after {tries} tries'
    pi = pairs[len(Q) % 7]
    t, r = m.pair_geom1[pi], m.pair_geom2[pi]
    n = len(Q)
    d.qpos[:], d.qvel[:] = q0[n % len(q0)], v0[n % len(q0)]
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    dist = rng.uniform(-0.03, 0.0005)
    side = 'lf' if 'lf_rub' in pname(pi).split('|')[1] else 'rf'
    ik(d, rng, side, r, t, dist)
    if pname(pi).endswith('lf_rub|fly/rf_rub'):  # both rubs: other leg to the same sphere pair
        ik(d, rng, 'rf', G('fly/rf_rub'), G('fly/lf_rub'), dist)
    qik = d.qpos.copy()
    d.ctrl[:] = 0
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            d.ctrl[i] = qik[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        else:
            d.ctrl[i] = 1.0
    mujoco.mj_forward(m, d)
    if pi not in touching(d):
        continue
    state = (qik.copy(), d.qvel.copy(), 'ik')
    for _ in range(STEPS):
        mujoco.mj_step(m, d)
    if pi in touching(d) and np.isfinite(d.qpos).all():
        state = (d.qpos.copy(), d.qvel.copy(), 'rollout')
    d.qpos[:], d.qvel[:] = state[0], state[1]
    mujoco.mj_forward(m, d)
    Q.append(state[0]); V.append(state[1]); U.append(d.ctrl.copy()); WHICH.append(pi); KIND.append(state[2])
Q, V, U = map(np.array, (Q, V, U))
np.savez(OUT, qpos=Q, qvel=V, ctrl=U, pair=np.array(WHICH), kind=np.array(KIND))
# per-pair report: states and contact rows over the saved set
cnt = {p: 0 for p in pairs}; states = {p: 0 for p in pairs}
for q, v, u in zip(Q, V, U):
    d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, u
    mujoco.mj_forward(m, d)
    for p, c in touching(d).items():
        cnt[p] += c; states[p] += 1
print(f'saved {OUT}: {len(Q)} states, {int((np.array(KIND) == "rollout").sum())} after rollout, {int((np.array(KIND) == "ik").sum())} IK pose, qpos {Q.shape} qvel {V.shape} ctrl {U.shape}')
for p in pairs:
    print(f'{pname(p):50s} pair {p}: {states[p]:3d} states with a row, {cnt[p]:3d} rows')
