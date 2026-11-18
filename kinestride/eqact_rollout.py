"""Runtime eq_active (MuJoCo d->eq_active written by the host between steps), the way a host meets it: one K=1 launch per step,
the host rewriting each animal's eq_active rows of ctrl.bin (-DEQACT=1) between launches, the engine's own state carried from
launch to launch. Every launch is compared with MuJoCo mj_step from the same float32 state and the same eq_active (fresh MjData,
no warmstart, as the engine's process), so the error never compounds, at the equality stage's bars: qpos_next within REL x
max(1, |value|), qacc_euler = (qvel' - qvel) / h within EQ_EULER_BAR of its max (float32 cancellation in D (J qacc - aref), model.check). The schedule starts at eq_active0 and flips a random third of
every animal's equalities every SWITCH launches; it must switch each equality on and off, and an inactive-at-start one on.
usage: python -m kinestride.eqact_rollout MJCF OUT [ANIMALS] [LAUNCHES]   (CPU=1: the host engine)"""
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from kinestride.model import ENGINE, EQ_EULER_BAR, REL, STAGES, export, load, nearby

SWITCH = 3  # launches between the host's flips


def main(mjcf, out, count=64, launches=12):
    out = Path(out)
    m = export(mjcf, out)
    s = load(mjcf)
    s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
    if m.nu or m.nmocap or m.npair:
        raise SystemExit(f'FAIL eqact_rollout: {mjcf} has actuators, mocap bodies or pairs; the rollout writes only eq_active rows')
    binary = out / f'rollout_{count}'
    flags = (out / 'sizes.txt').read_text().split()
    subprocess.run([str(Path(__file__).resolve().parents[1] / 'tools/bin/mojo-cached'), 'build', *flags, '-DEQACT=1', f'-DFLIES={count}', f"-DSTAGE={STAGES['equality'][0]}",
                    '-DK=1', f"-DCPU={os.environ.get('CPU', 0)}", str(ENGINE / 'smooth.mojo'), '-o', str(binary)], check=True)
    if os.environ.get('BUILD_ONLY'):
        print(f'built {binary}')
        return
    rng = np.random.default_rng(7)
    q, v = nearby(m, count)
    q, v = q.astype(np.float32), v.astype(np.float32)
    act = np.tile(m.eq_active0.astype(np.float32), (count, 1))
    seen = np.zeros((2, m.neq), bool)  # [off, on] per equality over the run
    worst = np.zeros(2)  # qpos_next / REL, qacc_euler / EQ_EULER_BAR
    h = m.opt.timestep
    for n in range(launches):
        if n and n % SWITCH == 0:
            flip = rng.random((count, m.neq)) < 1 / 3
            act[flip] = 1 - act[flip]
        seen[0] |= (act == 0).any(0)
        seen[1] |= (act == 1).any(0)
        q.T.copy().tofile(out / 'qpos.bin')
        v.T.copy().tofile(out / 'qvel.bin')
        act.T.copy().tofile(out / 'ctrl.bin')
        subprocess.run([str(binary), str(out) + '/'], check=True, stdout=subprocess.DEVNULL)
        st = np.fromfile(out / 'state.bin', np.float32).reshape(-1, count).T
        gq, gv = st[:, :m.nq], st[:, m.nq:m.nq + m.nv]
        wq, wv = np.zeros((count, m.nq)), np.zeros((count, m.nv))
        for a in range(count):
            d = mujoco.MjData(s)
            d.qpos[:], d.qvel[:], d.eq_active[:] = q[a], v[a], act[a]
            mujoco.mj_step(s, d)
            wq[a], wv[a] = d.qpos, d.qvel
        ga, wa = (gv - v) / h, (wv - v) / h
        e = np.array([float((np.abs(gq - wq) / np.maximum(1, np.abs(wq))).max()), float(np.abs(ga - wa).max() / np.abs(wa).max())])
        worst = np.maximum(worst, e / [REL, EQ_EULER_BAR])
        print(f'launch {n}: {int(act.sum())}/{act.size} equalities on, qpos_next rel err {e[0]:.2e}, qacc_euler {e[1]:.2e} of its max')
        q, v = gq, gv  # the engine's own state goes on, as a host's loop does
    off0 = np.flatnonzero(m.eq_active0 == 0)
    if not seen.all() or not len(off0):
        raise SystemExit(f'FAIL eqact_rollout: schedule does not switch every equality both ways ({seen.sum(1)} of {m.neq}) or no equality starts inactive')
    if worst.max() > 1:
        raise SystemExit(f'FAIL eqact_rollout: {launches} launches x {count} animals, qpos_next {worst[0] * REL:.2e} (bar {REL:g}), qacc_euler {worst[1] * EQ_EULER_BAR:.2e} (bar {EQ_EULER_BAR:g})')
    print(f'PASS eqact_rollout: {launches} launches x {count} animals, {m.neq} equalities ({len(off0)} inactive at start) switched both ways, '
          f'qpos_next {worst[0] * REL:.2e} (bar {REL:g}), qacc_euler {worst[1] * EQ_EULER_BAR:.2e} (bar {EQ_EULER_BAR:g})')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:5]))
