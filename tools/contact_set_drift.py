"""usage: tools/contact_set_drift.py MJCF BENCH_DIR COUNT (CPU). Share of bench states whose contact pair set changes after k MuJoCo steps (ctrl from the bench's ctrl.bin)."""
import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from kinestride.model import load
m = load(sys.argv[1]); d = sys.argv[2]; n = int(sys.argv[3])
q = np.fromfile(f'{d}/qpos.bin', dtype=np.float32).reshape(m.nq, -1).T.astype(np.float64)
v = np.fromfile(f'{d}/qvel.bin', dtype=np.float32).reshape(m.nv, -1).T.astype(np.float64)
u = np.fromfile(f'{d}/ctrl.bin', dtype=np.float32).reshape(m.nu, -1).T.astype(np.float64)
dd = mujoco.MjData(m)
pset = lambda: frozenset((min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in dd.contact[:dd.ncon])
ch = {1: 0, 4: 0, 10: 0}
for i in np.linspace(0, q.shape[0] - 1, n).astype(int):
    dd.qpos[:] = q[i]; dd.qvel[:] = v[i]; dd.ctrl[:] = u[i]; mujoco.mj_forward(m, dd); s0 = pset()
    for k in range(1, 11):
        mujoco.mj_step(m, dd); mujoco.mj_forward(m, dd)  # mj_step leaves the pre-step contacts
        if k in ch: ch[k] += pset() != s0
print(f"{n} states: pair set changed after 1 step {ch[1] / n:.3%}, 4 steps {ch[4] / n:.3%}, 10 steps {ch[10] / n:.3%}")
# lines per wave per active-pair step on the 1-step-later sets, for orders keyed on: nothing, last step (stale), this step
S0, S1, S10 = [], [], []
for i in np.linspace(0, q.shape[0] - 1, n).astype(int):
    dd.qpos[:] = q[i]; dd.qvel[:] = v[i]; dd.ctrl[:] = u[i]; mujoco.mj_forward(m, dd); S0.append(tuple(sorted(pset())))
    for k in range(1, 11):
        mujoco.mj_step(m, dd); mujoco.mj_forward(m, dd)
        if k == 1: S1.append(tuple(sorted(pset())))
    S10.append(tuple(sorted(pset())))
def amp(order, sets):
    lines = ideal = 0
    for w in range(0, len(order) - 31, 32):
        L = [sets[i] for i in order[w:w + 32]]
        for ip in range(max(len(s) for s in L)):
            lines += len({s[ip] for s in L if len(s) > ip}); ideal += 1
    return lines / ideal
b = np.arange(n); r = np.random.default_rng(0).permutation(b)
print(f"step+1 sets: subsample order {amp(b, S1):.2f}, random {amp(r, S1):.2f}, keyed 1 step stale {amp(sorted(r, key=lambda i: S0[i]), S1):.2f}, oracle {amp(sorted(r, key=lambda i: S1[i]), S1):.2f}")
print(f"step+10 sets: keyed 10 steps stale {amp(sorted(r, key=lambda i: S0[i]), S10):.2f}, random {amp(r, S10):.2f}")
