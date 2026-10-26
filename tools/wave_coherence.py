"""usage: tools/wave_coherence.py MJCF BENCH_DIR COUNT (CPU). Lane coherence of the bench states: per 32-animal wave, how many distinct contact pairs the lanes visit at the same
loop index (the engine walks its active-pair list OPA in ascending order, one lane per animal, so lanes on different
pairs read different J / jar / D rows). Orders: bench (as stored), random (a crowd), sorted by pair set.
CPU only: MuJoCo float64 contacts on the bench's float32 qpos."""
import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from kinestride.model import load
m = load(sys.argv[1]); d = sys.argv[2]; n = int(sys.argv[3])
q = np.fromfile(f'{d}/qpos.bin', dtype=np.float32).reshape(m.nq, -1).T.astype(np.float64)
v = np.fromfile(f'{d}/qvel.bin', dtype=np.float32).reshape(m.nv, -1).T.astype(np.float64)
N = q.shape[0]; idx = np.linspace(0, N - 1, n).astype(int) if n < N else np.arange(N)
dd = mujoco.MjData(m); sets, ncon = [], []
for i in idx:
    dd.qpos[:] = q[i]; dd.qvel[:] = v[i]; mujoco.mj_forward(m, dd)
    s = sorted({(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in dd.contact[:dd.ncon]})
    sets.append(tuple(s)); ncon.append(dd.ncon)
def union(order):  # wave-uniform walk: every pair any lane has, ascending; one line per row per pair
    lines = cur = 0
    for w in range(0, len(order) - 31, 32):
        L = [sets[i] for i in order[w:w + 32]]
        lines += len(set().union(*L))
        cur += sum(len({s[ip] for s in L if len(s) > ip}) for ip in range(max(len(s) for s in L)))
    return lines / cur
def amp(order):
    lines, ideal, distinct = 0, 0, []
    for w in range(0, len(order) - 31, 32):
        L = [sets[i] for i in order[w:w + 32]]
        distinct.append(len(set(L)))
        for ip in range(max(len(s) for s in L)):
            ps = [s[ip] for s in L if len(s) > ip]
            lines += len(set(ps)); ideal += 1
    return lines / max(ideal, 1), np.mean(distinct)
rng = np.random.default_rng(0); base = np.arange(len(sets))
print(f"states {len(sets)} of {N}; contacts mean {np.mean(ncon):.2f}; pairs per state mean {np.mean([len(s) for s in sets]):.2f}; distinct pair sets {len(set(sets))}")
stale = lambda i: sets[i - 1] if i > 0 else sets[i]  # key from the state 2 ms (20 fly steps) earlier in the rollout
rnd = rng.permutation(base)
for name, o in (('bench', base), ('random', rnd), ('sorted', sorted(base, key=lambda i: sets[i])), ('stale', sorted(rnd, key=stale))):
    a, ds = amp(o); print(f"{name:7s} lines per wave per active-pair step {a:.2f} (ideal 1); distinct pair sets per wave {ds:.2f}; union walk / current lines {union(o):.2f}")
