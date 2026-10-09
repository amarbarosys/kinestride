"""States for check.sh implicit's midpoint_contact part (models/midpoint_contact.xml): body p and body q each at a random distance from
their static sphere (w1 at the origin, w2 at x 0.3) so that about half the states have a sphere pair inside its margin (live contact: no midpoint
for that body), body r far away on its own (always midpoint). usage: python -m kinestride.midpoint_states OUT.npz [count]"""
import sys

import numpy as np


def make(count=64, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros((count, 21))
    v = np.zeros((count, 18))

    def quat(n):
        a = rng.normal(size=(n, 4))
        return a / np.linalg.norm(a, axis=1, keepdims=True)
    for b, off in enumerate((0.0, 0.3, 2.0)):
        q[:, 7 * b + 3:7 * b + 7] = quat(count)
        v[:, 6 * b:6 * b + 3] = rng.uniform(-0.5, 0.5, (count, 3))
        v[:, 6 * b + 3:6 * b + 6] = rng.uniform(-6, 6, (count, 3))
    for b, w in ((0, (0.0, 0.0, 0.5)), (1, (0.3, 0.0, 0.5))):
        d = rng.uniform(0.06, 0.16, count)  # sphere radii 0.05 + 0.05 + margin 0.01: in contact below 0.11
        u = rng.normal(size=(count, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        q[:, 7 * b:7 * b + 3] = np.array(w) + d[:, None] * u
    q[:, 14:17] = [2.0, 0.0, 0.5] + rng.uniform(-0.1, 0.1, (count, 3))
    return q, v


if __name__ == '__main__':
    q, v = make(*(int(a) for a in sys.argv[2:3]))
    np.savez(sys.argv[1], qpos=q, qvel=v)
