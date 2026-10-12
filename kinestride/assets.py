"""Mesh builders for the renderer's world parts (model.meshes world=...): an icosphere, MuJoCo box and ellipsoid geoms as
triangles, and a splitter that keeps every triangle side short (render.raster gives one thread a triangle's whole screen box).
Moved from fly-connectome agent/body_engine/assets.py, which keeps the scene (walls, grass, flowers, markers) and imports these.
Oracle only since 2026-10-12: model.meshes runs assets.mojo; assets_check.py compares the two.
"""
import numpy as np


def sphere(n=2):
    """unit icosphere, n subdivisions: (vertices, faces), outward winding."""
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(n):
        mid, g = {}, []
        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                mid[k] = len(v) - 1
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            g += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = g
    return np.array(v), np.array(f)


def primitive(box, size):
    """(vertices, faces) of a MuJoCo box (half sizes) or ellipsoid (radii) in its geom frame."""
    if box:
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * size[:3]
        f = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
        return v, f
    v, f = sphere(2)
    return v * size[:3], f


def split(v, f, edge=0.4):
    """triangles cut in half along their longest side until every side is under `edge` mm: render.raster gives one
    thread a triangle's whole screen box, so a long blade or wall near the camera held a thread for thousands of pixels."""
    v, out = [np.asarray(p, float) for p in v], []
    todo = [tuple(t) for t in f]
    while todo:
        a, b, c = todo.pop()
        sides = [(np.linalg.norm(v[a] - v[b]), a, b, c), (np.linalg.norm(v[b] - v[c]), b, c, a), (np.linalg.norm(v[c] - v[a]), c, a, b)]
        n, i, j, k = max(sides)
        if n < edge:
            out.append((a, b, c))
            continue
        v.append((v[i] + v[j]) / 2)
        m = len(v) - 1
        todo += [(i, m, k), (m, j, k)]
    return np.array(v), np.array(out)
