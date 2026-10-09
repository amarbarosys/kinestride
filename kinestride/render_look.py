"""Look at and time the renderer (docs/RENDERER.md): N standing flies, one close-up (N = 1) or a GX x GX grid seen
from above at an angle, W x H. Writes OUT/look-N.ppm and prints the renderer's own ms per frame (mean of 200).

usage: python -m kinestride.render_look OUT N [W H] [-DFLAG=..]   (under gpu-wait; N a square for the grid)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from kinestride.model import export, meshes, standing
from kinestride.render_check import MJCF, build

out, n = Path(sys.argv[1]), int(sys.argv[2])
w, h = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[3].isdigit() else (1920, 1080)
extra = [a for a in sys.argv[3:] if a.startswith('-D')]
out.mkdir(parents=True, exist_ok=True)
m = export(MJCF, out)
meshes(MJCF, out)
q, v = standing(m, n)
q.astype(np.float32).T.copy().tofile(out / 'qpos.bin')
v.astype(np.float32).T.copy().tofile(out / 'qvel.bin')
np.zeros((m.nu, n), np.float32).tofile(out / 'ctrl.bin')
build('smooth.mojo', out / f'pose{n}', (out / 'sizes.txt').read_text().split() + [f'-DFLIES={n}', '-DSTAGE=1'])
subprocess.run([str(out / f'pose{n}'), str(out) + '/'], check=True, capture_output=True)
prows = (out / 'smooth.bin').stat().st_size // 4 // n
gx = int(round(n ** 0.5))
build('render.mojo', out / f'render{n}', [f'-DW={w}', f'-DH={h}', f'-DNBODY={m.nbody}', f'-DFLIES={n}', f'-DPROWS={prows}',
                                         f'-DGX={gx}', *extra, *(out / 'rsizes.txt').read_text().split()])
if n == 1:
    root = np.fromfile(out / 'smooth.bin', np.float32)[:3]
    eye, target, spacing = root + [4.2, -3.6, 2.4], root + [0, 0, -0.3], 0
else:
    c = (gx - 1) * 5.0 / 2
    eye, target, spacing = [c - 22, c - 30, 26], [c, c, 0], 5.0
r = subprocess.run([str(out / f'render{n}'), str(out) + '/', str(out / 'smooth.bin'), str(out / f'look-{n}.ppm'), '200',
                    *map(str, eye), *map(str, target), '45', str(spacing)], check=True, capture_output=True, text=True)
print(f'look-{n}.ppm: {r.stdout.strip()}')
