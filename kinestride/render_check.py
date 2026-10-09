"""Renderer check (docs/RENDERER.md step 5): our engine's poses drawn by our renderer against MuJoCo's own
renderer, same state and camera, three views. The fly's silhouette must overlap at IoU >= 0.98 in every view.

The pose comes from the engine itself (smooth.mojo STAGE 1 on a standing state from model.standing), so the
check covers the path a user meets: engine buffer -> render.mojo. MuJoCo's side is its segmentation render
(mesh geoms only, all geom groups on). Side-by-side images go to OUT/view*.ppm for a look.

usage: env MUJOCO_GL=egl python -m kinestride.render_check [OUT_DIR]   (under gpu-wait)
"""
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from kinestride.model import ENGINE, export, meshes, standing

MJCF = str(__import__('pathlib').Path(__file__).resolve().parents[1] / '.work/assets/fly/body.xml')
W, H = 960, 540
VIEWS = ((90, -20), (0, -45), (225, -10))  # azimuth, elevation in degrees (MuJoCo free camera)
DIST = 6.0  # mm
IOU = 0.98
MOJO = str(Path.home() / 'iTools/bin/mojo-cached')


def build(src, out, flags):
    subprocess.run([MOJO, 'build', *flags, str(ENGINE / src), '-o', str(out)], check=True, capture_output=True)


def main(out):
    out = Path(out)
    m = export(MJCF, out)
    meshes(MJCF, out, world=[])  # the arena's box and ellipsoid geoms drawn too, as MuJoCo draws them (alpha: animal parts only)
    q, v = standing(m, 2)
    q, v = q[:1], v[:1]
    q.astype(np.float32).T.copy().tofile(out / 'qpos.bin')
    v.astype(np.float32).T.copy().tofile(out / 'qvel.bin')
    np.zeros((m.nu, 1), np.float32).tofile(out / 'ctrl.bin')
    build('smooth.mojo', out / 'pose1', (out / 'sizes.txt').read_text().split() + ['-DFLIES=1', '-DSTAGE=1'])
    subprocess.run([str(out / 'pose1'), str(out) + '/'], check=True, capture_output=True)
    prows = (out / 'smooth.bin').stat().st_size // 4
    build('render.mojo', out / 'render1', [f'-DW={W}', f'-DH={H}', f'-DNBODY={m.nbody}', '-DFLIES=1', f'-DPROWS={prows}',
                                           *(out / 'rsizes.txt').read_text().split()])
    d = mujoco.MjData(m)
    d.qpos[:] = q[0]
    mujoco.mj_forward(m, d)
    lookat = d.xpos[1].copy()
    rend = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    mesh = m.geom_type == mujoco.mjtGeom.mjGEOM_MESH
    fails = []
    for n, (az, el) in enumerate(VIEWS):
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, DIST, az, el
        a, e = np.radians(az), np.radians(el)
        eye = lookat - DIST * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        rgba = out / f'view{n}.rgba'
        r = subprocess.run([str(out / 'render1'), str(out) + '/', str(out / 'smooth.bin'), str(rgba), '20',
                            *map(str, eye), *map(str, lookat), str(m.vis.global_.fovy), '0'], check=True, capture_output=True, text=True)
        ours = np.fromfile(rgba, np.uint8).reshape(H, W, 4)
        rend.enable_segmentation_rendering()
        rend.update_scene(d, cam, scene_option=opt)
        seg = rend.render()
        theirs = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & mesh[np.clip(seg[..., 0], 0, m.ngeom - 1)]
        rend.disable_segmentation_rendering()
        rend.update_scene(d, cam, scene_option=opt)
        img = rend.render()
        mine = ours[..., 3] == 255
        iou = (mine & theirs).sum() / max((mine | theirs).sum(), 1)
        side = np.concatenate([ours[..., :3], img], axis=1)
        (out / f'view{n}.ppm').write_bytes(f'P6\n{2 * W} {H}\n255\n'.encode() + side.tobytes())
        ok = iou >= IOU
        print(f'{"PASS" if ok else "FAIL"} view {n} (az {az}, el {el}): silhouette IoU {iou:.4f}, '
              f'{mine.sum()} vs {theirs.sum()} px; {r.stdout.strip()}')
        fails += [] if ok else [f'view{n}']
    if fails:
        sys.exit(f'FAIL {len(fails)}/{len(VIEWS)} views: {" ".join(fails)}')
    print(f'PASS render_check: {len(VIEWS)} views at IoU >= {IOU}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.work/render_check')
