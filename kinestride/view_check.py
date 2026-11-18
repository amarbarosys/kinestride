"""view.mojo (the Viewer loop with no brain) against the engine and the renderer it wraps, byte for byte: after N launches of
one step, the state equals smooth's state after one launch of N steps (K-step parity, check.sh loopk); the first frame equals
render.mojo drawing the start state's pose (smooth STAGE 1, kinematics only) from the same camera; a frame shows the pose
the step began from. smooth's main adds 100 timing launches after the first, so N stays small. A standing fly, servos
holding its pose.

usage: python -m kinestride.view_check build [N]     (CPU: export, states, mesh tables, four binaries; N default 10)
       gpu-wait run --timeout 900 -- python -m kinestride.view_check run [N]
"""
import os, subprocess, sys
from pathlib import Path
import numpy as np
from kinestride.model import export, meshes, standing, FEET, ENGINE

BODY = str(ENGINE.parent / '.work/assets/fly/body.xml')
OUT = ENGINE.parent / '.work/viewchk'
CAM = ['0', '-18', '6', '0', '0', '1', '40']
MOJO = str(Path(__file__).resolve().parents[1] / 'tools/bin/mojo-cached')


def build(n):
    OUT.mkdir(parents=True, exist_ok=True)
    m = export(BODY, OUT, FEET, True)
    q, v = standing(m, 1)
    q.astype(np.float32).T.copy().tofile(OUT / 'start-qpos.bin')  # view and smooth write the state back in place
    v.astype(np.float32).T.copy().tofile(OUT / 'start-qvel.bin')
    u = np.array([q[0, m.jnt_qposadr[m.actuator_trnid[i, 0]]] if m.actuator_trntype[i] == 0 else 1.0 for i in range(m.nu)])
    u.astype(np.float32).tofile(OUT / 'ctrl.bin')
    meshes(BODY, OUT)
    kin = OUT / 'kin'  # kinematics only: its own export, since smooth refuses tables sized for another stage
    mk = export(BODY, kin)
    np.zeros(mk.nu, np.float32).tofile(kin / 'ctrl.bin')  # STAGE 1 reads no ctrl, but smooth loads it
    (kin / 'start-qpos.bin').write_bytes((OUT / 'start-qpos.bin').read_bytes())
    (kin / 'qvel.bin').write_bytes((OUT / 'start-qvel.bin').read_bytes())
    flags = (OUT / 'sizes.txt').read_text().split() + (OUT / 'rsizes.txt').read_text().split()
    common = [*flags, '-DFLIES=1', '-DW=960', '-DH=540', f'-DPROWS={7 * (m.nbody - 1)}', *os.environ.get('MOJO_DEFS', '').split()]
    for src, k, exe, stage in (('view.mojo', 1, 'view', 7), ('smooth.mojo', n, 'smooth', 7), ('render.mojo', 1, 'render', 7)):
        subprocess.run([MOJO, 'build', '-I', str(ENGINE), *common, f'-DK={k}', f'-DSTAGE={stage}', str(ENGINE / src), '-o', str(OUT / exe)], check=True)
    subprocess.run([MOJO, 'build', '-I', str(ENGINE), *(kin / 'sizes.txt').read_text().split(), '-DFLIES=1', '-DSTAGE=1', '-DK=0',
                    str(ENGINE / 'smooth.mojo'), '-o', str(OUT / 'kin_bin')], check=True)
    print(f'built view, smooth (K={n}), kin (STAGE 1), render in {OUT}')


def run(n):
    d = str(OUT) + '/'

    def start():  # both binaries begin from the same saved state
        for f in ('qpos.bin', 'qvel.bin'):
            (OUT / f).write_bytes((OUT / ('start-' + f)).read_bytes())
    start()
    subprocess.run([str(OUT / 'view'), d, '1', *CAM, str(OUT / 'view1.rgba')], check=True, stdout=subprocess.DEVNULL)
    start()
    subprocess.run([str(OUT / 'view'), d, str(n), *CAM, str(OUT / 'view.rgba')], check=True, stdout=subprocess.DEVNULL)
    start()
    subprocess.run([str(OUT / 'smooth'), d], check=True, stdout=subprocess.DEVNULL)
    nq = len((OUT / 'start-qpos.bin').read_bytes()) // 4
    (OUT / 'kin' / 'qpos.bin').write_bytes((OUT / 'kin' / 'start-qpos.bin').read_bytes())
    subprocess.run([str(OUT / 'kin_bin'), str(OUT / 'kin') + '/'], check=True, stdout=subprocess.DEVNULL)  # STAGE 1, K=0: the start state's pose
    sm = np.fromfile(OUT / 'kin' / 'smooth.bin', np.float32)
    nbody = int([f.split('=')[1] for f in (OUT / 'sizes.txt').read_text().split() if f.startswith('-DNBODY=')][0])
    sm[:7 * (nbody - 1)].tofile(OUT / 'pose.bin')  # rows xpos, xquat of bodies 1.. (F = 1)
    subprocess.run([str(OUT / 'render'), d, str(OUT / 'pose.bin'), str(OUT / 'render.rgba'), '1', *CAM, '0'], check=True, stdout=subprocess.DEVNULL)
    ok = True
    vq = np.fromfile(OUT / 'view.rgba.qpos', np.float32)
    sq = np.fromfile(OUT / 'state.bin', np.float32)[:nq]
    if vq.tobytes() == sq.tobytes():
        print(f'PASS state after {n} x K=1 launches == smooth K={n} (byte-equal, {nq} floats)')
    else:
        ok = False
        print(f'FAIL state: max |view - smooth| {np.abs(vq - sq).max():.3e}')
    vf, rf = (OUT / 'view1.rgba').read_bytes(), (OUT / 'render.rgba').read_bytes()
    if vf == rf:
        print(f'PASS first frame == render.mojo on the start pose (byte-equal, {len(vf)} bytes)')
    else:
        a, b = np.frombuffer(vf, np.uint8), np.frombuffer(rf, np.uint8)
        ok = False
        print(f'FAIL frame: {int((a != b).sum())} of {a.size} bytes differ')
    print('PASS view_check' if ok else 'FAIL view_check')
    return ok


if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if sys.argv[1] == 'build':
        build(n)
    else:
        sys.exit(0 if run(n) else 1)
