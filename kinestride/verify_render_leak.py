"""The renderer must not leak: stream 10,000 frames through a pipe and sample the process at frames 100 and 10,000.
A leak is growth: open fds must be equal, VRAM (amdgpu drm-memory-vram over the process's fdinfo) must not
grow, host RSS may grow by page noise only (under 1 MB). Both may fall: with no MAX pool the runtime hands back
~6 MB of start-up allocations after the first frames. The scar: Mojo 1.0 input() leaked one pipe fd per call (1a07a08).

usage: python -m kinestride.verify_render_leak [RENDER_BINARY RENDER_DIR POSES]   (under gpu-wait; the
       defaults are render_check's 960 x 540 build, so run render_check first)
"""
import os
import subprocess
import sys

C = '.work/render_check/'
FRAMES, EARLY = 10_000, 100
W, H = 960, 540


def sample(pid):
    fds = os.listdir(f'/proc/{pid}/fd')
    vram = 0
    for fd in fds:
        try:
            for line in open(f'/proc/{pid}/fdinfo/{fd}'):
                if line.startswith('drm-memory-vram:'):
                    vram += int(line.split()[1])
        except OSError:
            pass
    rss = int(next(l for l in open(f'/proc/{pid}/status') if l.startswith('VmRSS:')).split()[1])
    return len(fds), vram, rss


def main(binary, rdir, poses):
    p = subprocess.Popen([binary, rdir, poses, '-', str(FRAMES), '-5', '-5', '2.5', '0', '0', '0.5', '45', '0'], stdout=subprocess.PIPE)
    frame, got = W * H * 4, []
    for n in range(1, FRAMES + 1):
        buf = p.stdout.read(frame)
        if len(buf) != frame:
            sys.exit(f'FAIL verify_render_leak: stream ended at frame {n}')
        if n in (EARLY, FRAMES):
            got.append(sample(p.pid))
    p.stdout.close()
    if p.wait() != 0:
        sys.exit(f'FAIL verify_render_leak: renderer exited {p.returncode}')
    (f0, v0, r0), (f1, v1, r1) = got
    ok = f0 == f1 and v1 <= v0 and r1 - r0 < 1024
    print(f'frames {EARLY} / {FRAMES}: fds {f0} / {f1}, VRAM {v0} / {v1} KiB, RSS {r0} / {r1} KiB -> {"PASS" if ok else "FAIL leak"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(*(sys.argv[1:4] if len(sys.argv) > 3 else (C + 'render1', C, C + 'smooth.bin')))
