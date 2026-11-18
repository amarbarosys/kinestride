"""Gate for kinestride/npc_live.mojo (check.sh npclive, private note BRAIN B4): reads T/run.log and the traces the
runner wrote. usage: tools/npclive_check.py T F
  fake == grid        the body-override path changes nothing when the body lands on the grid move (200 ticks, byte-equal)
  calib               walk >= 10 mm forward in 0.8 s; spins turn > 90 degrees each way; standing stays put
  scripted body       20 ticks: fwd tracked >= 0.9, turns >= 0.85, none fallen, the embodied NPCs eat
  base body           20 ticks, the imitated npc-mind brain: tracked >= 0.8, two runs byte-equal, every trace value finite
  mutants             drives zeroed / turn sign flipped: tracked < 0.3 (caught)
Exit 1 with FAIL on any miss."""
import re
import sys
import numpy as np

T, F = sys.argv[1], int(sys.argv[2])
N = 100
fail = 0


def check(name, ok, info=''):
    global fail
    print(('PASS ' if ok else 'FAIL ') + name + (': ' + info if info else ''))
    fail += 0 if ok else 1


def run(tag):
    for line in open(f'{T}/run.log'):
        if line.startswith('npclive ') and line.split()[1:4] == tag.split():
            w = line.split()
            k = {n: tuple(map(int, w[w.index(n) + 1].split('/'))) for n in ('fwd', 'left', 'right', 'flee', 'stay')}
            return k, int(w[w.index('fallen') + 1]), float(w[w.index('eats_body') + 1])
    raise SystemExit(f'FAIL run.log: no line for {tag}')


def frac(k, names):
    hit = sum(k[n][0] for n in names)
    cnt = sum(k[n][1] for n in names)
    return hit / max(cnt, 1), f'{hit}/{cnt}'


def same(a, b):
    return open(f'{T}/{a}', 'rb').read() == open(f'{T}/{b}', 'rb').read()


def finite(path):
    b = open(f'{T}/{path}', 'rb').read()
    rec = 12 * N + 32 * N + 24 * F
    ok = len(b) % rec == 0
    for t in range(len(b) // rec):
        r = b[t * rec:(t + 1) * rec]
        ok &= bool(np.isfinite(np.frombuffer(r[12 * N:44 * N], np.float64)).all())
        ok &= bool(np.isfinite(np.frombuffer(r[44 * N:44 * N + 12 * F], np.float32)).all())
    return ok


check('fake == grid, 200 ticks', same('sg.bin', 'sf.bin'))
cal = {}
for line in open(f'{T}/run.log'):
    m = re.match(r'calib (\d+) (\w+) s \S+ along_mm (\S+) left_mm (\S+) yaw_deg (\S+)', line)
    if m:
        cal.setdefault(m[2], (float(m[3]), float(m[4]), float(m[5])))
check('calib walk', cal['walk'][0] >= 10 and abs(cal['walk'][1]) < 5, 'along %.1f mm, left %.1f mm' % cal['walk'][:2])
check('calib spin left', cal['left'][2] > 90, '%.0f deg' % cal['left'][2])
check('calib spin right', cal['right'][2] < -90, '%.0f deg' % cal['right'][2])
check('calib stand', abs(cal['stand'][0]) < 1 and abs(cal['stand'][2]) < 5, 'along %.2f mm, %.1f deg' % (cal['stand'][0], cal['stand'][2]))
k, fallen, eats = run('scripted body 0')
f, s = frac(k, ['fwd'])
check('scripted fwd tracked >= 0.9', f >= 0.9, s)
f, s = frac(k, ['left', 'right'])
check('scripted turns tracked >= 0.85', f >= 0.85, s)
check('scripted none fallen, embodied NPCs eat', fallen == 0 and eats > 0, f'fallen {fallen}, eats per NPC {eats}')
k, fallen, eats = run('base body 0')
f, s = frac(k, ['fwd', 'left', 'right', 'flee', 'stay'])
check('base brain tracked >= 0.8', f >= 0.8 and fallen == 0, f'{s}, fallen {fallen}')
check('base brain two runs byte-equal', same('bb1.bin', 'bb2.bin'))
check('traces finite', all(finite(p) for p in ('sb.bin', 'bb1.bin')))
for mut, log in (('drives zeroed', 'm1.log'), ('turn sign flipped', 'm2.log')):
    k = None
    for line in open(f'{T}/{log}'):
        if line.startswith('npclive '):
            w = line.split()
            k = {n: tuple(map(int, w[w.index(n) + 1].split('/'))) for n in ('fwd', 'left', 'right', 'flee', 'stay')}
    f, s = frac(k, ['fwd', 'left', 'right', 'flee'])
    check(f'mutant {mut} caught (tracked < 0.3)', f < 0.3, s)
if fail:
    sys.exit(f'FAIL npclive: {fail} checks')
print('PASS npclive')
