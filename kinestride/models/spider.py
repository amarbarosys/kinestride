"""Build a jumping-spider body (Salticidae-like) as MJCF for the body engine, then check that it stands in MuJoCo.

Units and conventions follow the fly model (mm, g, s; gravity -9810 mm/s^2; dt 0.1 ms; floor contacts as explicit
plane-mesh pairs with the fly's solref/solimp/friction/margin; position servos). Every part is a convex mesh, so the
engine's ported plane-mesh contact covers it with no new code.

Anatomy (approximate, generic salticid; NOT measured from a specimen): prosoma 2.6 x 2.2 x 1.4 mm, opisthosoma
3.0 x 2.2 x 1.8 mm on a 2-hinge pedicel; 8 legs of coxa, femur, patella, tibia, metatarsus, tarsus (trochanter folded
into the coxa), leg I and IV the longest; 7 hinges per leg (coxa yaw + pitch, femur, patella, tibia yaw, metatarsus,
tarsus). Density 1e-3 g/mm^3 (water). Zero joint angles are the crouched stance, so servos at 0 hold it.
Not modelled: hydraulic leg extension (spiders lack extensors at two leg joints), silk, eyes.

usage: python -m kinestride.models.spider [OUT_DIR]   -> OUT_DIR/spider.xml + meshes; prints the stand test
"""
import struct
import sys
from pathlib import Path

import numpy as np

OUT = Path('.work/spider-body')
SEG = [('coxa', 0.5, 0.0), ('femur', 1.8, 35.0), ('patella', 0.8, -70.0), ('tibia', 1.3, -20.0),
       ('metatarsus', 1.2, -15.0), ('tarsus', 0.6, -10.0)]  # name, length mm, pitch relative to parent (deg, + up)
LEGS = [(1, 55.0, 1.10, 0.55), (2, 25.0, 0.90, 0.45), (3, -20.0, 0.90, 0.40), (4, -60.0, 1.05, 0.40)]  # pair, yaw, scale, radius
JUMP_S = 95.6  # servo boost of the jump variant (see build)
PREY_R = 0.5  # mm, prey ellipsoid radius (a ~1 mm fly-sized dot)
WATER_XY, WATER_R = (19.5, 23.0), 3.0  # mm: a bright drop of water on the ground, 30 mm from the start at 50 deg left (spider_drives.py)
KP, FMAX, DAMP = 2700.0, 4000.0, 3.0  # servo gain and force limit, dof damping (fly 45 / 65 / 0.06, x ~60 for ~20x mass and ~3x length)


def prism(path, length, r0, r1, n=8):
    """Convex tapered octagonal prism along +x from 0 to length (radius r0 -> r1), as binary STL."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda x, r: np.c_[np.full(n, x), r * np.cos(a), r * np.sin(a)]
    v = np.r_[ring(0, r0), ring(length, r1), [[-0.3 * r0, 0, 0]], [[length + 0.3 * r1, 0, 0]]]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + struct.pack('<I', len(tris)))
        for t in tris:
            f.write(struct.pack('<12fH', 0, 0, 0, *v[t[0]], *v[t[1]], *v[t[2]], 0))


def ellipsoid(path, ax, ay, az, n=10):
    """Convex polyhedral ellipsoid (semi-axes ax, ay, az), as binary STL."""
    th, ph = np.linspace(0, np.pi, n + 1)[1:-1], np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    v = [[0, 0, az]] + [[ax * np.sin(t) * np.cos(p), ay * np.sin(t) * np.sin(p), az * np.cos(t)] for t in th for p in ph] + [[0, 0, -az]]
    v, m, tris = np.array(v), 2 * n, []
    for i in range(m):
        tris.append((0, 1 + i, 1 + (i + 1) % m))
    for k in range(len(th) - 1):
        for i in range(m):
            a, b = 1 + k * m + i, 1 + k * m + (i + 1) % m
            tris += [(a, a + m, b), (b, a + m, b + m)]
    last = 1 + (len(th) - 1) * m
    for i in range(m):
        tris.append((len(v) - 1, last + (i + 1) % m, last + i))
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + struct.pack('<I', len(tris)))
        for t in tris:
            f.write(struct.pack('<12fH', 0, 0, 0, *v[t[0]], *v[t[1]], *v[t[2]], 0))


def build(out=OUT, prey=False, jump=False):
    """prey=True adds a free prey body (dark ellipsoid, free joint, no contacts, no servos; slide joints would be 'simple dofs', which the export refuses) and writes spider_prey.xml:
    its qpos is set each control step by the environment (eyes.mojo prey kernel), never by the brain.
    The prey body is gravity-compensated: prey_set places it before each control step and the K physics steps after must not drop it (0.49 mm per 10 ms frame, 0.15 deg at 191 mm, out of the AM row).
    jump=True writes spider_jump.xml: the first four servos of every leg (coxa yaw and pitch, femur, patella) get JUMP_S x the gain and
    force limit and the joint damping 3 + 0.3 JUMP_S (an explicit servo stays stable only for damping >= kp dt): the model's stand-in for
    the hydraulic leg extension real salticids jump with (they have no extensor muscles at two leg joints). Walking uses spider.xml."""
    out.mkdir(parents=True, exist_ok=True)
    ellipsoid(out / 'prosoma.stl', 1.3, 1.1, 0.7)
    ellipsoid(out / 'opisthosoma.stl', 1.5, 1.1, 0.9)
    if prey:
        ellipsoid(out / 'prey.stl', PREY_R, PREY_R, PREY_R, n=6)
        ellipsoid(out / 'water.stl', WATER_R, WATER_R, 0.05, n=12)
    assets, geoms, actuators, pairs = ['<mesh name="prosoma" file="prosoma.stl"/>', '<mesh name="opisthosoma" file="opisthosoma.stl"/>'], [], [], []

    def leg(side, pair, yaw, scale, radius):
        name = f'{"l" if side > 0 else "r"}{pair}'
        ys = np.radians(yaw)
        base = (0.9 * np.sin(ys), side * 0.9 * np.cos(ys), -0.2)  # on the prosoma's rim, forward legs further forward
        quat = (np.cos(side * np.radians(90 - yaw) / 2), 0, 0, np.sin(side * np.radians(90 - yaw) / 2))  # leg x points outward
        xml, close, r = [], [], radius
        for k, (seg, length, pitch) in enumerate(SEG):
            L = length * scale
            mesh = f'{name}_{seg}'
            prism(out / f'{mesh}.stl', L, r, r * 0.85)
            assets.append(f'<mesh name="{mesh}" file="{mesh}.stl"/>')
            q = (np.cos(np.radians(-pitch) / 2), 0, np.sin(np.radians(-pitch) / 2), 0)
            if k == 0:
                xml.append(f'<body name="{mesh}" pos="{base[0]:.3f} {base[1]:.3f} {base[2]:.3f}" quat="{quat[0]:.6f} 0 0 {quat[3]:.6f}">')
                joints = [(f'{name}_coxa_yaw', '0 0 1'), (f'{name}_coxa_pitch', '0 1 0')]
            else:
                prev = SEG[k - 1][1] * scale
                xml.append(f'<body name="{mesh}" pos="{prev:.3f} 0 0" quat="{q[0]:.6f} 0 {q[2]:.6f} 0">')
                joints = [(f'{name}_{seg}', '0 0 1' if seg == 'tibia' else '0 1 0')]
            for j, axis in joints:
                boost = JUMP_S if jump and not j.endswith(('tibia', 'metatarsus', 'tarsus')) else 1.0
                xml.append(f'<joint name="{j}" axis="{axis}" range="-60 60"' + (f' damping="{DAMP + 0.3 * boost}"' if boost > 1 else '') + '/>')
                actuators.append(f'<position name="{j}" joint="{j}" kp="{KP * boost}" forcelimited="true" forcerange="{-FMAX * boost} {FMAX * boost}"/>')
            xml.append(f'<geom name="{mesh}" type="mesh" mesh="{mesh}"/>')
            pairs.append(mesh)
            close.append('</body>')
            r *= 0.85
        return '\n'.join(xml + close[::-1])

    legs = '\n'.join(leg(s, *p) for s in (1, -1) for p in LEGS)
    pairs = ['prosoma', 'opisthosoma'] + pairs
    PREYBODY = ''
    if prey:
        assets.append('<mesh name="prey" file="prey.stl"/>')
        PREYBODY = ('<body name="prey" pos="0 0 0" gravcomp="1">\n<freejoint name="prey"/>\n'
                    '<geom name="prey" type="mesh" mesh="prey" rgba="0.02 0.02 0.02 1"/>\n'
                    '<body name="prey_tip" pos="0 0 0" gravcomp="1"><geom type="sphere" size="1e-3" mass="1e-9"/></body>\n</body>\n')  # a child: a leaf free body is a "simple dof"
        assets.append('<mesh name="water" file="water.stl"/>')
        PREYBODY += f'<body name="water" pos="{WATER_XY[0]} {WATER_XY[1]} 0.05"><geom name="water" type="mesh" mesh="water" rgba="0.95 0.95 0.95 1"/></body>\n'  # no joint: fixed, after the prey (prey_set: the last 7 qpos)

    xml = f'''<mujoco model="spider">
<compiler angle="degree" meshdir="."/>
<option timestep="1e-4" gravity="0 0 -9810" cone="pyramidal" iterations="100" ls_iterations="50"/>
<default>
  <geom contype="0" conaffinity="0" density="1e-3" rgba="0.35 0.3 0.25 1"/>
  <joint damping="{DAMP}" armature="1e-4" limited="false"/>
</default>
<asset>
{chr(10).join(assets)}
</asset>
<worldbody>
<geom name="ground_plane" type="plane" size="40 40 1" rgba="0.6 0.6 0.6 1"/>
<body name="prosoma" pos="0 0 2.6">
<freejoint name="root"/>
<geom name="prosoma" type="mesh" mesh="prosoma"/>
<body name="opisthosoma" pos="-1.3 0 0.1">
<joint name="pedicel_pitch" axis="0 1 0"/>
<joint name="pedicel_yaw" axis="0 0 1"/>
<geom name="opisthosoma" type="mesh" mesh="opisthosoma" pos="-1.5 0 0"/>
</body>
{legs}
</body>
{PREYBODY}</worldbody>
<contact>
{chr(10).join(f'<pair geom1="ground_plane" geom2="{g}" solref="2e-4 1" solimp="0.98 0.99 1e-5 0.5 3" friction="1 1" margin="0.001"/>' for g in pairs)}
</contact>
<actuator>
<position name="pedicel_pitch" joint="pedicel_pitch" kp="{KP}" forcelimited="true" forcerange="{-FMAX} {FMAX}"/>
<position name="pedicel_yaw" joint="pedicel_yaw" kp="{KP}" forcelimited="true" forcerange="{-FMAX} {FMAX}"/>
{chr(10).join(actuators)}
</actuator>
</mujoco>
'''
    name = 'spider_prey.xml' if prey else ('spider_jump.xml' if jump else 'spider.xml')
    (out / name).write_text(xml)
    return out / name


def stand(path, seconds=0.5):
    """Servos at 0 (the stance) from the build pose: root height and tilt after `seconds`, and the worst |qvel|."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    zs = []
    for k in range(round(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if k % 500 == 0:
            zs.append(round(float(d.qpos[2]), 3))
    up = np.zeros(9)
    mujoco.mju_quat2Mat(up, d.qpos[3:7])
    tilt = float(np.degrees(np.arccos(np.clip(up[8], -1, 1))))
    return {'mass_g': float(m.body_subtreemass[1]), 'nbody': m.nbody, 'nv': m.nv, 'nu': m.nu, 'npair': m.npair,
            'root_z_every_50ms': zs, 'tilt_deg': tilt, 'max_qvel': float(np.abs(d.qvel).max()),
            'stands': bool(zs[-1] > 1.0 and tilt < 20 and np.abs(d.qvel).max() < 1.0)}


if __name__ == '__main__':
    p = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    r = stand(p)
    print(p, r)
    sys.exit(0 if r['stands'] else 'FAIL spider: does not stand in MuJoCo')
