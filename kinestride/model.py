"""Any MuJoCo model (fly, spider, any MJCF body) to the engine's two flat arrays, plus the oracle that
checks each ported stage against MuJoCo itself.

The engine reads no MJCF. `export` writes mi.bin (int32) and mf.bin (float32): the mjModel fields a
ported stage needs, by their MuJoCo names, concatenated in the order of INTS / FLOATS. Sizes go to
sizes.txt as -D flags, so each body type gets its own compiled engine with its loop bounds and array
offsets as constants (the shiu_worker pattern). A new species is a new MJCF file, not new code.

`check STAGE`: random states from ~/iTools/science/mj-oracle (free joint anywhere, any orientation;
hinges and slides inside their ranges), MuJoCo runs the stage in float64, the engine runs it in
float32 on the GPU, and every output field must agree to REL of its own scale (mj-oracle compare).

usage: python -m kinestride.model export MJCF OUT_DIR
       python -m kinestride.model check  MJCF OUT_DIR STAGE [STATES]
       python -m kinestride.model meshes MJCF OUT_DIR   (renderer tables; fails above 1e-4 decode error)
"""
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path.home() / 'iTools/science/mj-oracle'))
from mj_oracle import compare, reference, states, velocities  # noqa: E402  (random states, MuJoCo reference, per-field compare)

INTS = ('body_parentid', 'body_jntadr', 'body_jntnum', 'jnt_type', 'jnt_qposadr', 'geom_bodyid',
        'body_rootid', 'jnt_dofadr', 'dof_parentid', 'dof_bodyid', 'M_rowadr', 'M_rownnz', 'M_colind',
        'body_dofnum', 'body_dofadr')
FLOATS = ('body_pos', 'body_quat', 'jnt_pos', 'jnt_axis', 'qpos0', 'body_ipos', 'body_iquat', 'geom_pos', 'geom_quat',
          'body_mass', 'body_subtreemass', 'body_inertia', 'dof_armature',
          'dof_damping', 'jnt_stiffness', 'qpos_spring', 'opt.gravity', 'opt.timestep')
KIN = ('xpos', 'xquat', 'xipos', 'geom_xpos', 'xanchor', 'xaxis')
STAGES = {  # name: (STAGE number in smooth.mojo, MuJoCo functions run in order, output fields)
    'kinematics': (1, 'mj_kinematics', KIN),
    'crb': (2, 'mj_kinematics,mj_comPos,mj_crb', KIN + ('subtree_com', 'cinert', 'cdof', 'crb', 'M')),
    'factorM': (3, 'mj_kinematics,mj_comPos,mj_crb,mj_factorM', KIN + ('subtree_com', 'cinert', 'cdof', 'crb', 'M', 'qLD', 'qLDiagInv')),
    'rne': (4, 'mj_kinematics,mj_comPos,mj_crb,mj_factorM,mj_fwdVelocity',  # mj_comVel, mj_passive, mj_rne(flg_acc 0)
            KIN + ('subtree_com', 'cinert', 'cdof', 'crb', 'M', 'qLD', 'qLDiagInv', 'cvel', 'cdof_dot', 'qfrc_passive', 'qfrc_bias')),
}
# 'step': one mj_step with contacts, constraints and actuators off (S1 is contact-free): mj_Euler with implicit
# dof damping. qacc_euler = (qvel after - qvel) / dt and qpos_next are derived from MuJoCo's own step.
STAGES['step'] = (5, 'mj_step', STAGES['rne'][2] + ('qH', 'qHDiagInv', 'qacc_euler', 'qpos_next'))
# 'sptendon': 'step' plus the tendon outputs (mj_tendon: ten_length, ten_velocity, ten_J as dense rows) on a model with spatial tendons
# (models/tendon_spatial.xml); the tendons' spring and damper reach qfrc_passive, so every step field sees them.
STAGES['sptendon'] = (5, 'mj_step', STAGES['step'][2] + ('ten_length', 'ten_velocity', 'ten_J'))
TENDON_FIELDS = ('ten_length', 'ten_velocity', 'ten_J')
# 'contact' (S2 probe): plane-mesh pairs whose mesh geom matches FEET, pyramidal cone, Newton solver, then the
# implicit Euler step. States come from an actuated MuJoCo rollout (feet on the floor), not random poses.
STAGES['contact'] = (6, 'mj_step', STAGES['step'][2] + ('ncon', 'qacc', 'qfrc_constraint'))
# 'actuated' (S3): + mj_transmission (hinge/slide joints, adhesion bodies) and mj_fwdActuation (fixed gain, affine
# or no bias, no activation dynamics), random controls, actuators on.
STAGES['actuated'] = (7, 'mj_step', STAGES['contact'][2] + ('actuator_force', 'qfrc_actuator'))
CONVEX_POOL = 48  # box and ellipsoid contacts one animal can hold at once (the fly at a wall: a few parts x 1-5 contacts)
PRIM_POOL = 96  # primitive-pair contacts one animal can hold at once (the flight model's 842 body-body pairs: 26 per random state, 54 max)
PRIM = {mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CYLINDER}  # primitive pairs (convex.mojo kinds 4-7)
SHAPE_KIND = {mujoco.mjtGeom.mjGEOM_ELLIPSOID: 0, mujoco.mjtGeom.mjGEOM_MESH: 1, mujoco.mjtGeom.mjGEOM_BOX: 2, mujoco.mjtGeom.mjGEOM_SPHERE: 4,
              mujoco.mjtGeom.mjGEOM_CAPSULE: 5, mujoco.mjtGeom.mjGEOM_CYLINDER: 6}  # convex.mojo shape kinds


def pair_geoms(m, i):
    """Explicit pair i's geoms in mj_collideGeoms' order (geom1 the lower type; the compiler keeps the pair element's own order), the
    order every contact of the pair and the engine's pair table use (normal from geom1 to geom2, the Jacobian signed geom2 minus geom1)."""
    g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
    return (g1, g2) if m.geom_type[g1] <= m.geom_type[g2] else (g2, g1)


def prim_pair(m, i):
    """The convex.mojo kind of explicit pair i when both geoms are primitives (geom1 the lower type, as mj_collideGeoms orders them):
    5 capsule-capsule, 6 sphere-capsule, 7 sphere-cylinder (analytic), 4 the rest (GJK + EPA); 0 otherwise (sphere-sphere has its own path)."""
    G = mujoco.mjtGeom
    t1, t2 = (G(m.geom_type[g]) for g in pair_geoms(m, i))
    if t1 not in PRIM or t2 not in PRIM or (t1 == t2 == G.mjGEOM_SPHERE):
        return 0
    return {(G.mjGEOM_CAPSULE, G.mjGEOM_CAPSULE): 5, (G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE): 6, (G.mjGEOM_SPHERE, G.mjGEOM_CYLINDER): 7}.get((t1, t2), 4)
HF_POOL = 192  # exact heightfield contacts one animal can hold at once (bumpy terrain: ~80 on average, 55 pairs x up to 16)
HF_SLOTS = 16  # contacts kept per heightfield pair with exact heightfield contact (check.sh terrain_exact reports MuJoCo's maximum)
STAGES['sptendon_act'] = (7, 'mj_step', STAGES['actuated'][2] + ('ten_length', 'ten_velocity', 'ten_J'))  # the actuated path on spatial tendons (tendon servo, tendon passive)
# 'equality' (S6 without pairs): connect, weld and joint equality constraints (eq_rows), Newton on always-active rows, then
# the implicit Euler step; contacts and actuation off, random states violate the constraints. Test body models/eq_test.xml.
STAGES['equality'] = (6, 'mj_step', STAGES['step'][2] + ('ncon', 'qacc', 'qfrc_constraint'))
# 'mocap': the same fields with mocap bodies (mocap_pos/quat random around the model pose, models/eq_mocap.xml), welded or connected to the animal.
STAGES['mocap'] = STAGES['equality']
STAGES['tendon_limit'] = (6, 'mj_step', STAGES['equality'][2] + TENDON_FIELDS)  # tendon limit rows (mj_instantiateLimit, unilateral) ride the equality block: models/tendon_spatial.xml with limits (with_tendon_limits)
STAGES['tendon_eq'] = STAGES['equality']  # tendon equality rows (mjEQ_TENDON, polycoef) on the equality block: with_tendon_equalities (SPATIAL builds add the tendon fields)
STAGES['joint_limit'] = (6, 'mj_step', STAGES['equality'][2])  # hinge/slide limit rows (mj_instantiateLimit, unilateral) on the equality block: with_joint_limits
FEET = '.'  # every plane-mesh pair (55 on the fly: each body part on the floor); 'tarsus5' (6 feet) sank through the floor in MuJoCo too
VSCALE = float(os.environ.get('VSCALE', 10))  # random qvel in +-VSCALE per dof (10: velocity terms well above float32 noise; qpos moves ~1e-3 per fly step); the flight gate draws up to flapping rate
REL = float(os.environ.get('REL', 1e-5))  # check.sh sets a measured bar for `arena` (3e-3) and `terrain_exact` (4e-4), reasons there
EQ_EULER_BAR = 1e-4  # equality stage, qacc_euler only: see check()
SLEEP_QPOS_BAR = 1e-5  # sleep stage, the sleepers' frozen qpos: see check()
CHAOS_SEEDS = int(os.environ.get('CHAOS_SEEDS', 12))  # k-step gate: MuJoCo reruns of each start under a 1e-6 perturbation (see check())
FLOOR_X = 3  # contact stage: bar = max(REL, 3 x the field's float32 input floor), see check()
FLOOR_X_CONE = 15  # elliptic cone: the converged nonlinear cone solve amplifies float32 rounding of D, J and jar: measured 4e-5 (noslip 0) and 5.3e-5 (noslip 5) on qfrc_constraint, floor 4.3e-6
WORLD_FIRST = ('xpos', 'xquat', 'xipos', 'cinert', 'crb', 'cvel')  # body-indexed fields whose row 0 is the world
ENGINE = Path(__file__).parent


def cube(mjcf, side=40.0):
    """A closed cube of side `side` (mm) around the arena: the ground plane plus 4 wall planes and a ceiling plane,
    normals inward, each colliding with every fly part the ground does (the ground pair's parameters copied). The
    old box walls, lid and stones and their pairs are removed, so every contact is plane-mesh (ported). Written next
    to mjcf (its meshes are relative paths); returns the new path."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(mjcf)
    root = tree.getroot()
    world, contact = root.find('worldbody'), root.find('contact')
    ground = world.find("geom[@name='ground_plane']")
    for g in list(world.findall('geom')):
        if g is not ground and g.get('type') in ('box', 'ellipsoid') and not g.get('name', '').startswith('stripe'):
            world.remove(g)
    keep = [q for q in contact.findall('pair') if q.get('geom1') == 'ground_plane']
    for q in list(contact):
        contact.remove(q)
    h = side / 2
    faces = {'wall_xn': (f'{-h} 0 {h}', '1 0 0'), 'wall_xp': (f'{h} 0 {h}', '-1 0 0'), 'wall_yn': (f'0 {-h} {h}', '0 1 0'),
             'wall_yp': (f'0 {h} {h}', '0 -1 0'), 'ceiling': (f'0 0 {side}', '0 0 -1')}
    for name, (pos, z) in faces.items():
        g = ET.SubElement(world, 'geom', dict(ground.attrib))
        g.set('name', name); g.set('pos', pos); g.set('zaxis', z); g.set('size', f'{h} {h} 1')
        g.attrib.pop('quat', None)
    for name in ('ground_plane', *faces):
        for q in keep:
            r = ET.SubElement(contact, 'pair', dict(q.attrib))
            r.set('geom1', name); r.set('name', f"{q.get('geom2')}-{name}")
    out = Path(mjcf).with_name(f'cube{int(side)}.xml')
    tree.write(out)
    return str(out)


WORLD_DEFAULT = (0.0, 0.0, 0.3, 8.0, 4, 1, 2.0, 0.5, 0.0, 40.0, 5.0, 40.0, 0, 0, 0, 0)  # world.mojo's 16 params (world.py,
# the oracle, holds the same): slope x y, amplitude mm, wavelength mm, octaves, seed, lacunarity, gain, landmark density,
# cell mm, cone radius mm, cone height mm, 4 unused
_WORLD = []


def world_full(p):
    """Params padded to 16 with WORLD_DEFAULT's tail (landmarks off)."""
    return tuple(p) + WORLD_DEFAULT[len(p):]


def world_height_grad(x, y, p=WORLD_DEFAULT):
    """(height, dh/dx, dh/dy) of the world.mojo terrain at x, y (any shape, world mm), from its CLI (world IN OUT N);
    world.py stays the oracle of `python -m kinestride.world diff`."""
    out = Path('.work/world-mojo')
    out.mkdir(parents=True, exist_ok=True)
    if not _WORLD:
        _WORLD.append(out / 'world')
        subprocess.run([str(Path.home() / 'iTools/bin/mojo-cached'), 'build', str(Path(__file__).with_name('world.mojo')),
                        '-o', str(_WORLD[0])], check=True)
    x, y = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32))
    src, dst = out / f'in-{os.getpid()}.bin', out / f'out-{os.getpid()}.bin'
    np.concatenate([np.stack([x.ravel(), y.ravel()], 1).ravel(), np.asarray(world_full(p), np.float32)]).astype(np.float32).tofile(src)
    subprocess.run([str(_WORLD[0]), str(src), str(dst), str(x.size)], check=True)
    r = np.fromfile(dst, np.float32).reshape(x.size, 3)
    return tuple(r[:, i].reshape(x.shape) for i in range(3))


def terrain(mjcf, params=(0.1, 0.05, 0, 8.0, 0, 0, 2.0, 0.5), n=257, half=40.0):
    """Ground plane replaced by the world.mojo terrain: its 8 params ride in <custom><numeric name="terrain"> (what the
    engine evaluates) and an n x n heightfield of +-half mm samples them (what MuJoCo collides with). Writes
    terrain.xml next to mjcf and, with octaves 0 (a pure ramp), its exact twin terrain_plane.xml: the same tilted
    plane as a plane geom, the oracle of the tilt gate. Returns the paths."""
    import xml.etree.ElementTree as ET
    full, height_grad = world_full, world_height_grad
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)  # row r along +y, column c along +x (mjModel.hfield_data order)
    H = height_grad(X, Y, params)[0].astype(np.float64)
    lo, span = H.min(), max(np.ptp(H), 1e-6)
    out = []
    for kind in ('hfield', 'plane') if not params[4] else ('hfield',):
        tree = ET.parse(mjcf)
        root = tree.getroot()
        ground = root.find("worldbody/geom[@name='ground_plane']")
        ground.attrib.pop('quat', None)
        if kind == 'hfield':
            asset = root.find('asset') if root.find('asset') is not None else ET.SubElement(root, 'asset')
            ET.SubElement(asset, 'hfield', name='terrain', nrow=str(n), ncol=str(n), size=f'{half} {half} {span} 1',
                          elevation=' '.join(f'{e:.9g}' for e in ((H - lo) / span)[::-1].ravel()))  # XML rows run +y first
            ground.set('type', 'hfield'); ground.set('hfield', 'terrain'); ground.set('pos', f'0 0 {lo}')
            ground.attrib.pop('size', None)
            custom = root.find('custom') if root.find('custom') is not None else ET.SubElement(root, 'custom')
            ET.SubElement(custom, 'numeric', name='terrain', data=' '.join(map(str, full(params))))
        else:
            ground.set('pos', '0 0 0'); ground.set('zaxis', f'{-params[0]} {-params[1]} 1')
        path = Path(mjcf).with_name('terrain.xml' if kind == 'hfield' else 'terrain_plane.xml')
        tree.write(path)
        out.append(str(path))
    return out


def shifted(mjcf, out, dx):
    """Copy mjcf to out with dx added to the x of every direct child of the worldbody (relative data: weld relpose,
    connect anchors, joint coefs, all unaffected by construction); returns the new path."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    for b in spec.worldbody.bodies:
        b.pos[0] += dx
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    return str(out)


def combine(fly, spider, offset=(8.0, 0.0, 0.0), name='world'):
    """Fly and spider in one model (one export, one engine binary): the spider's spec attached at `offset` with prefix
    sp_ (MjSpec.attach copies its assets, actuators and pairs), its own ground removed and its pairs retargeted to the
    shared ground_plane, mesh paths made absolute. MuJoCo 3.9 has no compiler/conflict (3.10), so every global option
    must already agree or the merge fails. Written next to the fly MJCF as NAME.xml; returns the path."""
    a, b = mujoco.MjSpec.from_file(str(fly)), mujoco.MjSpec.from_file(str(spider))
    oa, ob = a.compile().opt, b.compile().opt
    diff = [f for f in dir(oa) if not f.startswith('_') and not callable(getattr(oa, f)) and np.any(np.asarray(getattr(oa, f)) != np.asarray(getattr(ob, f)))
            and f not in ('enableflags', 'noslip_iterations')]  # the fly's win: energy reporting only, noslip 5
    if diff:
        raise SystemExit(f'FAIL combine: global options differ: {diff}')
    for spec, path in ((a, fly), (b, spider)):  # absolute mesh files: the merged MJCF loads from anywhere
        base = Path(path).parent / (spec.meshdir or '')
        for mesh in spec.meshes:
            mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
        spec.meshdir = ''
    a.attach(b, frame=a.worldbody.add_frame(pos=list(offset)), prefix='sp_')
    a.delete(a.geom('sp_ground_plane'))
    for pair in a.pairs:
        if pair.geomname1 == 'sp_ground_plane':
            pair.geomname1 = 'ground_plane'
    out = Path(fly).with_name(f'{name}.xml')
    out.write_text(a.to_xml())
    return str(out)


def airborne_mjcf(mjcf, spring=(2.0, 0.3, -0.04, 0.04)):
    """The flight MJCF as the tendon gate meets it (stage `actuated` needs the pyramidal cone and explicit pairs only): pyramidal
    cone, every geom contype/conaffinity 0 (no contacts, the model flies), and every fixed tendon given a spring, damper and
    spring length range (spring, damper, lower, upper; the shipped tendons have none, so their passive port would go
    untested). Written next to mjcf as airborne.xml; returns the path."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(mjcf)
    root = tree.getroot()
    opt = root.find('option') if root.find('option') is not None else ET.SubElement(root, 'option')
    opt.set('cone', 'pyramidal')
    for g in root.iter('geom'):
        g.set('contype', '0'); g.set('conaffinity', '0')
    for t in root.iter('fixed'):
        t.set('stiffness', str(spring[0])); t.set('damping', str(spring[1])); t.set('springlength', f'{spring[2]} {spring[3]}')
    out = Path(mjcf).with_name('airborne.xml')
    tree.write(out)
    return str(out)


def elliptic(mjcf, noslip=0):
    """The fly MJCF with the elliptic friction cone, noslip_iterations = noslip and solver tolerance 0 (the walking fly ships
    pyramidal, noslip 5, tolerance 1e-8): the gate of the elliptic Newton port, same pairs and states. The cone solve is
    nonlinear, so MuJoCo's own stop at tolerance 1e-8 leaves ~1e-4 of force error; tolerance 0 runs it to convergence and
    that is what the engine is compared with. Written next to mjcf as elliptic<noslip>.xml; returns the path."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(mjcf)
    opt = tree.getroot().find('option') if tree.getroot().find('option') is not None else ET.SubElement(tree.getroot(), 'option')
    opt.set('cone', 'elliptic'); opt.set('noslip_iterations', str(noslip)); opt.set('tolerance', '0')
    out = Path(mjcf).with_name(f'elliptic{noslip}.xml')
    tree.write(out)
    return str(out)


def onfaces(q, v, side):
    """Standing states moved onto the cube's 6 faces in turn (state n on face n % 6): the whole fly rotated so the
    floor's up becomes the face's inward normal and placed at the face; free-joint linear velocity rotated with it,
    angular velocity (body frame) unchanged. Gravity stays -z, so on walls and ceiling the fly hangs by contacts."""
    h = side / 2
    ups = [(0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1)]
    centres = [(0, 0, 0), (-h, 0, h), (h, 0, h), (0, -h, h), (0, h, h), (0, 0, side)]
    q, v = q.copy(), v.copy()
    for n in range(len(q)):
        up, c = np.array(ups[n % 6], float), np.array(centres[n % 6], float)
        axis, angle = np.cross([0, 0, 1], up), np.arccos(np.clip(up[2], -1, 1))
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array([1., 0, 0])
        r = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, r)
        R = R.reshape(3, 3)
        q[n, :3] = c + R @ q[n, :3]
        res = np.zeros(4)
        mujoco.mju_mulQuat(res, r, q[n, 3:7])
        q[n, 3:7] = res
        v[n, :3] = R @ v[n, :3]
    return q, v


INTEGRATORS = (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_RK4, mujoco.mjtIntegrator.mjINT_IMPLICIT, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
SOLVERS = (mujoco.mjtSolver.mjSOL_PGS, mujoco.mjtSolver.mjSOL_CG, mujoco.mjtSolver.mjSOL_NEWTON)  # all three ported
STALE_RK4 = ('qH', 'qHDiagInv')  # mj_RungeKutta and mj_implicit (LU in qLU) never factor qH: MuJoCo's d->qH after such a step is stale, not compared


def override(m):
    """ENGINE_OPT='integrator=3,kv=0.5': test-only option overrides applied to the model the engine AND MuJoCo both load.
    integrator: mjtIntegrator value. kv: velocity feedback on every affine-bias joint actuator (biasprm[2] = -kv), so
    implicitfast's actuator qDeriv is not zero (the shipped servos have kv 0). kp: servo gain and bias stiffness scale."""
    for kv in filter(None, os.environ.get('ENGINE_OPT', '').split(',')):
        key, val = kv.split('=')
        if key == 'integrator':
            m.opt.integrator = int(val)
        elif key == 'solver':
            m.opt.solver = int(val)
        elif key == 'noslip':  # noslip_iterations: 0 isolates the solver (noslip runs once over all contacts in the engine, per island in MuJoCo)
            m.opt.noslip_iterations = int(val)
        elif key == 'kv':
            joint = (m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) & (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
            m.actuator_biasprm[joint, 2] = -float(val)
        elif key == 'tkv':  # velocity feedback on every affine-bias tendon actuator (biasprm[2] = -tkv): its J' b J term in implicit qDeriv
            tendon = m.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON  # made affine: tendon_spatial.xml's servo ships biastype none
            m.actuator_biastype[tendon] = mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_biasprm[tendon, 2] = -float(val)
        elif key == 'cone':  # mjtCone value: the flight model ships elliptic, gated under Euler only; its fluid-derivative gate runs it pyramidal
            m.opt.cone = int(val)
        elif key == 'dt':  # timestep: explicit RK4 damps dofs explicitly, h B / M reaches 5.9 on the fly at h 1e-4 (stable below 2.8)
            m.opt.timestep = float(val)
        elif key == 'solref':  # every pair's solref, 'a:b'; negative = MuJoCo's direct format (-stiffness, -damping)
            m.pair_solref[:] = [float(x) for x in val.split(':')]
        elif key == 'rowsolref':  # every equality's solref, joint's solref and tendon's solref_lim, 'a:b' (check.sh solref_direct)
            m.eq_solref[:] = m.jnt_solref[:] = m.tendon_solref_lim[:] = [float(x) for x in val.split(':')]
        elif key == 'kp':  # servo stiffness scale: explicit RK4 at the shipped kp (h 1e-4, light limbs) is unstable in MuJoCo itself
            joint = (m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) & (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
            m.actuator_gainprm[joint, 0] *= float(val)
            m.actuator_biasprm[joint, 1] *= float(val)
        else:
            raise SystemExit(f'FAIL ENGINE_OPT: unknown key {key}')


def arena(m, q, v, seed=0):
    """Standing states moved into the legacy arena's convex geoms (box walls, stripes and beacons; ellipsoid stones): state n
    keeps its pose, is turned by a random yaw about its root, then its root is slid along the inward normal of one of the four walls (n % 4 < 2),
    or radially toward one of the stones (else), until MuJoCo's deepest box or ellipsoid contact has the target distance, drawn
    from -0.03 .. 0.0008 mm (penetrating to just inside the 1 mm margin: what a fly brushing a wall sees; deeper states are
    1000 x the margin and stiffer than anything the fly meets). Free-joint linear velocity is turned with the yaw."""
    G = mujoco.mjtGeom
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed + 7)
    convex = (G.mjGEOM_BOX, G.mjGEOM_ELLIPSOID)
    walls = [(-20.0, 0, 1, 0), (20.0, 0, -1, 0), (0, -20.0, 0, 1), (0, 20.0, 0, -1)]  # inner face x or y, inward normal
    stones = [(m.geom_pos[g][:2], m.geom_size[g][:2]) for g in range(m.ngeom) if m.geom_type[g] == G.mjGEOM_ELLIPSOID]
    q, v = q.copy(), v.copy()

    def deepest(qn):  # smallest distance over box and ellipsoid contacts (inf: none)
        d.qpos[:] = qn
        mujoco.mj_forward(m, d)
        return min([c.dist for c in d.contact[:d.ncon] if m.geom_type[c.geom1] in convex] or [np.inf])

    for n in range(len(q)):
        yaw = rng.uniform(0, 2 * np.pi)
        r = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        if n % 4 < 2 or not stones:
            fx, fy, nx, ny = walls[rng.integers(4)]
            t = rng.uniform(-15, 15)
            at = lambda dd: np.array([fx + nx * dd if nx else t, fy + ny * dd if ny else t])  # dd: distance of the fly from the wall
            far = 10.0
        else:
            c, a = stones[rng.integers(len(stones))]
            ang = rng.uniform(0, 2 * np.pi)
            u = np.array([np.cos(ang), np.sin(ang)])
            at = lambda dd: c + a * dd * u  # dd: radius in semi-axes
            far = 8.0
        res = np.zeros(4)
        mujoco.mju_mulQuat(res, r, q[n, 3:7])
        qn = q[n].copy()
        qn[3:7] = res
        target = rng.uniform(-0.03, 0.0008)
        lo, hi = 0.0, far  # lo: deep (inside), hi: clear
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            qn[:2] = at(mid)
            if deepest(qn) > target:
                hi = mid
            else:
                lo = mid
        qn[:2] = at(hi)
        q[n] = qn
        v[n, :3] = R @ v[n, :3]
    return q, v


STONES = (('stone_box', 'box', (1.0, 1.0, 0.3), (4.56, 3.04, -0.28)), ('stone_ell', 'ellipsoid', (1.5, 1.5, 0.5), (-1.27, 4.2, -0.48)),
          ('stone_box2', 'box', (1.0, 1.0, 0.3), (-4.37, -2.58, -0.28)))  # spider (mm): tops 0.02 above the floor under the tarsi of l1, l3, r4


def stones(mjcf, out):
    """Copy of an MJCF with static box and ellipsoid stones (STONES) and a contact pair from every stone to every mesh geom that
    has a floor pair (the floor pair's margin, solref, solimp and friction): the spider's legs meet box-mesh and ellipsoid-mesh
    contacts like the fly meets its arena's. Written next to the source so the mesh paths hold. Returns the new path."""
    G = mujoco.mjtGeom
    spec = mujoco.MjSpec.from_file(str(mjcf))
    ref = next(p for p in spec.pairs)
    meshes = [p.geomname2 for p in spec.pairs if spec.geom(p.geomname1).type == G.mjGEOM_PLANE]
    for name, kind, size, pos in STONES:
        g = spec.worldbody.add_geom(name=name, type=G.mjGEOM_BOX if kind == 'box' else G.mjGEOM_ELLIPSOID, size=list(size), pos=list(pos))
        g.rgba = [0.5, 0.5, 0.5, 1]
        for mg in meshes:
            spec.add_pair(geomname1=name, geomname2=mg, margin=ref.margin, solref=ref.solref, solimp=ref.solimp, friction=ref.friction)
    path = Path(mjcf).with_name(Path(out).name if str(out).endswith('.xml') else 'stones.xml')
    path.write_text(spec.to_xml())
    return str(path)


def load(mjcf):
    m = mjcf if isinstance(mjcf, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(mjcf))
    if not isinstance(mjcf, mujoco.MjModel):
        override(m)
    for d in np.flatnonzero(m.dof_simplenum):  # simple = ipos 0 and iquat identity on a childless free body: M0 is the crb diagonal, every shortcut is exact
        b = m.dof_bodyid[d]
        if not (m.jnt_type[m.dof_jntid[d]] == mujoco.mjtJoint.mjJNT_FREE and m.tree_dofnum[m.dof_treeid[d]] == 6 and m.body_subtreemass[b] == m.body_mass[b]):
            raise SystemExit('FAIL export: simple dofs other than a standalone free body are not ported')
    if (m.nflex or m.opt.integrator not in INTEGRATORS or m.opt.solver not in SOLVERS or m.actuator_damping.any()
            or m.dof_dampingpoly.any() or m.jnt_stiffnesspoly.any() or m.opt.disableflags & ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)):  # multiccd off only changes contact counts (with_auto_pairs)
        raise SystemExit('FAIL export: only Euler/implicitfast, linear joint springs and dof dampers are ported (no flex, '
                         'actuator damping, polynomial springs or dampers, disable flags)')
    if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
        raise SystemExit('FAIL export: the elliptic cone is gated with the Euler integrator only (check.sh cone), not RK4 or implicitfast')
    if m.tendon_frictionloss.any() or m.tendon_stiffnesspoly.any() or m.tendon_dampingpoly.any() or m.tendon_actfrclimited.any():
        raise SystemExit('FAIL export: only tendons without friction loss, with linear spring and damper, are ported (limits: eq_rows)')
    return m


def fluid(m):
    """MuJoCo mj_fluid block (engine_passive.c:842), appended last: ints per ellipsoid row (body, geom) then the inertia-box
    bodies (every body with mass that has no ellipsoid geom); floats opt.density, opt.viscosity, opt.wind, then per ellipsoid
    row the geom's semi-axes (mju_geomSemiAxes) and its 12 geom_fluid coefficients. Rows are geoms with geom_fluid[0] > 0;
    a body with any such geom takes the ellipsoid model only. Empty for a model without density and viscosity."""
    if not (m.opt.density or m.opt.viscosity):
        return None
    G = mujoco.mjtGeom
    rows = np.nonzero(m.geom_fluid[:, 0] > 0)[0]
    axes = []
    for g in rows:
        s, t = m.geom_size[g], m.geom_type[g]
        axes.append([s[0]] * 3 if t == G.mjGEOM_SPHERE else [s[0], s[0], s[1] + s[0]] if t == G.mjGEOM_CAPSULE
                    else [s[0], s[0], s[1]] if t == G.mjGEOM_CYLINDER else list(s))
    ell = {m.geom_bodyid[g] for g in rows}
    box = [b for b in range(m.nbody) if m.body_mass[b] >= 1e-15 and b not in ell]
    ints = np.array([v for g in rows for v in (m.geom_bodyid[g], g)] + box)
    floats = np.concatenate([[m.opt.density, m.opt.viscosity], m.opt.wind] + [np.r_[a, m.geom_fluid[g]] for a, g in zip(axes, rows)])
    return ints, floats, {'NFLROW': len(rows), 'NFLBOX': len(box)}


def tendons(m):
    """Fixed tendons (mj_tendon: length = sum coef q, J = coef, constant) plus mj_passive's tendon spring and damper
    (engine_passive.c:775): ints per tendon wrap start and count, then each wrap's dof address, then each wrap's qpos
    address; floats each wrap's coef, then per tendon stiffness, damping, lengthspring 2. Joints are hinge or slide only.
    New wrap kinds (spatial tendons) append their own arrays after these."""
    if not m.ntendon:
        return None
    joint = m.wrap_type == mujoco.mjtWrap.mjWRAP_JOINT  # spatial wraps (site, geom, pulley) carry no joint: placeholder 0, see spatial()
    obj = np.where(joint, m.wrap_objid, 0)
    if (m.jnt_type[obj][joint] > 3).any() or (m.jnt_type[obj][joint] < 2).any():
        raise SystemExit('FAIL export: tendon wraps only on hinge and slide joints')
    ints = np.concatenate([m.tendon_adr, m.tendon_num, np.where(joint, m.jnt_dofadr[obj], 0), np.where(joint, m.jnt_qposadr[obj], 0)])
    floats = np.concatenate([m.wrap_prm, m.tendon_stiffness, m.tendon_damping, m.tendon_lengthspring.ravel()])
    return ints, floats, {'NTEN': m.ntendon, 'NWRAP': m.nwrap}


def spatial(m):
    """Spatial tendons (mj_tendon, engine_core_smooth.c:922; wrap geometry mju_wrap, engine_util_misc.c:281), appended right after
    tendons(): ints per wrap its type (mjtWrap), its body (site: the site's; sphere, cylinder: the geom's) and its sidesite's body
    (-1: none), then per tendon the body whose floating origin frames the whole path; floats per wrap 11: local position (site or
    geom), local quaternion (geom), radius, sidesite local position. Pulley divisors, sidesite ids and the fixed tendons' coefs are
    already in tendons()' wrap_prm. A path may join two trees: every point and root com is moved into the frame of moving[0]'s tree."""
    W = mujoco.mjtWrap
    if not m.ntendon or (m.wrap_type == W.mjWRAP_JOINT).all():
        return None
    body = np.zeros(m.nwrap, int)
    side = np.full(m.nwrap, -1)
    fl = np.zeros((m.nwrap, 11))
    for w in range(m.nwrap):
        t, o = m.wrap_type[w], m.wrap_objid[w]
        if t == W.mjWRAP_SITE:
            body[w], fl[w, :3] = m.site_bodyid[o], m.site_pos[o]
        elif t in (W.mjWRAP_SPHERE, W.mjWRAP_CYLINDER):
            body[w], fl[w, :3], fl[w, 3:7], fl[w, 7] = m.geom_bodyid[o], m.geom_pos[o], m.geom_quat[o], m.geom_size[o][0]
            sid = int(round(m.wrap_prm[w]))
            if sid >= 0:
                side[w], fl[w, 8:11] = m.site_bodyid[sid], m.site_pos[sid]
    frame = []
    for i in range(m.ntendon):
        ws = range(m.tendon_adr[i], m.tendon_adr[i] + m.tendon_num[i])
        bs = [int(b) for w in ws for b in (body[w], side[w]) if b >= 0 and m.wrap_type[w] != W.mjWRAP_JOINT]
        moving = [b for b in bs if any(m.body_dofnum[a] for a in _ancestors(m, b))]
        frame.append(moving[0] if moving else 0)
    return np.concatenate([m.wrap_type, body, side, frame]), fl.ravel(), {'SPATIAL': 1}


def _ancestors(m, b):
    """b and its ancestor bodies up to (not including) the world."""
    out = []
    while b > 0:
        out.append(b)
        b = m.body_parentid[b]
    return out


def cross_trees(m, regex, convex):
    """Couplings across two moving trees the export would carry: pairs, equalities and tendons whose moving bodies
    span more than one body_rootid. A body is moving when an ancestor carries a dof."""
    def moving(b):
        return any(m.body_dofnum[a] > 0 for a in _ancestors(m, b))
    out = []
    if regex is not None:
        for p in pairs(m, regex, convex)[0]:
            g1, g2 = pair_geoms(m, p)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if moving(b1) and moving(b2) and m.body_rootid[b1] != m.body_rootid[b2]:
                out.append(f'pair {p}')
    E, O = mujoco.mjtEq, mujoco.mjtObj
    for i in range(m.neq):  # inactive ones too: the host may switch them on (eq_active)
        t = int(m.eq_type[i])
        if t in (E.mjEQ_CONNECT, E.mjEQ_WELD):
            if m.eq_objtype[i] == O.mjOBJ_SITE:
                b1, b2 = int(m.site_bodyid[m.eq_obj1id[i]]), int(m.site_bodyid[m.eq_obj2id[i]])
            else:
                b1, b2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            if moving(b1) and moving(b2) and m.body_rootid[b1] != m.body_rootid[b2]:
                out.append(f'equality {i}')
        elif t == E.mjEQ_JOINT:
            j1, j2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            if j2 < 0:
                continue
            b1, b2 = int(m.jnt_bodyid[j1]), int(m.jnt_bodyid[j2])
            if moving(b1) and moving(b2) and m.body_rootid[b1] != m.body_rootid[b2]:
                out.append(f'equality {i}')
    W, troots = mujoco.mjtWrap, []
    for i in range(m.ntendon):
        roots = set()
        for w in range(m.tendon_adr[i], m.tendon_adr[i] + m.tendon_num[i]):
            t, o = m.wrap_type[w], int(m.wrap_objid[w])
            if t == W.mjWRAP_PULLEY:
                continue
            bodies = []
            if t == W.mjWRAP_JOINT:
                bodies.append(int(m.jnt_bodyid[o]))
            elif t == W.mjWRAP_SITE:
                bodies.append(int(m.site_bodyid[o]))
            elif t in (W.mjWRAP_SPHERE, W.mjWRAP_CYLINDER):
                bodies.append(int(m.geom_bodyid[o]))
                sid = int(round(m.wrap_prm[w]))
                if sid >= 0:
                    bodies.append(int(m.site_bodyid[sid]))
            for b in bodies:
                if moving(b):
                    roots.add(int(m.body_rootid[b]))
        if len(roots) > 1:
            out.append(f'tendon {i}')
        troots.append(roots)
    for i in range(m.neq):  # a tendon equality couples its two tendons' trees
        o1, o2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
        if m.eq_type[i] == E.mjEQ_TENDON and o2 >= 0 and len(troots[o1] | troots[o2]) > 1:
            out.append(f'equality {i}')
    return out


def pairs(m, regex, convex=False):
    """Exported contact pairs: geom2 named by regex, of a ported kind (plane or heightfield vs mesh, sphere vs sphere; with
    convex also box or ellipsoid vs mesh, GJK + EPA). Returns (pair ids, {unported kind: count}); the caller must say what it drops."""
    import re
    from collections import Counter
    G = mujoco.mjtGeom
    ported = {(G.mjGEOM_PLANE, G.mjGEOM_MESH), (G.mjGEOM_HFIELD, G.mjGEOM_MESH), (G.mjGEOM_SPHERE, G.mjGEOM_SPHERE)}
    if convex:
        ported |= {(G.mjGEOM_BOX, G.mjGEOM_MESH), (G.mjGEOM_ELLIPSOID, G.mjGEOM_MESH), (G.mjGEOM_MESH, G.mjGEOM_MESH)} | {(a, b) for a in PRIM for b in PRIM if int(a) <= int(b)}
    kind = lambda i: tuple(G(m.geom_type[g]) for g in pair_geoms(m, i))
    if os.environ.get('KINDS'):  # diagnostics: export only these pair kinds, e.g. KINDS=capsule-cylinder,sphere-capsule
        ported = {k for k in ported if '-'.join(g.name[7:].lower() for g in k) in os.environ['KINDS'].split(',')}
    sel = [i for i in range(m.npair) if re.search(regex, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, pair_geoms(m, i)[1]))]
    return [i for i in sel if kind(i) in ported], Counter('-'.join(g.name[7:].lower() for g in kind(i)) for i in sel if kind(i) not in ported)


def hpattern(m, couple):
    """Sparsity of H = M + J'DJ for mj_factorI: rows of M plus every pair's coupled dofs, filled by backward
    elimination. Row k = k's chain in the elimination tree (parent = largest column left after elimination), a
    superset of the fill, so each row's pattern is a prefix of the rows below it, as factorI requires."""
    rows = [set() for _ in range(m.nv)]
    for v in range(m.nv):
        p = m.dof_parentid[v]
        while p >= 0:
            rows[v].add(p)
            p = m.dof_parentid[p]
    for u in couple:
        for a in u:
            rows[a] |= {c for c in u if c < a}
    par = [-1] * m.nv
    for k in range(m.nv - 1, -1, -1):
        if rows[k]:
            par[k] = max(rows[k])
            rows[par[k]] |= rows[k] - {par[k]}
    hrow = []
    for k in range(m.nv):
        c, p = [k], par[k]
        while p >= 0:
            c, p = [p] + c, par[p]
        hrow.append(c)
    return hrow


def feet(m, regex, couple=(), hrows=None, convex=False, hfexact=False, hull_out=None):
    """Contact pairs (pairs()): ints, floats and -D sizes for smooth.mojo STAGE 6.

    ints   per pair: geom1, geom2, geom2's body, hull vertex count (0: sphere-sphere), hull vertex offset,
           edge-address offset, chain length, chain offset, contact slot offset, slot count (3 mesh, 1 sphere),
           Jacobian offset (4 rows x slots x chain) (11); then every hull vertex's first edge (absolute, into the
           edge list), the edge lists (hull-local vertex ids, -1 ends a vertex), the dof chains (H's row pattern of the
           pair's deepest dof, ascending), their Jacobian signs (+1 geom2's body only, -1 geom1's only, 0 both or
           neither), H's pattern (rowadr, rownnz, colind as hpattern()), M's entry -> H's entry, opt.iterations,
           opt.ls_iterations, opt.noslip_iterations
    couple: extra dof sets that H must couple (equality constraints, eqs()); hrows, if a list, receives H's row patterns;
    hull_out, if a list, receives the hull table {mesh id: (vertex offset, first-edge offset)} (convex_block: geom1 of a mesh-mesh pair)
    floats per pair: margin, gap, friction 0-1, solref 2, solimp 5, invweight0 translation (both bodies), rbound of
           geom2 and geom1 (14); then hull vertices in the geom frame; then opt.impratio, opt.tolerance, opt.ls_tolerance, opt.noslip_tolerance, stat.meaninertia
    """
    elliptic = m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    if m.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON and (elliptic or convex or hfexact):  # CG and PGS: pyramidal rows of plane-mesh and sphere pairs (check.sh solver)
        raise SystemExit('FAIL export: CG and PGS are gated with the pyramidal cone and plane-mesh / sphere pairs only (check.sh solver)')
    if (m.geom_contype & m.geom_conaffinity).any():
        raise SystemExit('FAIL export: contacts need explicit pairs only')
    sel, drop = pairs(m, regex, convex)
    if drop:
        print(f'export: {sum(drop.values())} pairs of unported kinds NOT exported: {dict(drop)}')
    if convex and any(m.geom_type[pair_geoms(m, i)[0]] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH) and m.geom_type[pair_geoms(m, i)[1]] == mujoco.mjtGeom.mjGEOM_MESH
                      and m.pair_margin[i] + m.pair_gap[i] <= 0 for i in sel):  # mjc_Convex maxContacts: margin 0 takes native polygon clipping
        raise SystemExit('FAIL export: box-mesh and mesh-mesh pairs need margin > 0 (MuJoCo clips polygons at margin 0; not ported)')
    if any(m.pair_dim[i] not in (1, 3) or m.pair_solimp[i][4] != int(m.pair_solimp[i][4]) for i in sel):
        raise SystemExit('FAIL export: only condim 1 and 3 contacts with a whole solimp power are ported')
    if not convex and any(m.pair_dim[i] == 1 for i in sel):
        raise SystemExit('FAIL export: condim 1 pairs ride the convex block (CONVEX=1, check.sh flight_contacts)')

    def anc(b):
        c = set()
        while b > 0:
            c |= set(range(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b]))
            b = m.body_parentid[b]
        return c
    two = [(anc(m.geom_bodyid[pair_geoms(m, i)[0]]), anc(m.geom_bodyid[pair_geoms(m, i)[1]])) for i in sel]
    hrow = hpattern(m, [a ^ b for a, b in two] + list(couple))
    if hrows is not None:
        hrows.append(hrow)
    head, first, edges, chains, signs, fl, verts, hulls = [], [], [], [], [], [], [], {}
    nslot = nje = poolch = 0
    pool = [0, 0, 0]  # per kind (convex, exact heightfield, primitive pairs): slots its pairs could take in one step
    for i, (a1, a2) in zip(sel, two):
        g1, g2 = pair_geoms(m, i)
        b = m.geom_bodyid[g2]
        chain = hrow[max(a1 ^ a2)]
        nv = 0
        for g in (g2, g1) if m.geom_type[g1] == m.geom_type[g2] == mujoco.mjtGeom.mjGEOM_MESH else (g2,):  # mesh-mesh: geom1's hull too
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[g]
            ga = m.mesh_graphadr[mesh]
            nvg = m.mesh_graph[ga]
            nv = nv or nvg
            edgeadr = m.mesh_graph[ga + 2:ga + 2 + nvg]
            gid = m.mesh_graph[ga + 2 + nvg:ga + 2 + 2 * nvg]
            local = m.mesh_graph[ga + 2 + 2 * nvg:]
            local = local[:edgeadr.max() + list(local[edgeadr.max():]).index(-1) + 1]  # through the last vertex's -1
            if mesh not in hulls:  # one copy of each hull however many planes it meets (the cube: 6 pairs per part)
                hulls[mesh] = (len(verts) // 3, len(first))
                first += list(len(edges) + edgeadr)
                edges += list(local)
                verts += list(m.mesh_vert[m.mesh_vertadr[mesh] + gid].ravel())
        prim = prim_pair(m, i) if convex else 0
        meshmesh = convex and m.geom_type[g1] == m.geom_type[g2] == mujoco.mjtGeom.mjGEOM_MESH  # mjc_Convex, perturbation search: up to 5
        ns = (5 if m.geom_type[g1] == mujoco.mjtGeom.mjGEOM_BOX else 1) if m.geom_type[g1] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID) else (5 if meshmesh else 3 if nv else 1)
        if prim:  # mj_maxContact: capsule-capsule 2 (both ends when parallel); capsule/cylinder pairs up to 5 (mjc_Convex's perturbation search, multiccd); one with a sphere or ellipsoid
            G_ = mujoco.mjtGeom
            ns = 2 if prim == 5 else (5 if prim == 4 and {G_(m.geom_type[g1]), G_(m.geom_type[g2])} <= {G_.mjGEOM_CAPSULE, G_.mjGEOM_CYLINDER} else 1)
        if hfexact and m.geom_type[g1] == mujoco.mjtGeom.mjGEOM_HFIELD:
            ns = HF_SLOTS  # one contact per cell-triangle prism under the mesh (MuJoCo keeps up to 50)
        G = mujoco.mjtGeom
        pooled = (convex and m.geom_type[g1] in (G.mjGEOM_BOX, G.mjGEOM_ELLIPSOID, G.mjGEOM_MESH) and m.geom_type[g2] == G.mjGEOM_MESH) or (hfexact and m.geom_type[g1] == G.mjGEOM_HFIELD) or bool(prim)
        if pooled:  # convex.mojo pairs take slots from the animal's pool each step: no static slots, the record's ns is the cap
            head += [g1, g2, b, nv, *(hulls[m.geom_dataid[g2]] if nv else (0, 0)), len(chain), len(chains), 0, ns, 0]
            pool[2 if prim else m.geom_type[g1] == G.mjGEOM_HFIELD] += ns
            poolch = max(poolch, len(chain))
        else:
            head += [g1, g2, b, nv, *(hulls[m.geom_dataid[g2]] if nv else (0, 0)), len(chain), len(chains), nslot, ns, nje]
            nslot, nje = nslot + ns, nje + 4 * ns * len(chain)
        chains += chain
        signs += [(c in a2) - (c in a1) for c in chain]
        sr = m.pair_solref[i] if (m.pair_solref[i][0] > 0) == (m.pair_solref[i][1] > 0) else (0.02, 1.0)  # mixed format: MuJoCo's default (getsolparam)
        fl += [m.pair_margin[i], m.pair_gap[i], *m.pair_friction[i][:2], *sr, *m.pair_solimp[i],
               m.body_invweight0[m.geom_bodyid[g1], 0] + m.body_invweight0[b, 0], m.geom_rbound[g2], m.geom_rbound[g1]]
    if hull_out is not None:
        hull_out.append(hulls)
    hnnz = [len(r) for r in hrow]
    hadr = list(np.cumsum([0] + hnnz[:-1]))
    hcol = [c for r in hrow for c in r]
    mtoh = [hadr[v] + hrow[v].index(m.M_colind[a]) for v in range(m.nv) for a in range(m.M_rowadr[v], m.M_rowadr[v] + m.M_rownnz[v])]
    ints = np.array(head + first + edges + chains + signs + hadr + hnnz + hcol + mtoh + [m.opt.iterations, m.opt.ls_iterations, m.opt.noslip_iterations])
    floats = fl + verts + [m.opt.impratio, m.opt.tolerance, m.opt.ls_tolerance, m.opt.noslip_tolerance, m.stat.meaninertia]
    npool = min(pool[0], CONVEX_POOL) + min(pool[1], HF_POOL) + min(pool[2], PRIM_POOL)  # the pool is shared per animal; pairs beyond it drop contacts (the gates' ncon shows it)
    jpool = npool * 4 * poolch
    sizes = {'NCP': len(sel), 'NHV': len(verts) // 3, 'NHE': len(edges), 'NCH': len(chains), 'NH': len(hcol), 'NSLOT': nslot + npool, 'NJE': nje + jpool}
    if npool:
        sizes |= {'NPOOL': npool, 'JPOOL': jpool}
    if elliptic:
        sizes['ELLIPTIC'] = 1
    hf = sorted({m.pair_geom1[i] for i in sel if m.geom_type[m.pair_geom1[i]] == mujoco.mjtGeom.mjGEOM_HFIELD})
    if hf:  # terrain: the heightfield is MuJoCo's sample; the engine evaluates world.py's function from its 8 params
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, 'terrain')
        if len(hf) > 1 or t < 0 or m.numeric_size[t] != 16:
            raise SystemExit('FAIL export: one heightfield ground, with its 16 world params in <numeric name="terrain">')
        floats += list(m.numeric_data[m.numeric_adr[t]:m.numeric_adr[t] + 16])
        sizes |= {'HFGEOM': hf[0]}
    floats = np.array(floats)
    return ints, floats, sizes


EQH, EQF = 13, 31  # eq_block: ints and floats per equality constraint


def eq_rows(m, limits=True):
    """Every equality constraint of the ported kinds (connect, weld, joint, tendon), active or not (act: MuJoCo eq_active0, the
    value d->eq_active resets to; the host may switch any of them per animal, -DEQACT=1), as dicts for eq_block.
    MuJoCo engine_core_constraint.c:513 mj_instantiateEquality: anchors by body semantic (connect: data[0:3] on obj1,
    data[3:6] on obj2; weld: data[3:6] on obj1, data[0:3] on obj2, relpose data[6:10], torquescale data[10]) or by site
    semantic (site_pos, site_quat of both); both reduce to a local anchor, a local quaternion and a body per side."""
    E, O = mujoco.mjtEq, mujoco.mjtObj
    def anc(b):
        c = set()
        while b > 0:
            c |= set(range(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b]))
            b = m.body_parentid[b]
        return c
    if m.neq and (m.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON or m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL or m.opt.integrator):
        raise SystemExit('FAIL export: equality constraints are gated with the Newton solver and the pyramidal cone only '
                         'and the Euler integrator only (check.sh equality, equality_contact); CG, PGS, elliptic cones and implicit integrators with equalities are untested')
    if m.dof_frictionloss.any():
        raise SystemExit('FAIL export: dof friction loss (constraint rows not ported)')
    W = mujoco.mjtWrap

    def ten_union(t):  # dofs tendon t moves: its joints' dofs, its sites' and wrap geoms' body chains
        union = set()
        for w in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
            wt, o = m.wrap_type[w], int(m.wrap_objid[w])
            if wt == W.mjWRAP_JOINT:
                union.add(int(m.jnt_dofadr[o]))
            elif wt == W.mjWRAP_SITE:
                union |= anc(m.site_bodyid[o])
            elif wt in (W.mjWRAP_SPHERE, W.mjWRAP_CYLINDER):
                union |= anc(m.geom_bodyid[o])
        return union
    out = []
    for i in range(m.neq):
        t, o1, o2, d = int(m.eq_type[i]), int(m.eq_obj1id[i]), int(m.eq_obj2id[i]), m.eq_data[i]
        e = dict(type=t, act=int(m.eq_active0[i]), solref=m.eq_solref[i], solimp=m.eq_solimp[i], data=d[:5], qref=[0.0, 0.0], dof=[-1, -1], qadr=[-1, -1])
        if t in (E.mjEQ_CONNECT, E.mjEQ_WELD):
            site = m.eq_objtype[i] == O.mjOBJ_SITE
            b0, b1 = (m.site_bodyid[o1], m.site_bodyid[o2]) if site else (o1, o2)
            if site:
                a0, a1, s0, s1 = m.site_pos[o1], m.site_pos[o2], m.site_quat[o1], m.site_quat[o2]
            elif t == E.mjEQ_CONNECT:
                a0, a1, s0, s1 = d[0:3], d[3:6], [1, 0, 0, 0], [1, 0, 0, 0]
            else:
                a0, a1, s0, s1 = d[3:6], d[0:3], d[6:10], [1, 0, 0, 0]
            e |= dict(b=[int(b0), int(b1)], size=3 if t == E.mjEQ_CONNECT else 6, union=anc(b0) | anc(b1), a=[*a0, *a1], s=[*s0, *s1],
                      ts=float(d[10]), invw=[m.body_invweight0[b0, 0] + m.body_invweight0[b1, 0], m.body_invweight0[b0, 1] + m.body_invweight0[b1, 1]],
                      in0=anc(b0), in1=anc(b1), dof=[int(b0 if anc(b0) else b1), -1])  # dof 0: the body whose floating origin frames both anchors
        elif t == E.mjEQ_JOINT:
            js = [o1] + ([o2] if o2 >= 0 else [])
            if any(m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for j in js):
                raise SystemExit(f'FAIL export: joint equality {i} on a free or ball joint')
            e |= dict(b=[0, 0], size=1, union={int(m.jnt_dofadr[j]) for j in js}, a=[0.0] * 6, s=[1, 0, 0, 0, 1, 0, 0, 0], ts=1.0,
                      invw=[sum(m.dof_invweight0[m.jnt_dofadr[j]] for j in js), 0.0], in0=set(), in1=set(),
                      dof=[int(m.jnt_dofadr[j]) for j in js] + [-1] * (2 - len(js)), qadr=[int(m.jnt_qposadr[j]) for j in js] + [-1] * (2 - len(js)),
                      qref=[float(m.qpos0[m.jnt_qposadr[j]]) for j in js] + [0.0] * (2 - len(js)))
        elif t == E.mjEQ_TENDON:  # type 5 (3 and 4 are the limit rows): length 0 - length0 0 - poly(length 1 - length0 1), J = J0 - deriv J1
            ts_ = [o1] + ([o2] if o2 >= 0 else [])
            e |= dict(type=5, b=[0, 0], size=1, union=set().union(*map(ten_union, ts_)), a=[0.0] * 6, s=[1, 0, 0, 0, 1, 0, 0, 0], ts=1.0,
                      invw=[sum(float(m.tendon_invweight0[k]) for k in ts_), 0.0], in0=set(), in1=set(),
                      dof=ts_ + [-1] * (2 - len(ts_)), qref=[float(m.tendon_length0[k]) for k in ts_] + [0.0] * (2 - len(ts_)))
        else:
            raise SystemExit(f'FAIL export: equality {i} of type {E(t).name} is not ported (connect, weld, joint, tendon)')
        if not e['union']:
            raise SystemExit(f'FAIL export: equality {i} constrains nothing movable')
        out.append(e)
    lim = np.flatnonzero(m.jnt_limited) if limits else []
    if len(lim) and (m.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON or m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL or m.opt.integrator):
        print(f'export: {len(lim)} joint limits NOT exported: their rows are gated with the Newton solver, the pyramidal cone and Euler only (check.sh joint_limit)')
        lim = []
    for j in lim:  # mj_instantiateLimit joint rows (engine_core_constraint.c), hinge and slide: dist = side (range - qpos), J = -side on the dof
        if m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise SystemExit(f'FAIL export: joint {j} has a limit on a ball or free joint (only hinge and slide limits are ported)')
        dof = int(m.jnt_dofadr[j])
        for side in (-1, 1):
            out.append(dict(type=4, solref=m.jnt_solref[j], solimp=m.jnt_solimp[j], b=[0, 0], size=1, union={dof}, a=[0.0] * 6,
                            s=[1, 0, 0, 0, 1, 0, 0, 0], ts=1.0, invw=[float(m.dof_invweight0[dof]), 0.0], in0=set(), in1=set(),
                            dof=[dof, side], qadr=[int(m.jnt_qposadr[j]), -1], data=[float(m.jnt_range[j][(side + 1) // 2]), float(m.jnt_margin[j]), 0.0, 0.0, 0.0], qref=[0.0, 0.0]))
    for t in np.flatnonzero(m.tendon_limited):  # mj_instantiateLimit tendon rows (engine_core_constraint.c:1277), one per side, unilateral
        if m.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON or m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL or m.opt.integrator:
            raise SystemExit('FAIL export: tendon limits are gated with the Newton solver, the pyramidal cone and the Euler integrator only (check.sh tendon_limit)')
        union = ten_union(t)
        if not union:
            raise SystemExit(f'FAIL export: limited tendon {t} moves nothing')
        for side in (-1, 1):  # dist = side (range - length): lower bound at side -1, upper at +1
            out.append(dict(type=3, solref=m.tendon_solref_lim[t], solimp=m.tendon_solimp_lim[t], b=[0, 0], size=1, union=union, a=[0.0] * 6,
                            s=[1, 0, 0, 0, 1, 0, 0, 0], ts=1.0, invw=[float(m.tendon_invweight0[t]), 0.0], in0=set(), in1=set(),
                            dof=[int(t), side], qadr=[-1, -1], data=[float(m.tendon_range[t][(side + 1) // 2]), float(m.tendon_margin[t]), 0.0, 0.0, 0.0], qref=[0.0, 0.0]))
    return out


def eq_block(m, eq, hrow):
    """Equality block (last in mi/mf): ints per constraint (type, body 0, body 1, rows, chain length, chain offset, first
    row, Jacobian offset, dof 0, dof 1, qpos address 0, 1; connect and weld: the body whose floating origin is the frame; eq_active0, 1 for
    a limit row), then every chain (H's row pattern of the deepest dof, so
    J' D J lands on H's rows by chain position, as the contact pairs do) and per chain dof a mask (1: in body 0's
    chain, 2: body 1's); floats per constraint: local anchors 0 and 1, local quaternions 0 and 1, torquescale, solref 2,
    solimp 5, inverse weight translation and rotation (joint: dof), joint polynomial data 5, joint qpos0 of both."""
    J = mujoco.mjtJoint
    is_quat = lambda c: m.jnt_type[m.dof_jntid[c]] == J.mjJNT_BALL or (m.jnt_type[m.dof_jntid[c]] == J.mjJNT_FREE and c >= m.jnt_dofadr[m.dof_jntid[c]] + 3)
    head, chains, masks, fl = [], [], [], []
    nrow = nje = 0
    for e in eq:
        chain = hrow[max(e['union'])]
        head += [e['type'], *e['b'], e['size'], len(chain), len(chains), nrow, nje, *e['dof'], *e['qadr'], e.get('act', 1)]
        nrow, nje = nrow + e['size'], nje + e['size'] * len(chain)
        chains += chain
        masks += [(c in e['in0']) + 2 * (c in e['in1']) + 4 * is_quat(c) for c in chain]
        sr = e['solref'] if (e['solref'][0] > 0) == (e['solref'][1] > 0) else (0.02, 1.0)  # mixed format: MuJoCo's default (getsolparam); direct (-k, -b) kept
        fl += [*e['a'], *e['s'], e['ts'], *sr, *e['solimp'], *e['invw'], *e['data'], *e['qref']]
    if any(a['type'] in (3, 4) and b['type'] not in (3, 4) for a, b in zip(eq, eq[1:])):
        raise SystemExit('FAIL export: limit rows (joint, tendon) must follow the bilateral equality rows')
    return (np.array(head + chains + masks), np.array(fl),
            {'NEQ': len(eq), 'NEQR': nrow, 'NEQB': sum(e['size'] for e in eq if e['type'] not in (3, 4)), 'NEQCH': len(chains), 'NEQJ': nje})


def mocap_block(m):
    """Mocap bodies (engine_core_smooth.c:86): body_mocapid per body, last block. A mocap body is a child of the world with
    no joint (MuJoCo's own rule); its pose arrives per animal in ctrl.bin rows nu.. (pos 3, quat 4 per mocap body)."""
    if not m.nmocap:
        return None
    return np.asarray(m.body_mocapid), np.zeros(0), {'NMOCAP': m.nmocap}


def with_inactive(mjcf, out, every=2):
    """mjcf with every `every`-th equality (from the first) inactive at start (active="false", eq_active0 0), mesh paths made absolute;
    written to `out`; returns the path. The gate bodies of check.sh eqact_runtime: the export carries them, the host may switch them on."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    base = Path(mjcf).parent / (spec.meshdir or '')
    for mesh in spec.meshes:
        mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
    spec.meshdir = ''
    for i, e in enumerate(spec.equalities):
        e.active = i % every != 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def with_tendon_limits(mjcf, out, count=256, lo=0.35, hi=0.65, solref=0.02, tolerance=None):
    """mjcf with every tendon limited to [lo, hi] quantiles of its length over nearby() states (so random states violate each side in
    about a third of them), margin a tenth of that span, solref_lim (solref, 1) and solimp_lim default, opt.tolerance replaced when given
    (0: Newton to convergence, as elliptic()); mesh paths made absolute; written to `out`; returns the path. The gate bodies of check.sh
    tendon_limit: solref 0.02 (MuJoCo's default, stiff: the rows' forces are D (J qacc - aref) with aref 1e3 x the difference, float32
    leaves qfrc_constraint within 2.4e-4 of its max on the CPU, 3.9e-4 on the card, bar 5e-4) and solref 0.1 with tolerance 0 (within 6e-6, the default bar)."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    base = Path(mjcf).parent / (spec.meshdir or '')
    for mesh in spec.meshes:
        mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
    spec.meshdir = ''
    m0 = spec.compile()
    q, v = nearby(m0, count)
    d = mujoco.MjData(m0)
    L = np.zeros((count, m0.ntendon))
    for n in range(count):
        d.qpos[:] = q[n]
        mujoco.mj_forward(m0, d)
        L[n] = d.ten_length
    for g in spec.geoms:  # the stage runs contacts off; a body with collision geoms would need explicit pairs at export
        g.contype = g.conaffinity = 0
    for j in spec.joints:  # tendon limits alone: joint limit rows have their own stage (check.sh joint_limit)
        j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    for t, ten in enumerate(spec.tendons):
        a, b = np.quantile(L[:, t], lo), np.quantile(L[:, t], hi)
        ten.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        ten.range = [a, b]
        ten.margin = 0.1 * (b - a)
        ten.solref_limit = [solref, 1.0]
    if tolerance is not None:
        spec.option.tolerance = tolerance
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def with_tendon_equalities(mjcf, out, count=256, solref=0.1, tolerance=None, fixed=False):
    """mjcf with two tendon equality constraints (mjEQ_TENDON): the first tendon coupled to the last by MuJoCo's quartic in the last's
    length change (polycoef 0.1 s0, 0.8, 0.5 / s1, -0.3 / s1^2, 0.2 / s1^3, s the tendon's length spread about length0 over nearby()
    states, so every term moves the residual), and the second tendon alone at an offset of 0.1 s (one object). fixed=True first adds
    two fixed tendons over hinge/slide joints (on a body without spatial tendons: the engine's fixed tendon path). solref (solref, 1),
    contacts, joint and tendon limits off, opt.tolerance replaced when given, mesh paths made absolute; written to `out`; returns the path."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    base = Path(mjcf).parent / (spec.meshdir or '')
    for mesh in spec.meshes:
        mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
    spec.meshdir = ''
    for g in spec.geoms:
        g.contype = g.conaffinity = 0
    for j in spec.joints:
        j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    for ten in spec.tendons:
        ten.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    if fixed:
        m0 = spec.compile()
        hs = [m0.joint(j).name for j in range(m0.njnt) if m0.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and m0.joint(j).name]
        pick = [hs[k * len(hs) // 4] for k in range(4)]
        for name, parts in (('tfix0', ((pick[0], 1.0), (pick[1], -0.5))), ('tfix1', ((pick[2], 0.7), (pick[3], 0.3)))):
            t = spec.add_tendon(name=name)
            for jn, c in parts:
                t.wrap_joint(jn, c)
    m0 = spec.compile()
    if m0.ntendon < 2:
        raise SystemExit(f'FAIL with_tendon_equalities: {mjcf} has {m0.ntendon} tendons, needs 2')
    q, v = nearby(m0, count)
    d = mujoco.MjData(m0)
    L = np.zeros((count, m0.ntendon))
    for n in range(count):
        d.qpos[:] = q[n]
        mujoco.mj_forward(m0, d)
        L[n] = d.ten_length
    s = np.maximum(np.sqrt(((L - m0.tendon_length0) ** 2).mean(0)), 1e-9)
    names = [m0.tendon(t).name for t in range(m0.ntendon)]
    a, b, c = 0, m0.ntendon - 1, 1
    for n1, n2, data in ((names[a], names[b], [0.1 * s[a], 0.8, 0.5 / s[b], -0.3 / s[b] ** 2, 0.2 / s[b] ** 3]), (names[c], None, [0.1 * s[c], 0, 0, 0, 0])):
        e = spec.add_equality(type=mujoco.mjtEq.mjEQ_TENDON, objtype=mujoco.mjtObj.mjOBJ_TENDON, name1=n1)
        if n2:
            e.name2 = n2
        e.data = data + [0.0] * 6
        e.solref = [solref, 1.0]
    if tolerance is not None:
        spec.option.tolerance = tolerance
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def with_joint_limits(mjcf, out, count=256, lo=0.35, hi=0.65, solref=0.02, tolerance=None):
    """mjcf with every hinge and slide joint limited to [lo, hi] quantiles of its qpos over nearby() states, margin a tenth of the
    span, solref (solref, 1), contacts and tendon limits off, as with_tendon_limits; the gate bodies of check.sh joint_limit."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    base = Path(mjcf).parent / (spec.meshdir or '')
    for mesh in spec.meshes:
        mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
    spec.meshdir = ''
    m0 = spec.compile()
    q, _ = nearby(m0, count)
    for g in spec.geoms:
        g.contype = g.conaffinity = 0
    for ten in spec.tendons:
        ten.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    for j in spec.joints:
        jj = m0.joint(j.name).id if j.name else None
        if jj is None or m0.jnt_type[jj] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        a, b = np.quantile(q[:, m0.jnt_qposadr[jj]], lo), np.quantile(q[:, m0.jnt_qposadr[jj]], hi)
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.range = [a, b]
        j.margin = 0.1 * (b - a)
        j.solref_limit = [solref, 1.0]
    if tolerance is not None:
        spec.option.tolerance = tolerance
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def with_sleep(mjcf, out, tolerance):
    """mjcf with sleeping enabled (opt.enableflags sleep, sleep_tolerance): the gate body of check.sh sleep. Mesh paths made absolute;
    written to `out`; returns the path."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    base = Path(mjcf).parent / (spec.meshdir or '')
    for mesh in spec.meshes:
        mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
    spec.meshdir = ''
    spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    spec.option.sleep_tolerance = tolerance
    m0 = spec.compile()  # the compiler's auto policy forbids sleep on an actuated tree (the fly compiles to AUTO_NEVER): allow it on every root
    for b in np.flatnonzero(m0.body_parentid[1:] == 0) + 1:
        spec.body(mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, int(b))).sleep = mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def auto_pairs(m):
    """MuJoCo's automatic contact pairs of a model as explicit pair data (mj_collision's static filters, engine_collision_driver.c: same
    body, the parent filter on weld bodies, the exclude list, the contype/conaffinity bitmask; mj_contactParam's mixing: priority, else
    condim max, solref and solimp by solmix, friction max per component; margin and gap are the geoms' sums). geom1 has the lower
    type, as mj_collideGeoms orders a contact. The broadphase (bounding boxes) is dynamic and stays with the engine (bounding spheres)."""
    coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    excl = {(int(a), int(b)) for a, b in zip(m.exclude_signature >> 16, m.exclude_signature & 0xFFFF)}
    out = []
    for i, ga in enumerate(coll):
        for gb in coll[i + 1:]:
            g1, g2 = (ga, gb) if (m.geom_type[ga], ga) <= (m.geom_type[gb], gb) else (gb, ga)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            w1, w2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
            if b1 == b2 or w1 == w2 or (min(w1, w2), max(w1, w2)) in excl or (min(b1, b2), max(b1, b2)) in excl:
                continue
            if m.body_weldid[m.body_parentid[w1]] == w2 or m.body_weldid[m.body_parentid[w2]] == w1:
                continue
            if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])):
                continue
            p1, p2 = m.geom_priority[g1], m.geom_priority[g2]
            if p1 != p2:
                a = g1 if p1 > p2 else g2
                condim, solref, solimp, fri = int(m.geom_condim[a]), m.geom_solref[a], m.geom_solimp[a], m.geom_friction[a]
            else:
                s1, s2 = m.geom_solmix[g1], m.geom_solmix[g2]
                mix = s1 / (s1 + s2) if s1 >= 1e-15 and s2 >= 1e-15 else 0.5 if s1 < 1e-15 and s2 < 1e-15 else 0.0 if s1 < 1e-15 else 1.0
                condim = int(max(m.geom_condim[g1], m.geom_condim[g2]))
                solref = mix * m.geom_solref[g1] + (1 - mix) * m.geom_solref[g2] if m.geom_solref[g1][0] > 0 and m.geom_solref[g2][0] > 0 else np.minimum(m.geom_solref[g1], m.geom_solref[g2])
                solimp = mix * m.geom_solimp[g1] + (1 - mix) * m.geom_solimp[g2]
                fri = np.maximum(m.geom_friction[g1], m.geom_friction[g2])
            out.append(dict(geom1=int(g1), geom2=int(g2), condim=condim, solref=solref, solimp=solimp, friction=[fri[0], fri[0], fri[1], fri[2], fri[2]],
                            margin=float(m.geom_margin[g1] + m.geom_margin[g2]), gap=float(m.geom_gap[g1] + m.geom_gap[g2])))
    return out


def centre_inside(m, d, c):
    """A contact where one geom's centre lies inside the other primitive geom: the penetration direction is tolerance-arbitrary (EPA stops on
    any of many equally deep faces, so MuJoCo, the CPU build and the card disagree at 1e-2); such states are rejected from a contact gate."""
    G = mujoco.mjtGeom
    for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
        p = d.geom_xmat[b].reshape(3, 3).T @ (d.geom_xpos[a] - d.geom_xpos[b])
        sz, ty = m.geom_size[b], m.geom_type[b]
        if ((ty == G.mjGEOM_SPHERE and np.linalg.norm(p) < sz[0]) or (ty == G.mjGEOM_ELLIPSOID and ((p / sz) ** 2).sum() < 1)
                or (ty == G.mjGEOM_CAPSULE and np.linalg.norm(p - np.array([0, 0, np.clip(p[2], -sz[1], sz[1])])) < sz[0])
                or (ty == G.mjGEOM_CYLINDER and abs(p[2]) < sz[1] and np.hypot(p[0], p[1]) < sz[0])):
            return True
    return False


def reject_degenerate(m, q, v, count):
    """The first `count` of the candidate states (q, v) whose contacts have no centre inside the other geom (centre_inside), MuJoCo's
    contacts deciding; prints how many were rejected. For the flight twin's random and near-rest draws (check.sh flight_contacts)."""
    d, keep = mujoco.MjData(m), []
    for n in range(len(q)):
        d.qpos[:], d.qvel[:] = q[n], v[n]
        mujoco.mj_forward(m, d)
        if not any(centre_inside(m, d, c) for c in d.contact[:d.ncon]):
            keep.append(n)
        if len(keep) == count:
            break
    if len(keep) < count:
        raise SystemExit(f'FAIL states: only {len(keep)} of {len(q)} candidates without a degenerate overlap, {count} needed')
    print(f'states: {keep[-1] + 1 - count} of {keep[-1] + 1} candidates rejected for a degenerate overlap (a geom centre inside the other geom)')
    return q[keep], v[keep]


def with_auto_pairs(mjcf, out, cone='pyramidal', explicit=True):
    """mjcf with its automatic contact pairs written as explicit <pair> elements and every geom's contype/conaffinity zeroed (MuJoCo then
    collides exactly these pairs: auto_pairs), the cone set; mesh paths made absolute; written to `out`; returns the path. The flight
    model's gate body (check.sh flight_contacts); tools/auto_pairs_check.py proves the pair set and the mixed parameters against MuJoCo's
    own dynamic pairs."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    base = Path(mjcf).parent / (spec.meshdir or '')
    for mesh in spec.meshes:
        mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
    spec.meshdir = ''
    m0 = spec.compile()
    name = lambda g: mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, g)
    if explicit:  # explicit=False: the same model through the same writer (poses rounded alike), dynamic pairs kept, for tools/auto_pairs_check.py
        for p in auto_pairs(m0):
            pr = spec.add_pair(geomname1=name(p['geom1']), geomname2=name(p['geom2']), condim=p['condim'], margin=p['margin'], gap=p['gap'])
            pr.solref, pr.solimp, pr.friction = list(p['solref']), list(p['solimp']), p['friction']
        for g in spec.geoms:
            g.contype = g.conaffinity = 0
    spec.option.cone = getattr(mujoco.mjtCone, 'mjCONE_' + cone.upper())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def with_gravcomp(mjcf, out):
    """mjcf with body_gravcomp 1.0, 0.5, 0.0, 0.8 on its bodies in turn (MjSpec, mesh paths made absolute: combine() does the
    same), written to `out`; the gate body of check.sh gravcomp for the fly and the spider. Returns the path."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    base = Path(mjcf).parent / (spec.meshdir or '')
    for mesh in spec.meshes:
        mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
    spec.meshdir = ''
    for i, b in enumerate(b for b in spec.bodies if b.name != 'world'):
        b.gravcomp = (1.0, 0.5, 0.0, 0.8)[i % 4]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def with_actfrc(mjcf, out, count=64, keep=False, contacts=True):
    """with_gravcomp(mjcf) plus joint-level actuator force limits (jnt_actfrcrange, clampVec at the end of mj_fwdActuation,
    engine_forward.c:788): actuatorgravcomp on every second joint, and on two of every three hinge/slide joints a range
    [-0.6 s, 0.8 s], s the median |qfrc_actuator| at that dof over the check's states and servo targets, so the clamp binds on
    about half the states and on both sides. The states (standing rollouts of the model WITHOUT the limits: clamped servos let
    the fly drop and bounce into contact poses no other stage meets) go to out.npz, check's STATES. keep: the model's own
    gravcomp and actuatorgravcomp (models/gravcomp_act.xml). contacts False: pairs and contype off. The gate bodies of check.sh actfrc."""
    spec = mujoco.MjSpec.from_file(str(mjcf) if keep else with_gravcomp(mjcf, out))
    if not keep:
        spec.meshdir = ''
    joints = list(spec.joints)
    for i, j in enumerate(joints):
        j.actgravcomp = j.actgravcomp if keep else i % 2 == 0
    if not contacts:
        for pr in list(spec.pairs):
            spec.delete(pr)
        for g in spec.geoms:
            g.contype = g.conaffinity = 0
    m0 = spec.compile()
    q, v = standing(m0, count)
    np.savez(Path(out).with_suffix('.npz'), qpos=q, qvel=v)
    rng, d, f = np.random.default_rng(2), mujoco.MjData(m0), np.zeros((count, m0.nv))
    u = np.array([q[:, m0.jnt_qposadr[m0.actuator_trnid[i, 0]]] + rng.uniform(-0.3, 0.3, count)
                  if m0.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT else rng.uniform(0, 1, count) for i in range(m0.nu)]).T
    for n in range(count):
        d.qpos[:], d.qvel[:], d.ctrl[:] = q[n], v[n], u[n]
        mujoco.mj_forward(m0, d)
        f[n] = d.qfrc_actuator
    k = 0
    for jj, j in enumerate(joints):
        if m0.jnt_type[jj] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        s = float(np.median(np.abs(f[:, m0.jnt_dofadr[jj]])))
        k += 1
        if s > 0 and k % 3:
            j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.actfrcrange = [-0.6 * s, 0.8 * s]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def with_spatial_tendons(mjcf, out, stiffness=5.0, damping=0.05):
    """mjcf plus three spatial tendons, for the gate on bodies that ship none (fly, spider): a straight site-site tendon between two
    far-apart bodies, the same two sites with a sphere wrap placed on the straight path at qpos0 (a sphere geom, no mass, no
    contact, on a body between them), and a three-site pulley split. Sites sit near body origins at 2 percent of the model's
    extent. MjSpec, mesh paths made absolute (combine() does the same); written to `out`; returns the path."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    base = Path(mjcf).parent / (spec.meshdir or '')
    for mesh in spec.meshes:
        mesh.file = str((base / mesh.file).resolve()) if mesh.file else mesh.file
    spec.meshdir = ''
    m0 = spec.compile()
    ext = float(m0.stat.extent)
    names = [b.name for b in spec.bodies if b.name != 'world']
    moving = [n for n in names if m0.body_dofnum[m0.body(n).id] or m0.body_parentid[m0.body(n).id] > 0]
    pick = [moving[i] for i in (len(moving) // 5, len(moving) // 2, 4 * len(moving) // 5, len(moving) // 3)]  # a, mid, b, c
    for k, n in enumerate(pick):
        spec.body(n).add_site(name=f'tsite{k}', pos=[0.02 * ext * (1 + k), 0.01 * ext, 0.0], size=[0.002 * ext] * 3)
    d0 = mujoco.MjData(m0)
    m1 = spec.compile()
    d1 = mujoco.MjData(m1)
    d1.qpos[:] = m1.qpos0
    mujoco.mj_forward(m1, d1)
    pa, pb = d1.site('tsite0').xpos, d1.site('tsite2').xpos
    mid = spec.body(pick[1])
    c = (pa + pb) / 2
    local = d1.body(pick[1]).xmat.reshape(3, 3).T @ (c - d1.body(pick[1]).xpos)
    mid.add_geom(name='tsphere', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.25 * np.linalg.norm(pb - pa), 0, 0], pos=list(local), mass=0,
                 contype=0, conaffinity=0, group=5)
    side = mid.add_site(name='tside', pos=list(local + 0.5 * np.linalg.norm(pb - pa) * np.array([0, 0, 1.0])), size=[0.002 * ext] * 3)
    for name, parts in (('tstraight', ['tsite0', 'tsite2']), ('twrap', ['tsite0', ('tsphere', 'tside'), 'tsite2']), ('tsplit', ['tsite3', 'tsite0', 'PULLEY', 'tsite0', 'tsite2', 'PULLEY', 'tsite0', 'tsite1'])):
        t = spec.add_tendon(name=name, stiffness=stiffness, damping=damping)
        for part in parts:
            if part == 'PULLEY':
                t.wrap_pulley(2.0)
            elif isinstance(part, tuple):
                t.wrap_geom(part[0], part[1])
            else:
                t.wrap_site(part)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(spec.to_xml())
    return str(out)


def gravcomp_block(m):
    """Body gravity compensation (engine_passive.c:817), last block: ints jnt_actgravcomp per joint (those joints' share goes to
    qfrc_actuator, engine_forward.c:770, only when the model has actuators), floats body_gravcomp per body; joint-level actuator
    force limits clamp that sum after it (actfrc_block); a sleeping animal skips the whole step, gravcomp included (sleep_block)."""
    if not m.body_gravcomp.any() or not np.linalg.norm(m.opt.gravity):
        return None
    return np.asarray(m.jnt_actgravcomp), np.asarray(m.body_gravcomp), {'GRAVCOMP': 1}


def actfrc_block(m):
    """Joint-level actuator force limits (clampVec(qfrc_actuator, jnt_actfrcrange, jnt_actfrclimited, njnt, jnt_dofadr),
    engine_forward.c:788, after actuator gravcomp), last block: ints per limited joint its dofadr (the clamp touches that one
    dof, whatever the joint type), floats its range lo, hi; -DNAFRC. None without actuators (nu 0: mj_fwdActuation returns
    before the clamp) or without a limited joint."""
    lim = np.flatnonzero(m.jnt_actfrclimited)
    if not m.nu or not len(lim):
        return None
    return np.asarray(m.jnt_dofadr[lim]), np.asarray(m.jnt_actfrcrange[lim]).ravel(), {'NAFRC': len(lim)}


def implicit_block(m):
    """Full implicit integrator (mj_implicitSkip, engine_forward.c:1864), last block, only with opt.integrator implicit: MuJoCo's body-dof
    sparsity B (a body's row = its branch: ancestor dofs, own dofs, subtree dofs; rows of bodies welded to the world are never read),
    the dof-dof sparsity D (a dof's row = its body's row, diagonal at D_diag) and mapM2D (D entry -> M entry, M in M_rowadr layout),
    plus per body the number of ancestor dofs as copyFromParent counts them (engine_derivative.c:468). Ints per body (rownnz, rowadr,
    ancestor count), then B_colind, then per dof (rowadr, diag, rownnz), then D_colind and mapM2D."""
    if m.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICIT:
        return None
    anc = np.zeros(m.nbody, int)
    for n in range(1, m.nbody):
        a = m.body_weldid[m.body_parentid[n]]
        while a > 0:
            anc[n] += m.body_dofnum[a]
            a = m.body_weldid[m.body_parentid[a]]
    ints = np.concatenate([m.B_rownnz, m.B_rowadr, anc, m.B_colind, m.D_rowadr, m.D_diag, m.D_rownnz, m.D_colind, m.mapM2D,
                           pair_table(m.nv, m.D_rowadr, m.D_rownnz, m.D_colind)])
    return ints, np.zeros(0), {'NBB': m.nB, 'NDD': m.nD}


def pair_table(nv, rowadr, rownnz, colind):
    """(k, p) -> entry of a dof-dof sparse matrix in row k at column p, -1 if the pattern lacks it: nv x nv int32, so the velocity-derivative
    blocks (tendon damping, fluid) land on their entries with one load instead of a scan of row k per dof pair (the scan was most of
    50 ms per launch on the flight model's 130 fluid bodies)."""
    t = -np.ones((nv, nv), np.int32)
    for k in range(nv):
        for c in range(rownnz[k]):
            t[k, colind[rowadr[k] + c]] = rowadr[k] + c
    return t.ravel()


def sleep_block(m):
    """Sleep (mj_sleep, engine_sleep.c:563; mjENBL_SLEEP), last float block: opt.sleep_tolerance, then dof_length per dof (treeCanSleep's
    velocity weights). One animal is one tree and its own island in the engine, so the export takes one tree, the Euler integrator and a
    tree that may sleep; MuJoCo's wake rules reduce to a host-written velocity (no applied forces, no cross-tree links)."""
    if not m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
        return None
    P = mujoco.mjtSleepPolicy
    if m.ntree != 1 or m.opt.integrator or m.tree_sleep_policy[0] in (P.mjSLEEP_NEVER, P.mjSLEEP_AUTO_NEVER):
        raise SystemExit('FAIL export: sleep is gated on one tree per export, the Euler integrator and a tree allowed to sleep (check.sh sleep)')
    return np.zeros(0), np.concatenate([[m.opt.sleep_tolerance], m.dof_length]), {'SLEEP': 1}


def mpair_block(m):
    """implicitfast only, last block: pair_table of M's lower pattern (M_rowadr, M_rownnz, M_colind), the target of qH's velocity-derivative blocks."""
    if m.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        return None
    return pair_table(m.nv, m.M_rowadr, m.M_rownnz, m.M_colind), np.zeros(0), {'MPAIR': 1}


def mid_block(m):
    """Implicit midpoint of standalone free bodies (mj_midpoint, engine_forward.c:1708), last block, implicitfast only: per free joint of a
    childless one-joint tree (midpoint_eligible's static part, engine_forward.c:1519) its body, dof address and aligned flag (ipos 0).
    The runtime part (no live contact on the body) is smooth.mojo mid_eligible. Equalities are refused with non-Euler integrators."""
    if m.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        return None
    rows = [[b, m.jnt_dofadr[j], int(not m.body_ipos[b].any())] for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
            for b in [m.jnt_bodyid[j]] if m.tree_dofnum[m.dof_treeid[m.jnt_dofadr[j]]] == 6 and m.body_subtreemass[b] == m.body_mass[b]]
    return (np.array(rows).ravel(), np.zeros(0), {'NMID': len(rows)}) if rows else None


def actuators(m):
    """S3 tables: ints per actuator (transmission type, dof or body, ctrllimited, forcelimited, qpos address or -1),
    floats per actuator
    (gain, bias 0-2, ctrlrange 2, forcerange 2, gear), then eleven floats per annotated muscle."""
    T = mujoco.mjtTrn
    muscles = {}
    names = {}
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            names[name] = i
    for n in range(m.nnumeric):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_NUMERIC, n) or ''
        if not name.startswith('muscle_'):
            continue
        actuator = name[len('muscle_'):]
        if actuator not in names or actuator in muscles:
            raise SystemExit(f'FAIL export: muscle annotation {name!r} does not name one actuator exactly once')
        if m.numeric_size[n] != 11:
            raise SystemExit(f'FAIL export: {name} needs 11 numeric parameters')
        p = m.numeric_data[m.numeric_adr[n]:m.numeric_adr[n] + 11].copy()
        if not np.isfinite(p).all() or p[0] <= 0 or p[2] <= 0 or p[3] <= 0 or p[4] <= 1 or p[5] < 0 or p[6] < 0 or np.any(p[7:] <= 0):
            raise SystemExit(f'FAIL export: {name} has invalid parameters (positive scales/times, eccentric cap > 1, passive stiffness >= 0)')
        muscles[actuator] = p
    muscle_ids = {name: i for i, name in enumerate(muscles)}
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ''
        is_muscle = name in muscles
        joint = m.actuator_trntype[i] == T.mjTRN_JOINT and m.jnt_type[m.actuator_trnid[i, 0]] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        if (not (joint or m.actuator_trntype[i] in (T.mjTRN_BODY, T.mjTRN_TENDON)) or m.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED
                or m.actuator_biastype[i] not in (mujoco.mjtBias.mjBIAS_NONE, mujoco.mjtBias.mjBIAS_AFFINE)
                or m.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE or m.actuator_actearly[i] or m.actuator_delay[i]):
            raise SystemExit(f'FAIL export: actuator {i} is not a ported kind (hinge/slide joint, adhesion body or fixed tendon, fixed gain, no dynamics)')
        if is_muscle:
            tendon = int(m.actuator_trnid[i, 0]) if m.actuator_trntype[i] == T.mjTRN_TENDON else -1
            fixed = tendon >= 0 and np.all(m.wrap_type[m.tendon_adr[tendon]:m.tendon_adr[tendon] + m.tendon_num[tendon]] == mujoco.mjtWrap.mjWRAP_JOINT)
            if (not fixed or not m.actuator_ctrllimited[i] or m.actuator_ctrlrange[i, 0] != 0 or m.actuator_ctrlrange[i, 1] != 1
                    or m.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_NONE or m.actuator_forcelimited[i]
                    or not np.isclose(m.actuator_gear[i, 0], 1)):
                raise SystemExit(f'FAIL export: muscle {name} needs a fixed tendon, no bias or force limit, unit gear, and ctrlrange 0 1')
        if m.actuator_trntype[i] == T.mjTRN_BODY:
            b = m.actuator_trnid[i, 0]
            if any(m.pair_gap[k] > 0 and b in (m.geom_bodyid[m.pair_geom1[k]], m.geom_bodyid[m.pair_geom2[k]]) for k in range(m.npair)):
                raise SystemExit(f'FAIL export: adhesion actuator {i} on a body with a gap pair (excluded contacts in the gap band transmit adhesion in MuJoCo; not ported)')
    ints = np.array([[(4 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or '') in muscles else m.actuator_trntype[i]),
                      m.jnt_dofadr[m.actuator_trnid[i, 0]] if m.actuator_trntype[i] == T.mjTRN_JOINT else m.actuator_trnid[i, 0],  # tendon id remains in the ordinary transmission slot
                      m.actuator_ctrllimited[i], m.actuator_forcelimited[i],
                      muscle_ids[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)] if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or '') in muscles else (m.jnt_qposadr[m.actuator_trnid[i, 0]] if m.actuator_trntype[i] == T.mjTRN_JOINT else -1)] for i in range(m.nu)])
    bias = np.where((m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)[:, None], m.actuator_biasprm[:, :3], 0)
    floats = np.concatenate([m.actuator_gainprm[:, :1], bias, m.actuator_ctrlrange, m.actuator_forcerange, m.actuator_gear[:, :1]], axis=1)
    params = np.array(list(muscles.values()), dtype=np.float64).reshape(-1, 11)
    return ints.ravel(), np.concatenate([floats.ravel(), params.ravel()]), len(muscles)


def export(mjcf, out, regex=None, act=False, convex=False, hfexact=False, limits=None):
    m, out = load(mjcf), Path(out)
    has_muscle = any((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_NUMERIC, n) or '').startswith('muscle_') for n in range(m.nnumeric))
    if has_muscle and not act:
        raise SystemExit('FAIL export: muscle annotations require the stage-7 actuator block')
    cross = cross_trees(m, regex, convex)
    if cross and (m.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
                  or m.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON
                  or m.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise SystemExit(f"FAIL export: coupling across two trees ({', '.join(cross[:3])}) is gated with the Newton solver, the pyramidal cone and the Euler integrator only (check.sh cross)")
    out.mkdir(parents=True, exist_ok=True)
    ints = [getattr(m, f).ravel() for f in INTS]
    floats = [np.ravel(getattr(m.opt, f[4:]) if f.startswith('opt.') else getattr(m, f)) for f in FLOATS]
    sizes = {'NBODY': m.nbody, 'NJNT': m.njnt, 'NQ': m.nq, 'NGEOM': m.ngeom, 'NV': m.nv, 'NC': m.nC, 'NU': m.nu}
    if m.opt.integrator:  # absent = Euler: those binaries are unchanged
        sizes['INTEGRATOR'] = int(m.opt.integrator)
    if m.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:  # absent = Newton
        sizes['SOLVER'] = int(m.opt.solver)
    if limits is None:  # joint limit rows ride the constraint block: exported when the model already has one (contacts, equalities,
        limits = bool(regex) or bool(m.neq) or bool(m.tendon_limited.any())  # tendon limits); contact-free stages below S6 keep none
    eq, hr, hl = eq_rows(m, limits), [], []
    if regex or eq:  # equality rows ride the solver block: no pairs (regex '^$') still needs it
        ci, cf, cs = feet(m, regex or '^$', [e['union'] for e in eq], hr, convex, hfexact, hl)
        ints, floats, sizes = ints + [ci], floats + [cf], sizes | cs
    if act:
        ai, af, nmuscle = actuators(m)
        ints, floats = ints + [ai], floats + [af]
        if nmuscle:
            sizes['NMUSCLE'] = nmuscle
    for block in (fluid(m), tendons(m), spatial(m), convex_block(m, regex, convex, hfexact, hl[0] if hl else None) if regex and (convex or hfexact) else None):  # last, so every earlier block keeps its offsets and a model without them its bytes
        if block:
            ints, floats, sizes = ints + [block[0]], floats + [block[1]], sizes | block[2]
    if eq:  # after the other lanes' blocks too: a model without equalities keeps its bytes
        block = eq_block(m, eq, hr[0])
        ints, floats, sizes = ints + [block[0]], floats + [block[1]], sizes | block[2]
    for block in (mocap_block(m), gravcomp_block(m), implicit_block(m), mid_block(m), mpair_block(m), sleep_block(m), actfrc_block(m)):  # after the equality block, in Mojo's order: CMOC follows CEQM, CGC follows CMOC, CIMP follows CGC, CMID follows the implicit block, CAFR follows MPAIR, AFRF follows SLPF
        if block:
            ints, floats, sizes = ints + [block[0]], floats + [block[1]], sizes | block[2]
    if hfexact and os.environ.get('HTAB') and 'HFNR' in sizes:  # table heightfield: the engine reads node heights, not the function
        g = next(g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD)
        h = m.geom_dataid[g]
        n = int(m.hfield_nrow[h] * m.hfield_ncol[h])
        cht = sum(len(f) for f in floats)
        floats = floats + [np.concatenate([[0, 0], m.hfield_data[m.hfield_adr[h]:m.hfield_adr[h] + n] * m.hfield_size[h][2]])]
        sizes |= {'HTAB': 1, 'CHT': cht}
    np.concatenate(ints).astype(np.int32).tofile(out / 'mi.bin')
    np.concatenate(floats).astype(np.float32).tofile(out / 'mf.bin')
    (out / 'sizes.txt').write_text(' '.join(f'-D{k}={v}' for k, v in sizes.items()) + '\n')
    print(f'{mjcf}: {sizes} -> {out}')
    return m


def convex_block(m, regex, convex=True, hfexact=False, hulls=None):
    """Convex pairs for convex.mojo, appended last: ints per exported pair (feet order) the kind (0 plane, sphere or tangent-plane
    heightfield; 1 ellipsoid-mesh one contact; 2 box-mesh up to 5; 3 exact heightfield-mesh, GJK + EPA per cell prism); floats
    opt.ccd_tolerance, the heightfield's size 0-3 and origin z (zeros without one), then per pair geom1's size (0 for the other
    kinds). -DNCV=1, -DCCD64=1, -DCCD_ITER=opt.ccd_iterations; exact heightfield also -DHFEXACT=1 -DHFNR -DHFNC -DHF_SLOTS.
    None without convex or exact heightfield pairs."""
    G = mujoco.mjtGeom
    sel, _ = pairs(m, regex, convex)
    kind = [(prim_pair(m, i) or {G.mjGEOM_ELLIPSOID: 1, G.mjGEOM_BOX: 2, G.mjGEOM_MESH: 4}.get(m.geom_type[pair_geoms(m, i)[0]], 0) if m.geom_type[pair_geoms(m, i)[1]] == G.mjGEOM_MESH else prim_pair(m, i)) if convex else 0 for i in sel]
    if any(m.pair_dim[i] not in (1, 3) for i in sel):
        raise SystemExit('FAIL export: pair condim 4 and 6 are not ported (1 and 3 are)')
    if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and any(m.pair_dim[i] == 1 for i in sel):
        raise SystemExit('FAIL export: condim 1 pairs are gated with the pyramidal cone only (check.sh flight_contacts)')
    hf = [i for i in sel if m.geom_type[pair_geoms(m, i)[0]] == G.mjGEOM_HFIELD] if hfexact else []
    for i in hf:
        kind[sel.index(i)] = 3
    if not any(kind):
        return None
    head, sizes = [m.opt.ccd_tolerance, 0, 0, 0, 0, 0], {'NCV': 1, 'CCD_ITER': m.opt.ccd_iterations, 'CCD64': 1}
    if hf:
        g = m.pair_geom1[hf[0]]
        h = m.geom_dataid[g]
        if (m.geom_quat[g] != [1, 0, 0, 0]).any() or m.geom_pos[g][:2].any():
            raise SystemExit('FAIL export: exact heightfield needs the hfield geom unrotated and centred at x = y = 0')
        head = [m.opt.ccd_tolerance, *m.hfield_size[h], m.geom_pos[g][2]]
        sizes |= {'HFEXACT': 1, 'HFNR': int(m.hfield_nrow[h]), 'HFNC': int(m.hfield_ncol[h]), 'HF_SLOTS': HF_SLOTS}
    size = [m.geom_size[pair_geoms(m, i)[0]] if k in (1, 2, 4, 5, 6, 7) else np.zeros(3) for i, k in zip(sel, kind)]
    size2 = [m.geom_size[pair_geoms(m, i)[1]] if k in (4, 5, 6, 7) else np.zeros(3) for i, k in zip(sel, kind)]
    shapes = [tuple(SHAPE_KIND.get(G(m.geom_type[g]), 0) for g in pair_geoms(m, i)) for i in sel]
    mm = [m.geom_type[pair_geoms(m, i)[0]] == G.mjGEOM_MESH for i in sel]  # mesh-mesh (kind 4, both shapes MESH): geom1's hull offsets, -DNMM
    if any(mm):
        sizes['NMM'] = 1
    hull1 = [c for i, k in zip(sel, mm) for c in (hulls[m.geom_dataid[pair_geoms(m, i)[0]]] if k else (0, 0))] if any(mm) else []
    return np.array(kind + [c for ab in shapes for c in ab] + [int(m.pair_dim[i]) for i in sel] + hull1), np.concatenate([head, *size, *size2]), sizes


def cube_colour(m, tex, v):
    """MuJoCo cube texture (6 faces stacked: +x -x +y -y +z -z) sampled in each vertex's direction from the mesh origin, the
    way MuJoCo maps a cube texture on a mesh without texture coordinates; rgb in 0..1."""
    w, nc = int(m.tex_width[tex]), int(m.tex_nchannel[tex])
    img = m.tex_data[m.tex_adr[tex]:m.tex_adr[tex] + 6 * w * w * nc].reshape(6, w, w, nc)[..., :3] / 255.0
    if w % 5 == 0:  # a mip level, as MuJoCo's filtered lookup sees it: 5x5 texel means (its 5 % random black marks blur away)
        img = img.reshape(6, w // 5, 5, w // 5, 5, 3).mean((2, 4))
        w //= 5
    a = np.abs(v)
    ax = a.argmax(1)
    face = 2 * ax + (np.take_along_axis(v, ax[:, None], 1)[:, 0] < 0)
    x, y, z = v.T
    ma = np.maximum(a.max(1), 1e-30)
    sel = [face == k for k in range(6)]
    sc = np.select(sel, [-z, z, x, x, x, -x]) / ma  # GL cube-map face orientation
    tc = np.select(sel, [-y, -y, z, -z, -y, -y]) / ma
    return img[face, np.clip(((tc + 1) / 2 * w).astype(int), 0, w - 1), np.clip(((sc + 1) / 2 * w).astype(int), 0, w - 1)]


def vertex_colour(m, g, v):
    """Display colour per vertex of mesh geom g: material rgb x its texture (cube: sampled per vertex; 2d: its mean)."""
    mat = m.geom_matid[g]
    rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
    col = np.tile(rgba[:3], (len(v), 1)).astype(np.float64)
    tex = m.mat_texid[mat][mujoco.mjtTextureRole.mjTEXROLE_RGB] if mat >= 0 else -1
    if tex >= 0 and m.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_CUBE:
        col *= cube_colour(m, tex, v)
    elif tex >= 0:
        n = int(m.tex_nchannel[tex])
        col *= m.tex_data[m.tex_adr[tex]:m.tex_adr[tex] + m.tex_width[tex] * m.tex_height[tex] * n].reshape(-1, n)[:, :3].mean(0) / 255.0
    return col


_ASSETS = []


def _assets(*args, mesh=None):
    """Run assets.mojo (the world-part mesh builders; assets.py stays its oracle, assets_check.py): (vertices, faces).
    mesh: (vertices, faces) written as the input file of `split`. Files under .work/assets-mojo/, one pair per process."""
    out = Path('.work/assets-mojo')
    out.mkdir(parents=True, exist_ok=True)
    if not _ASSETS:
        _ASSETS.append(out / 'assets')
        subprocess.run([str(Path.home() / 'iTools/bin/mojo-cached'), 'build', str(Path(__file__).with_name('assets.mojo')),
                        '-o', str(_ASSETS[0])], check=True)
    src, dst = out / f'in-{os.getpid()}.bin', out / f'out-{os.getpid()}.bin'
    cmd = [str(_ASSETS[0]), *map(str, args), str(dst)]  # primitive BOX SX SY SZ OUT
    if mesh is not None:  # split IN OUT EDGE
        v, f = np.asarray(mesh[0], '<f8'), np.asarray(mesh[1], '<i4')
        src.write_bytes(np.array([len(v), len(f)], '<u4').tobytes() + v.tobytes() + f.tobytes())
        cmd = [str(_ASSETS[0]), args[0], str(src), str(dst), *map(str, args[1:])]
    subprocess.run(cmd, check=True)
    b = dst.read_bytes()
    nv, nf = np.frombuffer(b[:8], '<u4')
    return np.frombuffer(b[8:8 + 24 * nv], '<f8').reshape(nv, 3).copy(), np.frombuffer(b[8 + 24 * nv:], '<i4').reshape(nf, 3).astype(np.int64)


def assets_primitive(box, size):
    """(vertices, faces) of a MuJoCo box (half sizes) or ellipsoid (radii) in its geom frame, from assets.mojo."""
    return _assets('primitive', int(bool(box)), *(repr(float(s)) for s in size[:3]))


def assets_split(v, f, edge=0.4):
    """Every triangle side under `edge` (render.raster gives one thread a triangle's screen box), from assets.mojo."""
    return _assets('split', repr(float(edge)), mesh=(v, f))


def meshes(mjcf, out, grid=0, world=None):
    """Renderer tables (docs/RENDERER.md) for every mesh geom: rmi.bin int32 per part (body, vertex offset,
    vertex count, triangle offset, triangle count), rmf.bin float32 per part (local pos 3, local quat 4, box
    offset 3, box scale 3, bounding sphere centre 3 and radius 1, rgba 4), rv.bin int16 vertices quantised to the
    part's box then int16 unit vertex normals x 32767 (area-weighted face normals, mesh frame), then int16 vertex rgb 0..255
    (material x texture: vertex_colour), rt.bin uint16 triangles local to the part, rp.bin uint8 part per triangle. Returns the decode error relative to part size.
    world: also draw the world body's box and ellipsoid geoms (walls, stones) and these extra world parts, a list of
    (vertices (n, 3) world mm, faces (m, 3), rgba) (fly-connectome agent/body_engine/assets.py builds them; boxes, ellipsoids and the split come from assets.mojo); body 0 parts are drawn fixed (render.vertex)."""
    m, out = load(mjcf), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ints, floats, verts, norms, cols, tris, part, worst = [], [], [], [], [], [], [], 0.0

    def add(body, pos, quat, v, f, rgba, name, col=None):
        nonlocal worst
        if len(v) > 65535 or len(ints) // 5 >= 255:
            raise SystemExit(f'FAIL meshes: {name}: over 65,535 vertices (uint16 triangles) or 255 parts (uint8 part ids)')
        lo, hi = v.min(0), v.max(0)
        scale = np.maximum(hi - lo, 1e-12) / 65535
        qv = np.round((v - lo) / scale - 32768).astype(np.int16)
        worst = max(worst, float(np.abs((qv.astype(np.float64) + 32768) * scale + lo - v).max() / (hi - lo).max()))
        centre = (lo + hi) / 2
        ints.extend([body, len(verts), len(v), len(tris), len(f)])
        floats.extend([*pos, *quat, *(lo + 32768 * scale), *scale, *centre, np.linalg.norm(v - centre, axis=1).max(), *rgba])
        verts.extend(qv)
        f = np.asarray(f)
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # area-weighted face normals
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        norms.extend(np.round(vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30) * 32767).astype(np.int16))
        cols.extend(np.round(np.clip(np.tile(rgba[:3], (len(v), 1)) if col is None else col, 0, 1) * 255).astype(np.int16))
        tris.extend(np.asarray(f).astype(np.uint16))
        part.extend([len(ints) // 5 - 1] * len(f))

    for g in np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
        col = vertex_colour(m, g, v)
        if grid:  # vertex clustering on a grid x grid x grid lattice over the part (Flybody's 1.6 M-vertex meshes; render only)
            c = np.floor((v - v.min(0)) / (np.ptp(v, 0).max() / grid + 1e-12)).astype(np.int64)
            _, inv = np.unique(c, axis=0, return_inverse=True)
            inv = inv.ravel()
            v = np.array([np.bincount(inv, v[:, a]) for a in range(3)]).T / np.bincount(inv)[:, None]
            col = np.array([np.bincount(inv, col[:, a]) for a in range(3)]).T / np.bincount(inv)[:, None]
            f = inv[f]
            f = np.unique(f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])], axis=0)
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        add(m.geom_bodyid[g], m.geom_pos[g], m.geom_quat[g], v, f, rgba, f'geom {g}', col)
    if world is not None:
        primitive, split = assets_primitive, assets_split
        for g in np.flatnonzero((m.geom_bodyid == 0) & np.isin(m.geom_type, (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID))):
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            if rgba[3] == 0:  # invisible (the lid): nothing to draw, and split it alone overflows a part's 65,535 vertices
                continue
            v, f = split(*primitive(int(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX), m.geom_size[g]))
            add(0, m.geom_pos[g], m.geom_quat[g], v, f, rgba, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        for i, (v, f, rgba) in enumerate(world):
            v, f = split(v, f)
            add(0, (0, 0, 0), (1, 0, 0, 0), np.asarray(v, float), f, rgba, f'world part {i}')
    np.array(ints, np.int32).tofile(out / 'rmi.bin')
    np.array(floats, np.float32).tofile(out / 'rmf.bin')
    np.concatenate([np.array(verts, np.int16), np.array(norms, np.int16), np.array(cols, np.int16)]).tofile(out / 'rv.bin')
    np.array(tris, np.uint16).tofile(out / 'rt.bin')
    np.array(part, np.uint8).tofile(out / 'rp.bin')  # part of each triangle (69 parts here; export refuses 256+)
    (out / 'rsizes.txt').write_text(f'-DNPART={len(ints) // 5} -DNRV={len(verts)} -DNRT={3 * len(tris)}\n')  # NRT: indices
    print(f'{mjcf}: {len(ints) // 5} parts, {len(verts)} vertices, {len(tris)} triangles, '
          f'{(len(verts) * 18 + len(tris) * 6 + len(part) + len(floats) * 4 + len(ints) * 4) / 1024:.0f} KB; decode error {worst:.1e} of part size')
    return worst


def nearby(m, count, seed=0):
    """(qpos, qvel) near qpos0, where an equality-constrained model satisfies its constraints: free-joint position +-0.01, every
    rotation (free, ball) a random rotation of up to ~0.1 rad, hinges +-0.1 rad, slides +-0.01, qvel +-3. The physical regime of
    the equality stage; EQSPREAD=far swaps in mj_oracle's random states (violations of metres, forces that cancel to
    float32 noise in qacc_euler: MEASURED, not gated)."""
    rng, q = np.random.default_rng(seed), np.tile(m.qpos0, (count, 1))
    for j in range(m.njnt):
        a, t = m.jnt_qposadr[j], m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            q[:, a:a + 3] += rng.uniform(-0.01, 0.01, (count, 3))
            a += 3
        if t in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            r = q[:, a:a + 4] + 0.05 * rng.normal(size=(count, 4))
            q[:, a:a + 4] = r / np.linalg.norm(r, axis=1, keepdims=True)
        else:
            q[:, a] += rng.uniform(-0.1, 0.1, count) if t == mujoco.mjtJoint.mjJNT_HINGE else rng.uniform(-0.01, 0.01, count)
    return q, rng.uniform(-3, 3, (count, m.nv))


def standing(m, count, seed=0):
    """(qpos, qvel) of count states from actuated rollouts of the full model, adhesion on, one state every 2 ms after
    0.2 s of settling: half standing still (all 6 feet down), half with servos jittered +-0.2 rad every 50 ms
    (1 to 5 feet down)."""
    rng, q, v = np.random.default_rng(seed), [], []
    for jitter, n in ((0.0, count // 2), (0.2, count - count // 2)):
        d = mujoco.MjData(m)
        for k in range(2000 + 20 * n):
            if k % 500 == 0:
                d.ctrl[:] = np.where(m.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY, 1.0, rng.uniform(-jitter, jitter, m.nu))
            mujoco.mj_step(m, d)
            if k >= 2000 and k % 20 == 0:
                q.append(d.qpos.copy()); v.append(d.qvel.copy())
    return np.array(q), np.array(v)


def airborne(mjcf, out, lift=5.0, count=64):
    """STATES file of count contact-free poses: the standing rollout (always run with Euler, so the states do not depend on
    the integrator under test) with every free joint lifted by `lift` along z. No contact rows, so qacc_euler and
    qpos_next carry the integrator alone, without the active-set float32 noise of contact states."""
    m = load(mjcf)  # with ENGINE_OPT (a kv rollout is calmer than the shipped one: see check.sh integrator), forced to Euler
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    q, v = standing(m, count)
    for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):  # every animal of a combined model
        q[:, m.jnt_qposadr[j] + 2] += lift
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'airborne.npz', qpos=q, qvel=v)
    return str(out / 'airborne.npz')


def tendon_dense(s, want, q, v, count, show=True):
    """MuJoCo's tendon Jacobian is sparse rows (ten_J_rowadr/rownnz/colind): dense NTEN x NV per state, in place in `want`; also prints
    in how many states a geom wrap (sphere, cylinder) actually bends each path (the coverage of the gate)."""
    if 'ten_J' not in want:
        return
    dense = np.zeros((count, s.ntendon, s.nv))
    for n in range(count):
        for t in range(s.ntendon):
            a, z = s.ten_J_rowadr[t], s.ten_J_rownnz[t]
            dense[n, t, s.ten_J_colind[a:a + z]] = want['ten_J'][n][a:a + z]
    want['ten_J'] = dense.reshape(count, -1)
    if not show:
        return
    d = mujoco.MjData(s)
    W = mujoco.mjtWrap
    geom = [i for i in range(s.ntendon) if any(s.wrap_type[w] in (W.mjWRAP_SPHERE, W.mjWRAP_CYLINDER) for w in range(s.tendon_adr[i], s.tendon_adr[i] + s.tendon_num[i]))]
    bent = {i: 0 for i in geom}
    for n in range(count):
        d.qpos[:], d.qvel[:] = q[n], v[n]
        mujoco.mj_forward(s, d)
        for i in geom:
            bent[i] += int(d.ten_wrapnum[i] > s.tendon_num[i] - 1)  # a wrap adds its two tangent points to the sites
    print('wrapped paths: ' + ', '.join(f"{mujoco.mj_id2name(s, mujoco.mjtObj.mjOBJ_TENDON, i)} bent in {b}/{count} states" for i, b in bent.items()))


def reference_fresh(m, fn, fields, qpos, qvel=None, ctrl=None, xfrc=None, eqact=None):
    """mj_oracle.reference with a fresh MjData per state. One shared MjData carries qacc_warmstart from the previous state's step:
    harmless when the solver converges (Newton), but PGS stops unconverged at 100 sweeps, so its result depends on the warmstart
    and the engine's K=0 launch (warmstart 0) must be compared with a state that has none."""
    out = {f: [] for f in fields}
    calls = [getattr(mujoco, name) for name in fn.split(',')]
    for n, row in enumerate(qpos):
        d = mujoco.MjData(m)
        d.qpos[:] = row
        d.qvel[:] = 0 if qvel is None else qvel[n]
        d.ctrl[:] = 0 if ctrl is None else ctrl[n]
        if xfrc is not None:  # MuJoCo xfrc_applied (nbody x 6: force, torque at xipos)
            d.xfrc_applied[:] = xfrc[n]
        if eqact is not None:  # MuJoCo eq_active (neq), the engine's EQACT rows
            d.eq_active[:] = eqact[n]
        for call in calls:
            call(m, d)
        for f in fields:
            out[f].append(np.array(getattr(d, f)))
    return {f: np.array(v) for f, v in out.items()}


def check(mjcf, out, stage, count=64, k=0, oracle=None):
    """oracle: MuJoCo runs this MJCF instead of mjcf (same bodies and pair order), e.g. terrain_plane.xml for the
    engine's heightfield terrain.xml; standing states come from it too."""
    hfexact = bool(os.environ.get('HFEXACT'))  # exact heightfield contact (GJK + EPA per cell prism) instead of the tangent plane
    convex = bool(os.environ.get('CONVEX'))  # box and ellipsoid vs mesh pairs too (GJK + EPA), states moved into the arena's walls and stones
    feet = stage in ('contact', 'actuated', 'sptendon_act') or (stage == 'equality' and bool(load(mjcf).npair))  # equality + contacts: the contact path
    fx = FLOOR_X_CONE if load(mjcf).opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else FLOOR_X
    m, out = export(mjcf, out, FEET if feet else None, stage in ('actuated', 'sptendon_act'), convex, hfexact, True if stage == 'joint_limit' else None), Path(out)
    number, fn, fields = STAGES[stage]
    if spatial(m) and not set(TENDON_FIELDS) <= set(fields):  # SPATIAL binaries always append ten_length, ten_velocity, ten_J (smooth.mojo OTLEN): compare them
        fields = fields + TENDON_FIELDS
    ref = reference if m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON else reference_fresh  # see reference_fresh
    near = (stage in ('equality', 'mocap', 'tendon_limit', 'tendon_eq', 'joint_limit') and os.environ.get('EQSPREAD') != 'far') or bool(os.environ.get('NEAR_STATES'))  # NEAR_STATES=1: nearby() poses for any stage (flight contacts at rest)
    far = not near
    q, v = nearby(m, count) if near else standing(load(oracle) if oracle else m, count) if feet and not os.environ.get('RANDOM_STATES') else (states(m, count), velocities(m, count, scale=VSCALE))  # RANDOM_STATES=1: a contact body that flies (check.sh flight_contacts)
    if feet and (os.environ.get('RANDOM_STATES') or os.environ.get('NEAR_STATES')):  # a body-body contact gate: no state with a degenerate overlap (reject_degenerate)
        q4, v4 = nearby(m, 4 * count) if near else (states(m, 4 * count), velocities(m, 4 * count, scale=VSCALE))
        q, v = reject_degenerate(m, q4, v4, count)
    if (convex and not os.environ.get('STONES')) or os.environ.get('ARENA_STATES'):  # STONES: the model already carries its stones (model.stones), the standing rollout meets them; ARENA_STATES: the same states without the convex pairs (isolates what the pairs add)
        q, v = arena(m, q, v)
    if os.environ.get('STATES'):  # saved states instead (e.g. grooming poses: legs on the head, .work/.../groom_states.npz)
        z = np.load(os.environ['STATES'])
        q, v = z['qpos'][:count], z['qvel'][:count]
    if feet and Path(mjcf).name.startswith('cube'):  # every face of the cube carries states
        q, v = onfaces(q, v, float(Path(mjcf).stem[4:]))
    u = np.zeros((count, m.nu))
    if stage in ('actuated', 'sptendon_act'):  # servos within 0.3 rad of the current joint angle, adhesion anywhere in [0, 1]
        rng = np.random.default_rng(2)
        hold = bool(os.environ.get('HOLD'))  # HOLD=1: servos at the current angle, adhesion off (the sleep stage: the fly settles)
        for i in range(m.nu):
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                u[:, i] = q[:, m.jnt_qposadr[m.actuator_trnid[i, 0]]] + (0 if hold else rng.uniform(-0.3, 0.3, count))
            else:
                u[:, i] = 0 if hold else rng.uniform(0, 1, count)
    mp = mq = None
    if m.nmocap:  # mocap pose per animal rides in ctrl.bin rows nu..: pos 3, quat 4 per mocap body (quats left unnormalised on purpose)
        rng = np.random.default_rng(3)
        b = np.flatnonzero(m.body_mocapid >= 0)[np.argsort(m.body_mocapid[m.body_mocapid >= 0])]
        spread = 1 if far else 0.15  # near: a few mm and ~0.05 rad around the model pose
        mp = m.body_pos[b][None] + spread * rng.uniform(-0.03, 0.03, (count, m.nmocap, 3))
        mq = m.body_quat[b][None] + spread * 0.2 * rng.normal(size=(count, m.nmocap, 4))
        mq *= rng.uniform(0.9, 1.1, (count, m.nmocap, 1)) / np.linalg.norm(mq, axis=2, keepdims=True)
        u = np.concatenate([u, np.concatenate([mp, mq], axis=2).reshape(count, -1)], axis=1)
    ub = u  # ctrl.bin rows; the oracle's ctrl stays u
    nxf = int(os.environ.get('XFRC', 0))  # XFRC=k: k random (body, force 3, torque 3) slots per state, ctrl.bin rows after the mocap rows (-DNXF)
    if nxf:
        rng = np.random.default_rng(4)
        xb = rng.integers(1, m.nbody, (count, nxf))
        fs = float(np.abs(m.opt.gravity).max() * m.body_subtreemass[1])  # forces of the order of the first tree's weight, torques that times its size
        ts = fs * float(m.stat.meansize)
        xw = (rng.uniform(-1, 1, (count, nxf, 6)) * np.array([fs] * 3 + [ts] * 3)).astype(np.float32).astype(np.float64)  # float32-exact: no input-rounding floor
        X = np.zeros((count, m.nbody, 6))
        for n in range(count):
            np.add.at(X[n], xb[n], xw[n])  # two slots on one body add, as the engine's slots do
        ub = np.concatenate([u, np.concatenate([xb[..., None], xw], axis=2).reshape(count, -1)], axis=1)
        ref = lambda *a, X=X: reference_fresh(*a, xfrc=X)
    eqact = bool(os.environ.get('EQACT'))  # EQACT=1: every equality (inactive at start too) on or off at random per state, ctrl.bin rows after the push slots (-DEQACT=1)
    if eqact:
        A = np.random.default_rng(5).integers(0, 2, (count, m.neq)).astype(float)  # every equality is exported, in MuJoCo's order
        ub = np.concatenate([ub, A], axis=1)
        ref = lambda *a, X=(X if nxf else None), A=A: reference_fresh(*a, xfrc=X, eqact=A)
    ub.astype(np.float32).T.copy().tofile(out / 'ctrl.bin')  # (nu + 7 nmocap + 7 nxf + neq if EQACT, flies); zeros below the actuated stage
    if feet:  # only the exported pairs collide; actuators off below S3
        s = load(oracle or mjcf)
        if stage in ('contact', 'equality'):
            s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        on = set(pairs(s, FEET, convex)[0])
        for i in range(s.npair):
            if i not in on:
                s.pair_margin[i] = -1e10
        want = ref(s, fn, tuple(f for f in fields if f not in ('qacc_euler', 'qpos_next')) + ('qpos', 'qvel'), q, v, u)
        want['qacc_euler'] = (want.pop('qvel') - v) / m.opt.timestep
        want['qpos_next'] = want.pop('qpos')
        tendon_dense(s, want, q, v, count)
        d = mujoco.MjData(s)  # ncon: contacts that carry constraint rows (MuJoCo also lists excluded ones)
        ambiguous = np.zeros(count, int)  # pairs whose deepest vertex sits within float32 resolution of the margin
        switch = np.zeros(count)  # per state: the constraint row nearest its active-set switch, |J qacc - aref| / max |aref|
        live = [i for i in range(s.npair) if s.pair_margin[i] > -1 and s.geom_type[s.pair_geom1[i]] == mujoco.mjtGeom.mjGEOM_PLANE]
        noslip, s.opt.noslip_iterations = s.opt.noslip_iterations, 0  # switch: Newton's qacc (noslip moves it off the switch)
        def moving(b):
            return any(s.body_dofnum[a] > 0 for a in _ancestors(s, b))
        nconv = np.zeros(count, int)  # DIAG: contacts of box or ellipsoid geoms per state
        ncross = np.zeros(count, int)  # DIAG: contacts whose two geoms sit in different moving trees
        for n in range(count):
            d.qpos[:], d.qvel[:], d.ctrl[:] = q[n], v[n], u[n]
            mujoco.mj_forward(s, d)
            want['ncon'][n] = sum(c.efc_address >= 0 for c in d.contact[:d.ncon])
            nconv[n] = sum(c.efc_address >= 0 and s.geom_type[c.geom1] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_MESH) for c in d.contact[:d.ncon])  # convex-block shapes (stones, walls, flight primitives, mesh-mesh)
            ncross[n] = sum(c.efc_address >= 0 and s.body_rootid[s.geom_bodyid[c.geom1]] != s.body_rootid[s.geom_bodyid[c.geom2]] and moving(s.geom_bodyid[c.geom1]) and moving(s.geom_bodyid[c.geom2]) for c in d.contact[:d.ncon])
            jar = np.zeros(d.nefc)
            mujoco.mj_mulJacVec(s, d, jar, d.qacc)
            switch[n] = (np.abs(jar - d.efc_aref)[d.ne:].min() / max(np.abs(d.efc_aref).max(), 1e-30)) if d.nefc > d.ne and fx == FLOOR_X else 1.0  # equality rows never switch; pyramidal rows only: a cone has no row to switch
            for i in live:
                g1, g2 = s.pair_geom1[i], s.pair_geom2[i]
                a, c = s.mesh_vertadr[s.geom_dataid[g2]], s.mesh_vertnum[s.geom_dataid[g2]]
                depth = ((s.mesh_vert[a:a + c] @ d.geom_xmat[g2].reshape(3, 3).T + d.geom_xpos[g2] - d.geom_xpos[g1]) @ d.geom_xmat[g1].reshape(3, 3)[:, 2]).min()
                ambiguous[n] += abs(depth - s.pair_margin[i]) < 4 * np.spacing(np.float32(np.abs(d.geom_xpos[g2]).max()))
        s.opt.noslip_iterations = noslip
        if convex:  # the gate must meet convex contacts: no vacuous pass on states that touch no stone or wall
            ok, need = int((nconv > 0).sum()), int(os.environ.get('CONVEX_MIN', count // 4))
            print(f"{'PASS' if ok >= need else 'FAIL'} convex contacts present in {ok}/{count} states (at least {need})")
            if ok < need:
                sys.exit(1)
        if os.environ.get('CROSS_MIN'):  # the cross gate must meet cross-tree contacts (a world or one-tree pair would pass vacuously)
            ok, need = int((ncross > 0).sum()), int(os.environ['CROSS_MIN'])
            print(f"{'PASS' if ok >= need else 'FAIL'} cross-tree contacts present in {ok}/{count} states (at least {need})")
            if ok < need:
                sys.exit(1)
        # float32 floor: MuJoCo on the float32-rounded state (the engine's actual input) vs MuJoCo on the float64 state.
        # A contact force at solref 2e-4 moves 2.4e-5 of its scale from input rounding alone, so a field's bar is
        # max(REL, FLOOR_X x its floor); every field still has to beat 1e-5 unless float32 inputs cannot.
        r32 = lambda x: x.astype(np.float32).astype(np.float64)
        alt = ref(s, fn, tuple(f for f in fields if f not in ('qacc_euler', 'qpos_next', 'ncon')) + ('qpos', 'qvel'), r32(q), r32(v), r32(u))
        alt['qacc_euler'] = (alt.pop('qvel') - r32(v)) / m.opt.timestep  # same derivation as the reference's
        alt['qpos_next'] = alt.pop('qpos')
        tendon_dense(s, alt, r32(q), r32(v), count, show=False)
        floor = {f: float(np.abs(alt[f] - want[f]).max() / max(np.abs(want[f]).max(), 1e-30)) for f in alt}
        if stage == 'equality':  # same lift as the contact-free branch below, and proof that the fixture is mixed
            floor['qacc_euler'] = max(floor['qacc_euler'], EQ_EULER_BAR / FLOOR_X)
            print(f"equality + contacts: contact rows active in {int((want['ncon'] > 0).sum())}/{count} states")
        if k:  # K-step loop: the state after k chained mj_step calls, ctrl held; its own float32 floor
            chain = ','.join(['mj_step'] * k)
            want = ref(s, chain, ('qpos', 'qvel'), q, v, u)
            alt = ref(s, chain, ('qpos', 'qvel'), r32(q), r32(v), r32(u))
            floor = {f: float(np.abs(alt[f] - want[f]).max() / np.abs(want[f]).max()) for f in alt}
            # Contact chains are chaotic: a state whose own MuJoCo trajectory moves by more than the bar when the START is perturbed by
            # a 1e-6 relative amount (the engine's per-step float32 error) cannot be compared after k steps (a wall contact lost or
            # gained a step earlier changes qvel by 10-40 % of its range; the float32-rounding floor above misses it for states whose
            # rounding happens not to tip). Such states are dropped, at most half, and printed.
            dev = np.zeros(count)
            for sd in range(CHAOS_SEEDS):  # a contact-count flip (1 vs 5 contacts) is rare per draw: many draws, not three
                r = np.random.default_rng(100 + sd)
                pert = reference(s, chain, ('qpos', 'qvel'), q * (1 + 1e-6 * r.uniform(-1, 1, q.shape)), v * (1 + 1e-6 * r.uniform(-1, 1, v.shape)), u)
                for f in ('qpos', 'qvel'):
                    dev = np.maximum(dev, np.abs(pert[f] - want[f]).max(1) / np.abs(want[f]).max() / max(REL, FLOOR_X * floor[f]))
            unstable = dev > 1
            fields = ('qpos', 'qvel')
        print('float32 input floor: ' + ', '.join(f'{f} {e:.2e}' for f, e in floor.items()))
    elif stage in ('equality', 'mocap', 'tendon_limit', 'tendon_eq', 'joint_limit'):
        s = load(mjcf)
        s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
        ref = tuple(f for f in fields if f not in ('qacc_euler', 'qpos_next')) + ('qpos', 'qvel')
        r32 = lambda x: x.astype(np.float32).astype(np.float64)

        def run(qq, vv, pp, qm):  # mj_oracle.reference with the mocap pose set per state
            d, o = mujoco.MjData(s), {f: [] for f in ref}
            for n in range(count):
                d.qpos[:], d.qvel[:] = qq[n], vv[n]
                if s.nmocap:
                    d.mocap_pos[:], d.mocap_quat[:] = pp[n], qm[n]
                if eqact:
                    d.eq_active[:] = A[n]
                mujoco.mj_step(s, d)
                for f in ref:
                    o[f].append(np.array(getattr(d, f)))
            return {f: np.array(a) for f, a in o.items()}
        want = run(q, v, mp, mq)
        alt = run(r32(q), r32(v), None if mp is None else r32(mp), None if mq is None else r32(mq))  # alt: the float32-rounded input
        tendon_dense(s, want, q, v, count)
        tendon_dense(s, alt, r32(q), r32(v), count, show=False)
        for w, r in ((want, v), (alt, r32(v))):
            w['qacc_euler'] = (w.pop('qvel') - r) / m.opt.timestep
            w['qpos_next'] = w.pop('qpos')
        floor = {f: float(np.abs(alt[f] - want[f]).max() / max(np.abs(want[f]).max(), 1e-30)) for f in alt}
        # qacc_euler = M^-1 (qfrc_smooth + qfrc_constraint) turns the constraint forces' float32 error (qfrc_constraint within 5e-6 of its
        # max) into 4.6e-5 of its own max: the rows' forces are D (J qacc - aref), a difference of terms 10-100x larger than itself, and
        # M^-1 weighs the light rotational dofs 1000x. The bar is lifted to EQ_EULER_BAR; a wrong term moves it by 1e-1 (mutants).
        floor['qacc_euler'] = max(floor['qacc_euler'], EQ_EULER_BAR / FLOOR_X)
        print('float32 input floor: ' + ', '.join(f'{f} {e:.2e}' for f, e in floor.items()))
    elif stage in ('step', 'sptendon'):
        s = load(mjcf)
        s.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
        want = ref(s, fn, tuple(f for f in fields if f not in ('qacc_euler', 'qpos_next')) + ('qpos', 'qvel'), q, v)
        want['qacc_euler'] = (want.pop('qvel') - v) / m.opt.timestep
        want['qpos_next'] = want.pop('qpos')
        tendon_dense(s, want, q, v, count)
    else:
        want = ref(m, fn, fields, q, v)
    if oracle and 'geom_xpos' in want:  # world geoms are constants: take the engine model's (the heightfield sits elsewhere than its twin plane)
        w = np.flatnonzero(m.geom_bodyid == 0)
        want['geom_xpos'].reshape(count, -1, 3)[:, w] = m.geom_pos[w]
    want = {f: v[:, 1:] if f in WORLD_FIRST else v for f, v in want.items()}  # the engine skips body 0, the world
    q.astype(np.float32).T.copy().tofile(out / 'qpos.bin')  # (nq, flies): flies fastest, as the engine reads it
    v.astype(np.float32).T.copy().tofile(out / 'qvel.bin')  # (nv, flies)
    binary = out / f'{stage}_{count}'
    flags = (out / 'sizes.txt').read_text().split() + ([f'-DNXF={nxf}'] if nxf else []) + (['-DEQACT=1'] if eqact else [])
    subprocess.run([str(Path.home() / 'iTools/bin/mojo-cached'), 'build', *flags, f'-DFLIES={count}', f'-DSTAGE={number}', f'-DK={k}', f"-DWAVE={os.environ.get('WAVE', 0)}", f"-DDUMP={os.environ.get('DUMP', 0)}", f"-DCPU={os.environ.get('CPU', 0)}", *os.environ.get('MOJO_DEFS', '').split(), str(ENGINE / 'smooth.mojo'), '-o', str(binary)], check=True)
    if os.environ.get('BUILD_ONLY'):  # tools/check_all.sh phase 1: export and build on the CPU, run nothing
        print(f'built {binary}')
        return
    subprocess.run([str(binary), str(out) + '/'], check=True)
    if os.environ.get('ENGINE_ONLY'):  # engine run only (loopk: K=10 vs 10 x K=1 byte equality), no MuJoCo comparison
        print(f'engine only: {stage} K={k} ran, nothing compared with MuJoCo')
        return
    rows = np.fromfile(out / ('state.bin' if k else 'smooth.bin'), np.float32).reshape(-1, count).T
    if k and os.environ.get('SAVE_STATES'):  # the engine's own state after k steps, as a STATES file for a force check on states the engine reached
        np.savez(os.environ['SAVE_STATES'], qpos=rows[:, :m.nq].astype(np.float64), qvel=rows[:, m.nq:m.nq + m.nv].astype(np.float64))
    sizes = [want[f][0].size for f in fields]
    if rows.shape[1] != sum(sizes):
        raise SystemExit(f'FAIL {stage}: engine wrote {rows.shape[1]} values per fly, MuJoCo {sum(sizes)}')
    got = np.split(rows, np.cumsum(sizes)[:-1], axis=1)
    if k and m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):  # sleep: the asleep sets (qvel exactly zero) must agree, and be a mix
        aw, ag = (want['qvel'].reshape(count, -1) == 0).all(1), (got[fields.index('qvel')] == 0).all(1)
        print(f'sleep: asleep after {k} steps MuJoCo {int(aw.sum())}/{count}, engine {int(ag.sum())}/{count}, sets differ in {int((aw != ag).sum())} states')
        if (aw != ag).any() or not count // 4 <= aw.sum() <= 3 * count // 4:
            raise SystemExit(f'FAIL {stage}: asleep sets differ or MuJoCo asleep outside 25..75 % of the states')
        qa = fields.index('qpos')  # a sleeper's qpos is frozen at its sleeping step: one step early or late moves it by ~5e-4 (measured 1.3e-7)
        e = float(np.abs(got[qa][aw] - want['qpos'].reshape(count, -1)[aw]).max() / np.abs(want['qpos']).max())
        print(f"{'PASS' if e < SLEEP_QPOS_BAR else 'FAIL'} sleepers' qpos within {SLEEP_QPOS_BAR:g}: max rel err {e:.2e}")
        if e >= SLEEP_QPOS_BAR:
            sys.exit(1)
    floor = locals().get('floor', {})
    if os.environ.get('DIAG'):  # per state: error of each field vs the field's largest value, worst states first, with the convex contacts there
        unst = locals().get('unstable', np.zeros(count, bool)); nconv = locals().get('nconv', np.zeros(count, int))
        for f, g in zip(fields, got):
            e = np.abs(g - want[f].reshape(count, -1)).max(1) / max(np.abs(want[f]).max(), 1e-30)
            print(f'  DIAG {f} K={k}: median {np.median(e):.1e}, worst states ' + ', '.join(f'{n} {e[n]:.1e} (convex {nconv[n]}{", chaotic" if unst[n] else ""})' for n in np.argsort(-e)[:6]))
            if f in ('qacc', 'qfrc_constraint'):  # the worst state's worst entries (dof index, engine, MuJoCo)
                n = int(np.argmax(e)); w = want[f].reshape(count, -1)[n]; dd = np.abs(g[n] - w)
                print(f'  DIAG {f} state {n} worst dofs: ' + ', '.join(f'{j} ({g[n][j]:.4g} vs {w[j]:.4g})' for j in np.argsort(-dd)[:6]))
        if 'ncon' in fields:
            i = fields.index('ncon'); off = [(n, int(want['ncon'].reshape(count)[n]), int(got[i][n, 0])) for n in range(count) if int(got[i][n, 0]) != int(want['ncon'].reshape(count)[n])]
            print(f'  DIAG ncon mismatches {len(off)}/{count} (state, MuJoCo, engine): {off[:10]}')
    bad = []
    if k and 'unstable' in locals():
        print(f"{'PASS' if unstable.mean() <= 0.5 else 'FAIL'} chaotic states dropped {int(unstable.sum())}/{count} (MuJoCo itself moves by more than the bar over {k} steps under a 1e-6 relative start perturbation; at most half)")
        if unstable.mean() > 0.5:
            sys.exit(1)
        want = {f: np.asarray(a).reshape(count, -1)[~unstable] if f in ('qpos', 'qvel') else a for f, a in want.items()}
        got = [g[~unstable] for g in got]
    if 'ambiguous' in locals() and 'ncon' in fields:  # a contact at the margin to within float32 resolution may go either way, up to 3 rows
        i = fields.index('ncon')
        off = np.abs(got[i][:, 0] - want['ncon'].reshape(count)) > 3 * ambiguous
        if os.environ.get('DIAG') and 'nconv' in locals():  # per state: convex contacts, ncon want/got, qacc error
            qi = fields.index('qacc')
            e = np.abs(got[qi] - want['qacc'].reshape(count, -1)).max(1) / max(np.abs(want['qacc']).max(), 1e-30)
            for n in np.argsort(-e)[:12]:
                print(f'  DIAG state {n}: convex contacts {nconv[n]}, ncon want {int(want["ncon"][n])} got {int(got[i][n, 0])}, qacc err {e[n]:.2e}, ambiguous plane pairs {int(ambiguous[n])}')
        print(f'{"PASS" if not off.any() else "FAIL"} ncon         {int(off.sum())} states off beyond float32-ambiguous pairs '
              f'({int(ambiguous.sum())} ambiguous pairs in {int((ambiguous > 0).sum())} states)')
        bad += ['ncon'] if off.any() else []
        got, fields = got[:i] + got[i + 1:], fields[:i] + fields[i + 1:]
    if 'switch' in locals() and 'qfrc_constraint' in fields:  # a row within the qacc bar of its active-set switch
        # may land on either side in float32: one row's force then redistributes over the stance (fly ramp state 37:
        # |jar| 1.8e-5 of scale, force error 2.4e-4 along that row's direction). Those states skip qfrc_constraint only.
        i = fields.index('qfrc_constraint')
        w = want['qfrc_constraint'].reshape(count, -1)
        over = np.abs(got[i] - w).max(1) / max(np.abs(w).max(), 1e-30) > max(REL, fx * floor.get('qfrc_constraint', 0))
        flip = (switch < max(REL, fx * floor.get('qacc', 0))) & over  # only states the bar would fail count
        if os.environ.get('DIAG'):  # per state: qfrc_constraint error vs the field's largest value, nearest active-set switch
            e = np.abs(got[i] - w).max(1) / max(np.abs(w).max(), 1e-30)
            print(f"  DIAG qfrc_constraint bar {max(REL, fx * floor.get('qfrc_constraint', 0)):.1e}, switch bar {max(REL, fx * floor.get('qacc', 0)):.1e}: "
                  + ', '.join(f'state {n} err {e[n]:.1e} switch {switch[n]:.1e}' for n in np.argsort(-e)[:5]))
        print(f'{"PASS" if flip.mean() <= 0.1 else "FAIL"} active-set ambiguous states over the bar {int(flip.sum())}/{count} '
              f'(qfrc_constraint not compared there; at most 10 %)')
        bad += ['active-set'] if flip.mean() > 0.1 else []
        compare({'qfrc_constraint (ambiguous states)': (want['qfrc_constraint'].reshape(count, -1)[flip], got[i][flip])}, 1.0) if flip.any() else None
        want['qfrc_constraint'] = want['qfrc_constraint'].reshape(count, -1)[~flip]
        got[i] = got[i][~flip]
    # mj_Euler factors qH only when some dof has damping (and eulerdamp is on); otherwise it solves with qLD and qH stays 0
    undamped = m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER and (
        not m.dof_damping.any() or m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    skip = STALE_RK4 if undamped or m.opt.integrator in (mujoco.mjtIntegrator.mjINT_RK4, mujoco.mjtIntegrator.mjINT_IMPLICIT) else ()
    bad += [b for f, g in zip(fields, got) if f not in skip for b in compare({f: (want[f].reshape(len(g), -1), g)}, max(REL, fx * floor.get(f, 0)))]
    if bad:
        print(f'FAIL {len(bad)}/{len(fields)} {stage} fields: {" ".join(bad)}')
        sys.exit(1)
    lifted = ', '.join(f'{f} {fx * e:.1e}' for f, e in floor.items() if fx * e > REL)
    print(f'PASS {stage}' + (f' K={k}' if k else '') + f': {len(fields)} fields x {count} states within {REL:g}' + (f' (float32-floor bars: {lifted})' if lifted else ''))


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'airborne':
        sys.exit(print(airborne(args[0], args[1], *map(float, args[2:]))))
    if cmd == 'meshes':
        sys.exit(0 if meshes(*args) <= 1e-4 else 'FAIL meshes: decode error above 1e-4 of part size')
    import os
    export(*args) if cmd == 'export' else check(args[0], args[1], args[2], *map(int, args[3:]), oracle=os.environ.get('ORACLE'))
