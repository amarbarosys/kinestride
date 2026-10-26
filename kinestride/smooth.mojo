"""Smooth dynamics, ported from MuJoCo 3.9.0 src/engine/engine_core_smooth.c (Apache-2.0), batched:
one GPU thread per animal. STAGE 1: mj_kinematics1 + the inertia and geom part of mj_kinematics2.
STAGE 2: + mj_comPos (subtree_com, cinert, cdof) + mj_crb (sparse M in MuJoCo's own M_rowadr layout).
STAGE 3: + mj_factorM (mj_factorI: sparse L'DL of M in place, qLD, qLDiagInv).
STAGE 4: + mj_comVel (cvel, cdof_dot), mj_passive (joint springs, dof dampers), mj_rne without qacc (qfrc_bias).
STAGE 5: + mj_Euler, contact-free: implicit dof damping (qH = M + h diag(damping)), qacc, mj_integratePos.
STAGE 6: + plane-mesh foot contacts, pyramidal cone, mj_solPrimal (Newton); STAGE 7: + actuators (joint servos,
adhesion bodies) from ctrl.bin (NU x FLIES).

Model arrays come from kinestride/model.py export (mi.bin int32, mf.bin float32, field order
INTS / FLOATS there); sizes are -D constants, so any MJCF body (fly, spider) builds its own binary.
Orientation is kept as quaternions only (MuJoCo also builds 3x3 xmat; rotating by the quaternion is
the same map and stores 4 floats, not 9). Mocap bodies and the sleep filter are not ported.

Layout: every per-fly array is [row][fly], flies fastest, so a warp reads 32 neighbouring flies
in one transaction. Model arrays are shared by all flies and stay in cache.

usage: smooth DIR/      reads DIR/mi.bin, mf.bin, qpos.bin (NQ x FLIES), qvel.bin (NV x FLIES); writes DIR/smooth.bin, rows:
       xpos, xquat, xipos (bodies 1..), geom_xpos, xanchor, xaxis; STAGE 2 adds subtree_com,
       cinert (bodies 1..), cdof, crb (bodies 1..), M; STAGE 3 adds qLD, qLDiagInv; STAGE 4 adds cvel (bodies
       1..), cdof_dot, qfrc_passive, qfrc_bias; STAGE 5 adds qH, qHDiagInv, qacc_euler, qpos_next; prints us per launch.
       -DK=k: k steps per launch, state kept on the card; the state after the first launch goes to DIR/state.bin
       (qpos rows, then qvel rows), the state after 10 steps to DIR/state10.bin when K divides 10
"""
from std.ffi import external_call
from std.math import cos, exp, sin, sqrt
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major
from world import NP, PW, height_grad
from convex import Sp, ConSet, collide, collide_hfield, primitive_pair, Height, HCons, ELLIPSOID, MESH, BOX, T as CT, S as CS, V as CVEC  # convex.mojo's scalar type (float64 with -DCCD64=1, which the export sets)
import game_state as game

comptime NB = get_defined_int["NBODY"]()
comptime NJ = get_defined_int["NJNT"]()
comptime NQ = get_defined_int["NQ"]()
comptime NG = get_defined_int["NGEOM"]()
comptime NV = get_defined_int["NV"]()
comptime NC = get_defined_int["NC"]()
comptime F = get_defined_int["FLIES"]()
comptime STAGE = get_defined_int["STAGE", 1]()
comptime CPU = get_defined_int["CPU", 0]() == 1  # host build: the same step per animal in a loop, no device (one interactive animal: GPU latency, not throughput)
comptime WAVE = get_defined_int["WAVE", 0]() == 1 and not CPU  # one animal per 32-lane wave (live): serial code runs on every lane
comptime LN = 32 if WAVE else 1  # alike; loops written range(lane(), n, LN) split over lanes, sync() before others read
comptime BLOCK = 32 if WAVE else 64
comptime GRID = F if WAVE else (F + 63) // 64
comptime UNION = get_defined_int["UNION", 0]() == 1 and not WAVE  # contact loops walk every pair (slot) ascending and skip the ones this animal lacks: the lanes of a wave visit the same pair together (docs/plans/UNION_WALK.md)
comptime DUMP = get_defined_int["DUMP", 0]() == 1  # diagnostics (row_diag.py): write every scratch row and their offsets
comptime STOP = get_defined_int["STOP", 99]()
comptime INTEGRATOR = get_defined_int["INTEGRATOR", 0]()  # opt.integrator: 0 Euler, 1 RK4, 3 implicitfast (model.py export; absent = Euler)
comptime SOLVER = get_defined_int["SOLVER", 2]()  # opt.solver: 0 PGS, 1 CG, 2 Newton (model.py export; absent = Newton)
comptime K = get_defined_int["K", 0]()  # K > 0 (STAGE >= 5): K steps per launch, qpos/qvel written back in place
comptime STEPTOL = Float32(get_defined_int["STEPTOL_E7", 1 if get_defined_int["ELLIPTIC", 0]() == 1 else 10]()) * 1e-7  # Newton stops when a step moves qacc less than this
comptime LSFLOOR = Float32(get_defined_int["LSFLOOR_E7", 10]()) * 1e-7  # line-search slope floor, relative to alpha = 0  # phase timing (phases.sh): return after phase STOP
# mi.bin: body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_qposadr, geom_bodyid
comptime PARENT = 0
comptime JNTADR = NB
comptime JNTNUM = 2 * NB
comptime JTYPE = 3 * NB
comptime QADR = 3 * NB + NJ
comptime GBODY = 3 * NB + 2 * NJ
# STAGE 2: body_rootid, jnt_dofadr, dof_parentid, dof_bodyid, M_rowadr, M_rownnz
comptime ROOT = 3 * NB + 2 * NJ + NG
comptime DOFADR = ROOT + NB
comptime DPARENT = DOFADR + NJ
comptime DBODY = DPARENT + NV
comptime MADR = DBODY + NV
comptime MNNZ = MADR + NV
comptime MCOL = MNNZ + NV  # STAGE 3: M_colind
comptime BDOFNUM = MCOL + NC  # STAGE 4: body_dofnum, body_dofadr
comptime BDOFADR = BDOFNUM + NB
comptime NI0 = BDOFADR + NB
# STAGE 6 contact block (model.py feet): per pair 11 ints, hull first edges, edge lists, dof chains, 2 solver ints
comptime NCP = get_defined_int["NCP", 0]()
comptime DIMX = get_defined_int["DIMX", 3]()  # the largest pair condim: 4 and 6 add torsional and rolling rows (per pair condim and friction 2-4 then exported)
comptime NHV = get_defined_int["NHV", 0]()
comptime NHE = get_defined_int["NHE", 0]()
comptime NCH = get_defined_int["NCH", 0]()
comptime NH = get_defined_int["NH", 0]()  # H's nonzeros (M's plus the fill of pairs that couple two dof chains)
comptime CPH = NI0
comptime CFIRST = CPH + 11 * NCP
comptime CEDGE = CFIRST + NHV
comptime CCHAIN = CEDGE + NHE
comptime CSIGN = CCHAIN + NCH
comptime HADR = CSIGN + NCH  # H's pattern: rowadr, rownnz, colind (same layout as M's at MADR), then M entry -> H entry
comptime MTOH = HADR + 2 * NV + NH
comptime CITER = MTOH + NC
comptime NU = get_defined_int["NU", 0]()
comptime NMUSCLE = get_defined_int["NMUSCLE", 0]()
comptime CDIM = CITER + 3  # DIMX > 3: per pair condim (1, 3, 4, 6)
comptime CACT = CDIM + (NCP if DIMX > 3 else 0)  # STAGE 7 (model.py actuators): per actuator trntype, dof or body, ctrllimited, forcelimited, qpos address
comptime NI1 = CACT + 5 * NU if STAGE >= 7 else (CACT if STAGE >= 6 else NI0)
# model.py fluid() then tendons(), appended after the actuators: fluid ints per ellipsoid row (body, geom) then the inertia-box bodies
comptime NFLROW = get_defined_int["NFLROW", -1]()  # -1: no fluid block (walking models: the code below folds out)
comptime NFLBOX = get_defined_int["NFLBOX", 0]()
comptime NFLUID = NFLROW >= 0
comptime FLI = NI1
comptime NTEN = get_defined_int["NTEN", 0]()  # fixed tendons; tendon ints: wrap start and count per tendon, then each wrap's dof, then its qpos address
comptime NWRAP = get_defined_int["NWRAP", 0]()
comptime TNI = FLI + (2 * NFLROW + NFLBOX if NFLUID else 0)
comptime NCV = get_defined_int["NCV", 0]()  # convex pairs (box or ellipsoid vs mesh, convex.mojo): per pair its kind, between the tendon and equality blocks
# model.py spatial() right after the tendon arrays: per wrap type, body, sidesite body; per tendon the frame body; floats 11 per wrap
comptime SPATIAL = get_defined_int["SPATIAL", 0]() == 1
comptime CSP = TNI + 2 * NTEN + 2 * NWRAP
comptime NI2 = CSP + (3 * NWRAP + NTEN if SPATIAL else 0)
# equality constraints (model.py eq_block, last block): per constraint 12 ints, then every chain, then its masks
comptime NEQ = get_defined_int["NEQ", 0]()
comptime NEQR = get_defined_int["NEQR", 0]()  # rows: connect 3, weld 6, joint 1, then the tendon limit rows (unilateral)
comptime NEQB = get_defined_int["NEQB", 0]()  # bilateral rows; a row r >= NEQL (joint or tendon limit) costs 0.5 D jar^2 only while jar < 0
comptime NEQL = get_defined_int["NEQL", NEQB]()  # rows NEQB..NEQL: dof then tendon friction loss (mj_instantiateFriction), Huber cost bounded by frictionloss
comptime NEQFE = get_defined_int["NEQFE", 0]()  # the equality of friction row NEQB (one row per friction constraint)
comptime NEQCH = get_defined_int["NEQCH", 0]()
comptime NEQJ = get_defined_int["NEQJ", 0]()  # Jacobian entries: rows x chain length, summed
comptime EQH = 13  # the 13th: MuJoCo eq_active0 (1 for a limit row)
comptime EQF = 31
comptime NMM = get_defined_int["NMM", 0]() == 1  # a mesh-mesh pair exists: per pair geom1's hull vertex and first-edge offsets after the convex block
comptime PCLIP = get_defined_int["PCLIP", 0]() == 1  # margin-0 box-mesh / mesh-mesh pairs: the hulls' polygons (model.py poly_block) end the convex block
comptime NPOLY = get_defined_int["NPOLY", 0]()
comptime NPMAP = get_defined_int["NPMAP", 0]()
comptime NPVERT = get_defined_int["NPVERT", 0]()
comptime CPMV = NI2 + ((6 if NMM else 4) * NCP if NCV > 0 else 0)  # PCLIP: per hull vertex polygon adr and count, polygon ids, per polygon vertex adr and count, vertices
comptime CEQH = CPMV + (2 * NHV + NPMAP + 2 * NPOLY + NPVERT if PCLIP else 0)  # the convex block (kind, two shape kinds, condim per pair) sits between the tendon and equality blocks
comptime CEQC = CEQH + EQH * NEQ
comptime CEQM = CEQC + NEQCH
# mocap bodies (model.py mocap_block, after the equality block): body_mocapid per body; each animal's mocap pose (pos 3,
# quat 4 per mocap body) rides in rows NU.. of ctrl.bin
comptime NMOCAP = get_defined_int["NMOCAP", 0]()
comptime NXF = get_defined_int["NXF", 0]()  # MuJoCo xfrc_applied, sparse: slots per animal in ctrl.bin after the mocap rows, each body id, force 3, torque 3
comptime EQACT = get_defined_int["EQACT", 0]()  # 1: per-animal equality on/off, NEQ ctrl.bin rows after the push slots (MuJoCo eq_active)
comptime NEQE = get_defined_int["NEQE", NEQ]()  # MuJoCo's equalities: the block's first NEQE entries; friction and limit rows follow and have no eq_active
comptime NUC = NU + 7 * NMOCAP + 7 * NXF + (NEQE if EQACT else 0)
comptime CMOC = CEQM + NEQCH
# gravcomp (model.py gravcomp_block, last): ints jnt_actgravcomp per joint, floats body_gravcomp per body
comptime GRAVCOMP = get_defined_int["GRAVCOMP", 0]() == 1
comptime CGC = CMOC + (NB if NMOCAP > 0 else 0)
comptime CIMP = CGC + (NJ if GRAVCOMP else 0)
# full implicit integrator (model.py implicit_block, INTEGRATOR 2, last): per body B_rownnz, B_rowadr, ancestor dofs; B_colind; per dof D_rowadr, D_diag, D_rownnz; D_colind; mapM2D
comptime IMPL = INTEGRATOR == 2
comptime NBB = get_defined_int["NBB", 0]()  # MuJoCo nB: body-dof nonzeros
comptime NDD = get_defined_int["NDD", 0]()  # MuJoCo nD: dof-dof nonzeros
comptime BRN = CIMP
comptime BRA = BRN + NB
comptime BANC = BRA + NB
comptime BCOL = BANC + NB
comptime DRA = BCOL + NBB
comptime DDG = DRA + NV
comptime DRN = DDG + NV
comptime DCOL = DRN + NV
comptime M2D = DCOL + NDD
comptime DPAIR = M2D + NDD  # (k, p) -> D entry or -1 (model.py pair_table), NV x NV
# implicit midpoint of standalone free bodies (model.py mid_block, implicitfast, after the implicit block): per candidate body, dof address, aligned (ipos = 0)
comptime NMID = get_defined_int["NMID", 0]()
comptime CMID = DPAIR + NV * NV if IMPL else CIMP
comptime MPAIR = CMID + 3 * NMID  # implicitfast (model.py mpair_block): (k, p) -> M lower entry or -1, NV x NV
# joint-level actuator force limits (model.py actfrc_block, last): per limited joint its dofadr; floats lo, hi at AFRF
comptime NAFRC = get_defined_int["NAFRC", 0]()
comptime CAFR = MPAIR + (NV * NV if get_defined_int["MPAIR", 0]() == 1 else 0)
comptime NI = CAFR + NAFRC
comptime CKIND = NI2  # NCV: per pair 0 plane/sphere/tangent, 1 ellipsoid-mesh, 2 box-mesh, 3 exact heightfield, 4 primitive pair (GJK + EPA), 5 capsule-capsule, 6 sphere-capsule, 7 sphere-cylinder
comptime CGT = CKIND + NCP  # NCV: per pair the convex.mojo shape kinds of geom1 and geom2
comptime CCND = CKIND + 3 * NCP  # NCV: per pair condim (1: one frictionless row, 3: the pyramid)
comptime CMM = CKIND + 4 * NCP  # NMM: per pair geom1's hull vertex offset and first-edge offset (mesh-mesh pairs; 0 0 for the others)
# mf.bin: body_pos, body_quat, jnt_pos, jnt_axis, qpos0, body_ipos, body_iquat, geom_pos, geom_quat
comptime BPOS = 0
comptime BQUAT = 3 * NB
comptime JPOS = 7 * NB
comptime JAXIS = 7 * NB + 3 * NJ
comptime QPOS0 = 7 * NB + 6 * NJ
comptime IPOS = 7 * NB + 6 * NJ + NQ
comptime GPOS = 14 * NB + 6 * NJ + NQ
comptime IQUAT = IPOS + 3 * NB
comptime GQUAT = GPOS + 3 * NG
# STAGE 2: body_mass, body_subtreemass, body_inertia, dof_armature
comptime MASS = 14 * NB + 6 * NJ + NQ + 7 * NG
comptime SUBMASS = MASS + NB
comptime INERTIA = SUBMASS + NB
comptime ARMATURE = INERTIA + 3 * NB
comptime DAMP = ARMATURE + NV  # STAGE 4: dof_damping, jnt_stiffness, qpos_spring, opt.gravity, opt.timestep
comptime STIFF = DAMP + NV
comptime QSPRING = STIFF + NJ
comptime GRAV = QSPRING + NQ
comptime TSTEP = GRAV + 3
comptime NFL0 = TSTEP + 1
comptime CPF = NFL0  # STAGE 6: per pair margin, gap, friction 2, solref 2, solimp 5, invweight, rbound 2 and 1; hull verts; 3 opts
comptime CVERT = CPF + 14 * NCP
comptime COPT = CVERT + 3 * NHV
comptime HFG = get_defined_int["HFGEOM", -1]()  # heightfield ground geom (model.py feet), else none
comptime CFR = COPT + 5  # DIMX > 3: per pair friction 2-4 (torsional, rolling 1 and 2)
comptime HFO = CFR + (3 * NCP if DIMX > 3 else 0)  # world.mojo params (NP)
comptime CACF = HFO + (NP if HFG >= 0 else 0)  # STAGE 7: per actuator gain, bias 0-2, ctrlrange 2, forcerange 2, gear
comptime CMUSF = CACF + 9 * NU  # optional eleven-float parameter rows, packed only for annotated muscles
comptime NFL1 = CMUSF + 11 * NMUSCLE if STAGE >= 7 else (CACF if STAGE >= 6 else NFL0)
comptime FLF = NFL1  # fluid floats: density, viscosity, wind 3, then per ellipsoid row semi-axes 3 and the 12 geom_fluid coefs
comptime TNF = FLF + (5 + 15 * NFLROW if NFLUID else 0)  # tendon floats: wrap coefs, then stiffness, damping, lengthspring 2 per tendon
comptime CSPF = TNF + NWRAP + 4 * NTEN
comptime NFL2 = CSPF + (11 * NWRAP if SPATIAL else 0)
comptime CCV = NFL2  # NCV: opt.ccd_tolerance, heightfield size 0-3 and origin z (HFEXACT), then per pair geom1's size (3), then per pair geom2's size (3)
comptime CPN = NFL2 + (6 + 6 * NCP if NCV > 0 else 0)  # PCLIP: per polygon its normal
comptime CEQF = CPN + (3 * NPOLY if PCLIP else 0)  # per equality: anchors 0 and 1, quaternions 0 and 1, torquescale, solref, solimp, invweight 2, joint data 5, qpos0 2
comptime CGF = CEQF + EQF * NEQ
comptime GAME = get_defined_int["GAME", 0]() == 1 and STAGE >= 7  # game layer per step (~/Projects/science/fly-connectome/docs/plans/GAME_LAYER.md): energy, health, stamina, dead
comptime SLEEP = get_defined_int["SLEEP", 0]() == 1  # mj_sleep (model.py sleep_block): opt.sleep_tolerance, then dof_length per dof (the velocity weights)
comptime SLPF = CGF + (NB if GRAVCOMP else 0)
comptime AFRF = SLPF + (1 + NV if SLEEP else 0)  # NAFRC: jnt_actfrcrange lo, hi per limited joint
comptime GAMEF = AFRF + 2 * NAFRC  # GAME: rules appended to mf.bin: basal drain, work rate, health regen
comptime NFL = GAMEF + (3 if GAME else 0)
comptime NFLT = NFL + (2 + HFNR * HFNC if HTAB else 0)  # mf.bin length: HTAB appends the heightfield table last (CHT == NFL)
comptime HFNR = get_defined_int["HFNR", 2]()  # HFEXACT: grid rows and columns of the MuJoCo hfield (the engine's grid is virtual and endless, same step)
comptime HFNC = get_defined_int["HFNC", 2]()
comptime HTAB = get_defined_int["HTAB", 0]() == 1  # HFEXACT: node heights from a table in mf (model.py export HTAB=1), the grid clamped like MuJoCo's
comptime CHT = get_defined_int["CHT", 0]()  # HTAB: table centre x, y (world), then HFNR x HFNC node heights above the hfield origin, row r along +y
# smooth.bin rows (per animal): xpos, xquat, xipos of bodies 1.., geom_xpos, xanchor, xaxis
comptime OXQ = 3 * (NB - 1)
comptime OXI = 7 * (NB - 1)
comptime OG = 10 * (NB - 1)
comptime OA = 10 * (NB - 1) + 3 * NG
comptime OX = 10 * (NB - 1) + 3 * NG + 3 * NJ
comptime OSC = 10 * (NB - 1) + 3 * NG + 6 * NJ  # STAGE 2: subtree_com, cinert, cdof, crb, M
comptime OCI = OSC + 3 * NB
comptime OCD = OCI + 10 * (NB - 1)
comptime OCRB = OCD + 6 * NV
comptime OM = OCRB + 10 * (NB - 1)
comptime OLD = OM + NC  # STAGE 3: qLD, qLDiagInv
comptime OLDI = OLD + NC
comptime OCV = OLDI + NV  # STAGE 4: cvel (bodies 1..), cdof_dot, qfrc_passive, qfrc_bias
comptime OCDD = OCV + 6 * (NB - 1)
comptime OPAS = OCDD + 6 * NV
comptime OBIAS = OPAS + NV
comptime OQH = OBIAS + NV  # STAGE 5: qH, qHDiagInv, qacc_euler, qpos_next
comptime OQHI = OQH + NC
comptime OQACC = OQHI + NV
comptime OQPOS = OQACC + NV
comptime ONCON = OQPOS + NQ  # STAGE 6: ncon, qacc (constrained), qfrc_constraint
comptime OCQACC = ONCON + 1
comptime OQFC = OCQACC + NV
comptime OAF = OQFC + NV  # STAGE 7: actuator_force, qfrc_actuator
comptime OQACT = OAF + NU
comptime OMUS = OQACT + NV  # persistent activation and fatigue rows, two per muscle
comptime NOUT = OSC if STAGE == 1 else (OM + NC if STAGE == 2 else (OLDI + NV if STAGE == 3 else (OQH if STAGE == 4 else (OQPOS + NQ if STAGE == 5 else (OQFC + NV if STAGE == 6 else OMUS + 2 * NMUSCLE)))))
comptime OTLEN = NOUT  # SPATIAL: tendon outputs after the stage's own: ten_length, ten_velocity, ten_J (dense, NV per tendon)
comptime OTVEL = OTLEN + NTEN
comptime OTJ = OTVEL + NTEN
comptime NOUTX = OTJ + NTEN * NV if SPATIAL else NOUT
comptime OCACC = NOUTX  # scratch rows after the output: cacc, cfrc_body per body (world included)
comptime OCFRC = OCACC + 6 * NB
comptime NSLOT = get_defined_int["NSLOT", 0]()  # contact slots: 3 per plane-mesh pair (MuJoCo maxplanemesh), 1 per sphere pair
comptime NJE = get_defined_int["NJE", 0]()  # Jacobian entries: 4 rows x slots x chain length, summed over pairs
comptime ELL = get_defined_int["ELLIPTIC", 0]() == 1  # model.py feet(): opt.cone elliptic (3 rows per contact, normal then 2 friction), else pyramidal
comptime RS = max(4, DIMX) if ELL else (4 if DIMX <= 3 else 2 * (DIMX - 1))  # row stride per contact slot: pyramid 2 (condim - 1), elliptic condim, at least 4
comptime NR = DIMX if ELL else RS  # rows used per contact slot; a pair's rows past its own condim stay empty (J, D, aref 0)
comptime NPOOL = get_defined_int["NPOOL", 0]()  # NCV: slots shared by the convex and exact-heightfield pairs of one animal (the last NPOOL of NSLOT); each such pair takes what it needs per step
comptime JPOOL = get_defined_int["JPOOL", 0]()  # their Jacobian entries (the last JPOOL of NJE), bump-allocated per step
comptime NSTATIC = NSLOT - NPOOL
comptime JSTATIC = NJE - JPOOL
comptime NEFC = RS * NSLOT  # condim 3: 4 pyramid rows or 3 elliptic rows per contact
comptime OFLW = OCFRC + 6 * NB  # NFLUID scratch: fluid wrench per body about its root's subtree_com
comptime OVALID = OFLW + (6 * NB if NFLUID else 0)  # STAGE 6 scratch: slot valid, per row D aref jar jv state, J rows, solver vectors, H
comptime OD = OVALID + NSLOT
comptime OAREF = OD + NEFC
comptime OJAR = OAREF + NEFC
comptime OJV = OJAR + NEFC
comptime OST = OJV + NEFC
comptime OJ = OST + NEFC
comptime OQSM = OJ + NJE
comptime OQACCS = OQSM + NV
comptime OMA = OQACCS + NV
comptime OGRAD = OMA + NV
comptime OSRCH = OGRAD + NV
comptime OMV = OSRCH + NV
comptime OHN = OMV + NV
comptime OHNI = OHN + NH
comptime ODIAG = OHNI + NV  # Newton iterations, line-search evaluations (printed by main)
comptime OFRC = ODIAG + 3  # Newton diagnostics, then noslip scratch
comptime OFRN = OFRC + NEFC  # elliptic noslip: the new friction forces (OFRC keeps Newton's)
comptime OZ = OFRN + (NEFC if ELL else 0)
comptime OW = OZ + NV
comptime OWC = OW + 2 * NV  # noslip per step: w of every valid row (laid out as J), then per row pair A 00 01 11, base 0 1
comptime OAC = OWC + NJE
comptime OACT = OAC + 10 * NSLOT  # valid slots: count, then their ids ascending (row loops skip the ~95 % empty)
comptime OPART = OACT + 1 + NSLOT  # WAVE: per-lane partial sums, 3 x LN
comptime OWARM = OPART + (6 if ELL else 3) * LN  # K > 0: last step's qacc (MuJoCo qacc_warmstart), persists across launches
comptime OPA = OWARM + NV  # active pairs (some valid slot) of this step: count, then ids ascending: the solver loops visit these only
comptime OSD = OPA + 1 + NCP  # NCV: per pair this step's slot base, slot count and Jacobian base (pooled pairs only)
comptime OPP = OSD + 3 * NCP  # NCV: pool slots and pool Jacobian entries taken so far this step
comptime NROW1 = OPP + 2 if STAGE >= 6 else (OVALID if STAGE >= 4 else NOUT)
comptime OEQD = NROW1  # equality rows: D, aref, jar, jv, then the Jacobian entries (like OJ)
comptime OEQA = OEQD + NEQR
comptime OEQJAR = OEQA + NEQR
comptime OEQJV = OEQJAR + NEQR
comptime OEQJ = OEQJV + NEQR
comptime OGC = OEQJ + NEQJ  # qfrc_gravcomp (GRAVCOMP)
comptime NROW0 = OGC + (NV if GRAVCOMP else 0)
comptime ORKQ = NROW0  # RK4 scratch (INTEGRATOR 1 only): X0 qpos, then per stage velocity v_s and acceleration F_s, then one scaled velocity
comptime ORKV = ORKQ + NQ
comptime ORKF = ORKV + 4 * NV
comptime ORKW = ORKF + 4 * NV
comptime OQLU = NROW0  # INTEGRATOR 2: qLU (D layout), unfinished-row counters of the LU, then Dcvel, Dcacc, Dcfrcbody (6 rows per B entry)
comptime ORM = OQLU + NDD
comptime ODCV = ORM + NV
comptime ODCA = ODCV + 6 * NBB
comptime ODCF = ODCA + 6 * NBB
comptime NROWB = ORKW + NV if INTEGRATOR == 1 else (ODCF + 6 * NBB if IMPL else NROW0)
comptime OQMV = NROWB  # NMID: qvel_mid of every dof (the velocity mj_advance integrates positions with), then one eligibility flag per candidate
comptime NROW2 = NROWB + (NV + NMID if NMID > 0 else 0)
comptime OSOL = NROW2  # CG scratch (SOLVER 1 only): M^-1 grad, then grad and M^-1 grad of the last iteration
comptime OCGM = OSOL
comptime OCGO = OCGM + NV
comptime OCGMO = OCGO + NV
comptime OGF = OSOL  # PGS scratch (SOLVER 0 only), per row (contacts, then the equality block): force, live rows and visiting order (2 per row), 1/diag of AR; then z = L^-T J' f, the equality rows' w (laid out as OEQJ), the island union-find (a parent per body)
comptime OGP = OGF + NEFC + NEQR
comptime OGA = OGP + 2 * (NEFC + NEQR)
comptime OGZ = OGA + NEFC + NEQR
comptime OGWE = OGZ + NV
comptime OGU = OGWE + NEQJ
comptime SOLROWS = 3 * NV if SOLVER == 1 else (4 * (NEFC + NEQR) + NV + NEQJ + NB if SOLVER == 0 else 0)
comptime OGAME = OSOL + SOLROWS  # GAME: four float32 rows per animal, energy health stamina dead; all zero = not started
comptime OSLP = OGAME + (4 if GAME else 0)  # SLEEP: quiet checks in a row (MuJoCo tree_asleep + 11), SLEPT once asleep
comptime OQAPP = OSLP + (1 if SLEEP else 0)  # NXF: qfrc_applied (MuJoCo mj_xfrcAccumulate), NV rows
comptime OUPM = OQAPP + (NV if NXF > 0 else 0)  # UNION: masks of OPA (pairs) then OACT (slots), 24 bits per row as exact float32 integers
comptime OUSM = OUPM + (NCP + 23) // 24
comptime NROW = OUSM + (NSLOT + 23) // 24 if UNION and STAGE >= 6 else OUPM
comptime SLEPT: Float32 = 11  # one past mjMINAWAKE (10) quiet checks
comptime LI = row_major[NI]()
comptime LF = row_major[NFLT]()
comptime LQ = row_major[NQ * F]()
comptime LO = row_major[NROW * F]()
comptime LV = row_major[NV * F]()
comptime V3 = SIMD[DType.float32, 4]  # x, y, z, unused
comptime Q4 = SIMD[DType.float32, 4]  # w, x, y, z


@always_inline
def mul(a: Q4, b: Q4) -> Q4:
    return Q4(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
              a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


@always_inline
def rot(q: Q4, v: V3) -> V3:
    """MuJoCo mju_rotVecQuat: t = w v + q x v, v + 2 q x t."""
    var t = V3(q[0] * v[0] + q[2] * v[2] - q[3] * v[1], q[0] * v[1] + q[3] * v[0] - q[1] * v[2], q[0] * v[2] + q[1] * v[1] - q[2] * v[0], 0)
    return V3(v[0] + 2 * (q[2] * t[2] - q[3] * t[1]), v[1] + 2 * (q[3] * t[0] - q[1] * t[2]), v[2] + 2 * (q[1] * t[1] - q[2] * t[0]), 0)


@always_inline
def cross(a: V3, b: V3) -> V3:
    return V3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0)


@always_inline
def unit(q: Q4) -> Q4:
    var n = sqrt((q * q).reduce_add())
    return q / n if n > 1e-15 else Q4(1, 0, 0, 0)


comptime TI = TileTensor[DType.int32, type_of(LI), MutAnyOrigin]
comptime TF = TileTensor[DType.float32, type_of(LF), MutAnyOrigin]
comptime TQ = TileTensor[DType.float32, type_of(LQ), MutAnyOrigin]
comptime TV = TileTensor[DType.float32, type_of(LV), MutAnyOrigin]
comptime LU = row_major[max(NUC, 1) * F]()
comptime TU = TileTensor[DType.float32, type_of(LU), MutAnyOrigin]
comptime TO = TileTensor[DType.float32, type_of(LO), MutAnyOrigin]


@always_inline
def m3(mf: TF, k: Int) -> V3:
    return V3(mf[k], mf[k + 1], mf[k + 2], 0)


@always_inline
def m4(mf: TF, k: Int) -> Q4:
    return Q4(mf[k], mf[k + 1], mf[k + 2], mf[k + 3])


@always_inline
def put(res: TO, f: Int, row: Int, v: SIMD[DType.float32, 4], n: Int):
    for c in range(n):
        res[(row + c) * F + f] = v[c]


@always_inline
def get(res: TO, f: Int, row: Int) -> Float32:
    return res[row * F + f]


@always_inline
def has(res: TO, f: Int, mask: Int, i: Int) -> Bool:
    """UNION: bit i of the 24-bit masks in rows mask.. (OUPM pairs, OUSM slots)."""
    return (Int(get(res, f, mask + i // 24)) >> (i % 24)) & 1 != 0


@always_inline
def npairs(res: TO, f: Int) -> Int:
    """Length of the pair walk: every pair id with UNION, else this animal's OPA list."""
    comptime if UNION:
        return NCP
    return Int(get(res, f, OPA))


@always_inline
def pick(res: TO, f: Int, ip: Int) -> Int:
    """Step ip of the pair walk: the pair id, or -1 when UNION visits a pair this animal lacks. Both walks are ascending."""
    comptime if UNION:
        return ip if has(res, f, OUPM, ip) else -1
    return Int(get(res, f, OPA + 1 + ip))


@always_inline
def nslots(res: TO, f: Int) -> Int:
    """Length of the slot walk: every slot id with UNION, else this animal's OACT list."""
    comptime if UNION:
        return NSLOT
    return Int(get(res, f, OACT))


@always_inline
def slot(res: TO, f: Int, i: Int) -> Int:
    """Step i of the slot walk: the slot id, or -1 when UNION visits a slot this animal lacks. Both walks are ascending."""
    comptime if UNION:
        return i if has(res, f, OUSM, i) else -1
    return Int(get(res, f, OACT + 1 + i))


@always_inline
def col(q: Q4, k: Int) -> V3:
    """Column k of the rotation matrix of q (MuJoCo xmat[k], xmat[k+3], xmat[k+6])."""
    return rot(q, basis(k))


@always_inline
def basis(k: Int) -> V3:
    var e = V3(0)
    e[k] = 1
    return e


@always_inline
def body(res: TO, f: Int, i: Int) -> Tuple[V3, Q4]:
    """World frame of body i (0 = world), already written by this thread."""
    if i == 0:
        return (V3(0), Q4(1, 0, 0, 0))
    var r = 3 * (i - 1)
    var s = OXQ + 4 * (i - 1)
    return (V3(res[r * F + f], res[(r + 1) * F + f], res[(r + 2) * F + f], 0),
            Q4(res[s * F + f], res[(s + 1) * F + f], res[(s + 2) * F + f], res[(s + 3) * F + f]))


comptime S6 = SIMD[DType.float32, 8]  # spatial vector: rotation 0..2, translation 3..5


@always_inline
def get6(res: TO, f: Int, row: Int) -> S6:
    return S6(get(res, f, row), get(res, f, row + 1), get(res, f, row + 2), get(res, f, row + 3), get(res, f, row + 4), get(res, f, row + 5), 0, 0)


@always_inline
def put6(res: TO, f: Int, row: Int, v: S6):
    for c in range(6):
        res[(row + c) * F + f] = v[c]


@always_inline
def cross_motion(a: S6, v: S6) -> S6:
    """MuJoCo mju_crossMotion: a x v for motion vectors."""
    return S6(-a[2] * v[1] + a[1] * v[2], a[2] * v[0] - a[0] * v[2], -a[1] * v[0] + a[0] * v[1],
              -a[2] * v[4] + a[1] * v[5] - a[5] * v[1] + a[4] * v[2], a[2] * v[3] - a[0] * v[5] + a[5] * v[0] - a[3] * v[2],
              -a[1] * v[3] + a[0] * v[4] - a[4] * v[0] + a[3] * v[1], 0, 0)


@always_inline
def cross_force(a: S6, v: S6) -> S6:
    """MuJoCo mju_crossForce: a x* v for force vectors."""
    return S6(-a[2] * v[1] + a[1] * v[2] - a[5] * v[4] + a[4] * v[5], a[2] * v[0] - a[0] * v[2] + a[5] * v[3] - a[3] * v[5],
              -a[1] * v[0] + a[0] * v[1] - a[4] * v[3] + a[3] * v[4], -a[2] * v[4] + a[1] * v[5], a[2] * v[3] - a[0] * v[5],
              -a[1] * v[3] + a[0] * v[4], 0, 0)


@always_inline
def mul_inert(res: TO, f: Int, row: Int, v: S6) -> S6:
    """MuJoCo mju_mulInertVec with the 10-float com inertia stored at rows row..row+9."""
    var i = SIMD[DType.float32, 16](0)
    for k in range(10):
        i[k] = get(res, f, row + k)
    return S6(i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
              i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
              i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
              i[8] * v[1] - i[7] * v[2] + i[9] * v[3], i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
              i[7] * v[0] - i[6] * v[1] + i[9] * v[5], 0, 0)


@always_inline
def atan2p(y: Float32, x: Float32) -> Float32:
    """Atan2 for y >= 0 (amdgpu has no atan2f): 2 atan(y / (r + x)), atan reduced to |z| <= tan(pi/16), Taylor to z^9."""
    var r = sqrt(x * x + y * y)
    if r + x <= 0:
        return 3.14159265358979 if r > 0 else 0.0
    var z = y / (r + x)
    var flip = z > 1
    if flip:
        z = 1 / z
    z = z / (1 + sqrt(1 + z * z))
    z = z / (1 + sqrt(1 + z * z))
    var z2 = z * z
    var a = 4 * z * (1 + z2 * (-1.0 / 3 + z2 * (1.0 / 5 + z2 * (-1.0 / 7 + z2 / 9))))
    return 2 * (1.5707963267949 - a if flip else a)


@always_inline
def lane() -> Int:
    comptime if WAVE:
        return Int(thread_idx.x)
    return 0


@always_inline
def sync():
    comptime if WAVE:
        barrier()


def factorI(mi: TI, res: TO, f: Int, L: Int, LI_: Int, A: Int = MADR):
    """MuJoCo mj_factorI: sparse L'DL of rows L.. in place, backward over rows k: L(i, 0..i) -= L(i, 0..i) L(k, i) / L(k, k),
    then L(k, :) /= L(k, k); 1 / L(k, k) to rows LI_.. Pattern at A: rowadr, rownnz, colind (M's, or H's at HADR).
    WAVE: row k's updates (j, c), c <= j, are independent (a row's pattern is its chain, so row j's is a prefix of k's)
    and spread over lanes (probes/proto_factor.mojo, 10x at 1,024 animals)."""
    comptime if WAVE:
        sync()
        for k in range(NV - 1, -1, -1):
            var start = Int(mi[A + k])
            var diag = Int(mi[A + NV + k]) - 1
            var invD = 1 / get(res, f, L + start + diag)
            for t in range(lane(), diag * (diag + 1) // 2, LN):
                var j = Int((sqrt(Float32(8 * t + 1)) - 1) / 2)
                if (j + 1) * (j + 2) // 2 <= t:  # float rounding of the triangular root
                    j += 1
                var c = t - j * (j + 1) // 2
                var i = Int(mi[A + 2 * NV + start + j])
                res[(L + Int(mi[A + i]) + c) * F + f] += get(res, f, L + start + c) * (-get(res, f, L + start + j) * invD)
            sync()
            for c in range(lane(), diag, LN):
                res[(L + start + c) * F + f] = get(res, f, L + start + c) * invD
            res[(LI_ + k) * F + f] = invD
            sync()
        return
    for k in range(NV - 1, -1, -1):
        var start = Int(mi[A + k])
        var diag = Int(mi[A + NV + k]) - 1
        var end = start + diag
        var invD = 1 / get(res, f, L + end)
        res[(LI_ + k) * F + f] = invD
        for adr in range(end - 1, start - 1, -1):
            var i = Int(mi[A + 2 * NV + adr])
            var scl = -get(res, f, L + adr) * invD
            var ri = Int(mi[A + i])
            for c in range(Int(mi[A + NV + i])):
                res[(L + ri + c) * F + f] += get(res, f, L + start + c) * scl
        for c in range(diag):
            res[(L + start + c) * F + f] = get(res, f, L + start + c) * invD


def solveLD(mi: TI, res: TO, f: Int, X: Int, L: Int, LI_: Int, A: Int = MADR):
    """MuJoCo mj_solveLD, one vector: x <- (L'DL)^-1 x for the factor in rows L.., 1/D in rows LI_.."""
    for i in range(NV - 1, -1, -1):  # x <- L^-T x
        var start = Int(mi[A + i])
        var x_i = get(res, f, X + i)
        for adr in range(start, start + Int(mi[A + NV + i]) - 1):
            res[(X + Int(mi[A + 2 * NV + adr])) * F + f] -= get(res, f, L + adr) * x_i
    for i in range(NV):  # x <- D^-1 x
        res[(X + i) * F + f] *= get(res, f, LI_ + i)
    for i in range(NV):  # x <- L^-1 x
        var start = Int(mi[A + i])
        var dot: Float32 = 0
        for adr in range(start, start + Int(mi[A + NV + i]) - 1):
            dot += get(res, f, L + adr) * get(res, f, X + Int(mi[A + 2 * NV + adr]))
        res[(X + i) * F + f] -= dot


def mulM(mi: TI, res: TO, f: Int, Y: Int, X: Int):
    """MuJoCo mju_mulSymVecSparse with M (rows OM..): y = M x, lower triangle stored."""
    for i in range(NV):
        res[(Y + i) * F + f] = 0
    for i in range(NV):
        var start = Int(mi[MADR + i])
        var end = start + Int(mi[MNNZ + i]) - 1
        var x_i = get(res, f, X + i)
        var acc = get(res, f, OM + end) * x_i
        for adr in range(start, end):
            var j = Int(mi[MCOL + adr])
            acc += get(res, f, OM + adr) * get(res, f, X + j)
            res[(Y + j) * F + f] += get(res, f, OM + adr) * x_i
        res[(Y + i) * F + f] += acc


@always_inline
def pow2(a: Float32) -> Float32:
    return a * a


@always_inline
def pow4(a: Float32) -> Float32:
    return (a * a) * (a * a)


@always_inline
def local_wrench(mi: TI, mf: TF, res: TO, f: Int, b: Int, p: V3, q: Q4) -> S6:
    """MuJoCo mj_objectVelocity(flg_local = 1) of the frame (p, q) fixed to body b, minus wind: mju_transformSpatial of cvel
    from the root's subtree_com to p, rotated into the frame (engine_support.c, engine_util_spatial.c:478)."""
    var cv = get6(res, f, OCV + 6 * (b - 1))
    var r = Int(mi[ROOT + b])
    var dif = p - V3(get(res, f, OSC + 3 * r), get(res, f, OSC + 3 * r + 1), get(res, f, OSC + 3 * r + 2), 0)
    var w = V3(cv[0], cv[1], cv[2], 0)
    var v = V3(cv[3], cv[4], cv[5], 0) - cross(dif, w) - m3(mf, FLF + 2)
    var qi = Q4(q[0], -q[1], -q[2], -q[3])
    var lw = rot(qi, w)
    var lv = rot(qi, v)
    return S6(lw[0], lw[1], lw[2], lv[0], lv[1], lv[2], 0, 0)


@always_inline
def add_wrench(mi: TI, res: TO, f: Int, b: Int, p: V3, q: Q4, lfrc: S6):
    """Fold mj_applyFT: the local frame wrench rotated to world, taken to the root's subtree_com, added to the weld body's
    row. qfrc[v] = cdof[v] . W is the J'F + J'T chain walk, because cdof is a motion vector about that same com."""
    var bw = b
    while bw > 0 and mi[JNTNUM + bw] == 0:
        bw = Int(mi[PARENT + bw])
    if bw == 0:
        return
    var r = Int(mi[ROOT + b])
    var tq = rot(q, V3(lfrc[0], lfrc[1], lfrc[2], 0))
    var fo = rot(q, V3(lfrc[3], lfrc[4], lfrc[5], 0))
    tq += cross(p - V3(get(res, f, OSC + 3 * r), get(res, f, OSC + 3 * r + 1), get(res, f, OSC + 3 * r + 2), 0), fo)
    put6(res, f, OFLW + 6 * bw, get6(res, f, OFLW + 6 * bw) + S6(tq[0], tq[1], tq[2], fo[0], fo[1], fo[2], 0, 0))


def inertia_box(mi: TI, mf: TF, res: TO, f: Int, b: Int):
    """MuJoCo mj_inertiaBoxFluidModel (engine_passive.c:1054): box from the principal inertia, viscous sphere drag plus
    quadratic drag, about the body's centre of mass in the body frame."""
    var rho = mf[FLF]
    var mu = mf[FLF + 1]
    var m = mf[MASS + b]
    var I = m3(mf, INERTIA + 3 * b)
    var bx = V3(sqrt(max(Float32(1e-15), I[1] + I[2] - I[0]) / m * 6), sqrt(max(Float32(1e-15), I[0] + I[2] - I[1]) / m * 6),
                sqrt(max(Float32(1e-15), I[0] + I[1] - I[2]) / m * 6), 0)
    var bd = body(res, f, b)
    var p = V3(get(res, f, OXI + 3 * (b - 1)), get(res, f, OXI + 3 * (b - 1) + 1), get(res, f, OXI + 3 * (b - 1) + 2), 0)
    var q = mul(bd[1], m4(mf, IQUAT + 4 * b))  # ximat is the inertial frame's rotation (mj_objectVelocity, engine_core_util.c:827)
    var lv = local_wrench(mi, mf, res, f, b, p, q)
    var fr = S6(0)
    if mu > 0:
        var diam = (bx[0] + bx[1] + bx[2]) / 3
        for k in range(3):
            fr[k] = -3.14159265358979 * diam * diam * diam * mu * lv[k]
            fr[3 + k] = -3 * 3.14159265358979 * diam * mu * lv[3 + k]
    if rho > 0:
        fr[3] -= 0.5 * rho * bx[1] * bx[2] * abs(lv[3]) * lv[3]
        fr[4] -= 0.5 * rho * bx[0] * bx[2] * abs(lv[4]) * lv[4]
        fr[5] -= 0.5 * rho * bx[0] * bx[1] * abs(lv[5]) * lv[5]
        fr[0] -= rho * bx[0] * (pow4(bx[1]) + pow4(bx[2])) * abs(lv[0]) * lv[0] / 64
        fr[1] -= rho * bx[1] * (pow4(bx[0]) + pow4(bx[2])) * abs(lv[1]) * lv[1] / 64
        fr[2] -= rho * bx[2] * (pow4(bx[0]) + pow4(bx[1])) * abs(lv[2]) * lv[2] / 64
    add_wrench(mi, res, f, b, p, q, fr)


def ellipsoid(mi: TI, mf: TF, res: TO, f: Int, row: Int):
    """MuJoCo mj_ellipsoidFluidModel for one geom (engine_passive.c:1114) with mj_addedMassForces (:1176) and
    mj_viscousForces (:1230): added mass, Magnus and Kutta lift, blunt/slender drag, viscous resistance."""
    var rho = mf[FLF]
    var mu = mf[FLF + 1]
    var c = FLF + 5 + 15 * row
    var s = m3(mf, c)
    var coef = mf[c + 3]
    var blunt = mf[c + 4]
    var slender = mf[c + 5]
    var angd = mf[c + 6]
    var kutta = mf[c + 7]
    var magnus = mf[c + 8]
    var vm = m3(mf, c + 9)
    var vi = m3(mf, c + 12)
    var b = Int(mi[FLI + 2 * row])
    var g = Int(mi[FLI + 2 * row + 1])
    var bd = body(res, f, b)
    var q = mul(bd[1], m4(mf, GQUAT + 4 * g))
    var p = V3(get(res, f, OG + 3 * g), get(res, f, OG + 3 * g + 1), get(res, f, OG + 3 * g + 2), 0)
    var lv = local_wrench(mi, mf, res, f, b, p, q)
    var w = V3(lv[0], lv[1], lv[2], 0)
    var v = V3(lv[3], lv[4], lv[5], 0)
    # mj_addedMassForces
    var lin_mom = rho * vm * v
    var ang_mom = rho * vi * w
    var tq = cross(lin_mom, v) + cross(ang_mom, w)
    var fo = cross(lin_mom, w)
    # mj_viscousForces
    var volume = 4.0 / 3.0 * 3.14159265358979 * s[0] * s[1] * s[2]
    var dmax = max(max(s[0], s[1]), s[2])
    var dmin = min(min(s[0], s[1]), s[2])
    var dmid = s[0] + s[1] + s[2] - dmax - dmin
    var amax = 3.14159265358979 * dmax * dmid
    fo += cross(w, v) * (magnus * rho * volume)
    var pden = pow4(s[1] * s[2]) * pow2(v[0]) + pow4(s[2] * s[0]) * pow2(v[1]) + pow4(s[0] * s[1]) * pow2(v[2])
    var pnum = pow2(s[1] * s[2] * v[0]) + pow2(s[2] * s[0] * v[1]) + pow2(s[0] * s[1] * v[2])
    var aproj = 3.14159265358979 * sqrt(pden / max(Float32(1e-15), pnum))
    var nrm = V3(pow2(s[1] * s[2]) * v[0], pow2(s[2] * s[0]) * v[1], pow2(s[0] * s[1]) * v[2], 0)
    var vn = sqrt((v * v).reduce_add())
    var cosa = pnum / max(Float32(1e-15), vn * pden)
    var circ = cross(nrm, v) * (kutta * rho * cosa * aproj)
    fo += cross(circ, v)
    var dsph = 2.0 / 3.0 * (s[0] + s[1] + s[2])
    var imax = 8.0 / 15.0 * 3.14159265358979 * dmid * pow4(dmax)
    var mom = V3(0)
    for k in range(3):
        var ii = 8.0 / 15.0 * 3.14159265358979 * s[k] * pow4(max(s[(k + 1) % 3], s[(k + 2) % 3]))
        mom[k] = w[k] * (angd * ii + slender * (imax - ii))
    var drag_lin = mu * 3 * 3.14159265358979 * dsph + rho * vn * (aproj * blunt + slender * (amax - aproj))
    var drag_ang = mu * 3.14159265358979 * dsph * dsph * dsph + rho * sqrt((mom * mom).reduce_add())
    tq -= drag_ang * w
    fo -= drag_lin * v
    add_wrench(mi, res, f, b, p, q, S6(tq[0], tq[1], tq[2], fo[0], fo[1], fo[2], 0, 0) * coef)


def fluid(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_fluid (engine_passive.c:842) into qfrc_passive: every body with mass takes the ellipsoid model if any of its
    geoms has one (the exported rows), else the inertia box. Wrenches go to one row per weld body, are summed leaf to root
    like mj_rne's cfrc, and qfrc_fluid[v] = cdof[v] . W[body of v]."""
    for b in range(NB):
        put6(res, f, OFLW + 6 * b, S6(0))
    for r in range(NFLROW):
        ellipsoid(mi, mf, res, f, r)
    for k in range(NFLBOX):
        inertia_box(mi, mf, res, f, Int(mi[FLI + 2 * NFLROW + k]))
    for i in range(NB - 1, 0, -1):
        var p = Int(mi[PARENT + i])
        if p > 0:
            put6(res, f, OFLW + 6 * p, get6(res, f, OFLW + 6 * p) + get6(res, f, OFLW + 6 * i))
    for v in range(NV):
        res[(OPAS + v) * F + f] += (get6(res, f, OCD + 6 * v) * get6(res, f, OFLW + 6 * Int(mi[DBODY + v]))).reduce_add()


comptime P4 = SIMD[DType.float32, 4]
comptime MINVAL: Float32 = 1e-15


@always_inline
def acosp(x: Float32) -> Float32:
    """Acos on [-1, 1] (amdgpu has no acosf): atan2(sqrt((1 - x)(1 + x)), x)."""
    var c = min(max(x, Float32(-1)), Float32(1))
    return atan2p(sqrt((1 - c) * (1 + c)), c)


@always_inline
def asinp(x: Float32) -> Float32:
    var c = min(max(x, Float32(-1)), Float32(1))
    var a = atan2p(abs(c), sqrt((1 - c) * (1 + c)))
    return a if c >= 0 else -a


@always_inline
def is_intersect(e: P4, sl: P4) -> Bool:
    """The mju_wrap is_intersect (engine_util_misc.c:34): do the segments end0-sol0 and end1-sol1 cross; e and sl hold (x0, y0, x1, y1)."""
    var det = (sl[3] - e[3]) * (sl[0] - e[0]) - (sl[2] - e[2]) * (sl[1] - e[1])
    if abs(det) < MINVAL:
        return False
    var a = ((sl[2] - e[2]) * (e[1] - e[3]) - (sl[3] - e[3]) * (e[0] - e[2])) / det
    var b = ((sl[0] - e[0]) * (e[1] - e[3]) - (sl[1] - e[1]) * (e[0] - e[2])) / det
    return a >= 0 and a <= 1 and b >= 0 and b <= 1


@always_inline
def length_circle(p: P4, ind: Int, radius: Float32) -> Float32:
    """Curve length along the circle between p's two points (engine_util_misc.c:55); ind picks the long way round."""
    var n0 = sqrt(p[0] * p[0] + p[1] * p[1]); var n1 = sqrt(p[2] * p[2] + p[3] * p[3])
    var ax = p[0] / n0 if n0 > MINVAL else Float32(1); var ay = p[1] / n0 if n0 > MINVAL else Float32(0)
    var bx = p[2] / n1 if n1 > MINVAL else Float32(1); var by = p[3] / n1 if n1 > MINVAL else Float32(0)
    var angle = acosp(ax * bx + ay * by)
    var cr = p[1] * p[2] - p[0] * p[3]
    if (cr > 0 and ind != 0) or (cr < 0 and ind == 0):
        angle = 2 * 3.14159265358979 - angle
    return radius * angle


def wrap_circle(end: P4, hasside: Bool, side: P4, radius: Float32) -> Tuple[Float32, P4]:
    """MuJoCo wrap_circle (engine_util_misc.c:78): tangent points of the shorter (or sidesite-side) path around a circle,
    returns (curve length or -1, 2 x 2D points); side holds the 2D side point in its first two lanes."""
    var sqlen0 = end[0] * end[0] + end[1] * end[1]
    var sqlen1 = end[2] * end[2] + end[3] * end[3]
    var sqrad = radius * radius
    var none = P4(0)
    if sqlen0 < sqrad or sqlen1 < sqrad or radius < MINVAL:
        return (Float32(-1), none)
    var dx = end[2] - end[0]; var dy = end[3] - end[1]
    var dd = dx * dx + dy * dy
    if dd < MINVAL:
        return (Float32(-1), none)
    var a = -(dx * end[0] + dy * end[1]) / dd
    a = min(max(a, Float32(0)), Float32(1))
    var tx = a * dx + end[0]; var ty = a * dy + end[1]
    if tx * tx + ty * ty > sqrad and (not hasside or side[0] * tx + side[1] * ty >= 0):
        return (Float32(-1), none)
    var sqrt0 = sqrt(sqlen0 - sqrad); var sqrt1 = sqrt(sqlen1 - sqrad)
    var sol0 = P4(0); var sol1 = P4(0)
    var good0: Float32 = 0; var good1: Float32 = 0
    for i in range(2):
        var sgn: Float32 = 1 if i == 0 else -1
        var sol = P4((end[0] * sqrad + sgn * radius * end[1] * sqrt0) / sqlen0, (end[1] * sqrad - sgn * radius * end[0] * sqrt0) / sqlen0,
                     (end[2] * sqrad - sgn * radius * end[3] * sqrt1) / sqlen1, (end[3] * sqrad + sgn * radius * end[2] * sqrt1) / sqlen1)
        var good: Float32
        if hasside:
            var ux = sol[0] + sol[2]; var uy = sol[1] + sol[3]
            var un = sqrt(ux * ux + uy * uy)
            if un > MINVAL:
                ux = ux / un
                uy = uy / un
            else:
                ux = 1
                uy = 0
            good = ux * side[0] + uy * side[1]
        else:
            var vx = sol[0] - sol[2]; var vy = sol[1] - sol[3]
            good = -(vx * vx + vy * vy)
        if is_intersect(end, sol):
            good = -10000
        if i == 0:
            sol0 = sol
            good0 = good
        else:
            sol1 = sol
            good1 = good
    var pick = 0 if good0 > good1 else 1
    var pnt = sol0 if pick == 0 else sol1
    if is_intersect(end, pnt):
        return (Float32(-1), none)
    return (length_circle(pnt, pick, radius), pnt)


comptime D4 = SIMD[DType.float64, 4]


@always_inline
def atan2d(y: Float64, x: Float64) -> Float64:
    """Atan2 for y >= 0 in float64 (atan2p's reduction, 2 half-angle steps, Taylor to z^9: error ~1e-15)."""
    var r = sqrt(x * x + y * y)
    if r + x <= 0:
        return 3.14159265358979323846 if r > 0 else 0.0
    var z = y / (r + x)
    var flip = z > 1
    if flip:
        z = 1 / z
    z = z / (1 + sqrt(1 + z * z))
    z = z / (1 + sqrt(1 + z * z))
    var z2 = z * z
    var a = 4 * z * (1 + z2 * (-1.0 / 3 + z2 * (1.0 / 5 + z2 * (-1.0 / 7 + z2 / 9))))
    return 2 * (1.5707963267948966 - a if flip else a)


@always_inline
def asin64(x: Float64) -> Float64:
    var c = min(max(x, Float64(-1)), Float64(1))
    var a = atan2d(abs(c), sqrt((1 - c) * (1 + c)))
    return a if c >= 0 else -a


@always_inline
def acos64(x: Float64) -> Float64:
    var c = min(max(x, Float64(-1)), Float64(1))
    return atan2d(sqrt((1 - c) * (1 + c)), c)


def wrap_inside(end32: P4, radius32: Float32) -> Tuple[Float32, P4]:
    """MuJoCo wrap_inside (engine_util_misc.c:158), sidesite inside the circle: Newton on asin(A z) + asin(B z) - 2 asin(z) + G = 0
    with z starting at 1 - 1e-7 and stopping at |f| < 1e-6. In float32 asin(z) near 1 moves 1e-4 per ulp of z, so the loop cannot
    reach its tolerance (3 of 64 gate states failed): it runs in float64 (a few iterations, only for such a wrap). Returns
    (0 for a wrap, -1 for none, the wrap point twice)."""
    var end = D4(Float64(end32[0]), Float64(end32[1]), Float64(end32[2]), Float64(end32[3]))
    var radius = Float64(radius32)
    var minval: Float64 = 1e-15
    var zinit: Float64 = 1 - 1e-7
    var len0 = sqrt(end[0] * end[0] + end[1] * end[1]); var len1 = sqrt(end[2] * end[2] + end[3] * end[3])
    var dx = end[2] - end[0]; var dy = end[3] - end[1]
    var dd = dx * dx + dy * dy
    var none = P4(0)
    if len0 <= radius or len1 <= radius or radius < minval or len0 < minval or len1 < minval:
        return (Float32(-1), none)
    if dd > minval:
        var a = -(dx * end[0] + dy * end[1]) / dd
        if a > 0 and a < 1:
            var tx = end[0] + a * dx; var ty = end[1] + a * dy
            if sqrt(tx * tx + ty * ty) <= radius:
                return (Float32(-1), none)
    var mx = 0.5 * (end[0] + end[2]); var my = 0.5 * (end[1] + end[3])
    var mn = sqrt(mx * mx + my * my)
    if mn > minval:
        mx = mx / mn
        my = my / mn
    else:
        mx = 1
        my = 0
    var dflt = P4(Float32(radius * mx), Float32(radius * my), Float32(radius * mx), Float32(radius * my))
    var A = radius / len0; var B = radius / len1
    var cosG = (len0 * len0 + len1 * len1 - dd) / (2 * len0 * len1)
    if cosG < -1 + minval:
        return (Float32(-1), none)
    elif cosG > 1 - minval:
        return (Float32(0), dflt)
    var G = acos64(cosG)
    var z = zinit
    var fz = asin64(A * z) + asin64(B * z) - 2 * asin64(z) + G
    if fz > 0:
        return (Float32(0), dflt)
    var iter = 0
    while iter < 20 and abs(fz) > 1e-6:
        var df = A / max(minval, sqrt(1 - z * z * A * A)) + B / max(minval, sqrt(1 - z * z * B * B)) - 2 / max(minval, sqrt(1 - z * z))
        if df > -minval:
            return (Float32(0), dflt)
        var z1 = z - fz / df
        if z1 > z:
            return (Float32(0), dflt)
        z = z1
        fz = asin64(A * z) + asin64(B * z) - 2 * asin64(z) + G
        if fz > 1e-6:
            return (Float32(0), dflt)
        iter += 1
    if iter >= 20:
        return (Float32(0), dflt)
    var vx: Float64; var vy: Float64; var ang: Float64
    if end[0] * end[3] - end[1] * end[2] > 0:
        vx = end[0]
        vy = end[1]
        ang = asin64(z) - asin64(A * z)
    else:
        vx = end[2]
        vy = end[3]
        ang = asin64(z) - asin64(B * z)
    var vn = sqrt(vx * vx + vy * vy)
    if vn > minval:
        vx = vx / vn
        vy = vy / vn
    else:
        vx = 1
        vy = 0
    var ca = cos(Float32(ang)); var sa = sin(Float32(ang))  # no float64 sin/cos on the GPU; the angle is already converged
    var px = Float32(radius) * (ca * Float32(vx) - sa * Float32(vy)); var py = Float32(radius) * (sa * Float32(vx) + ca * Float32(vy))
    return (Float32(0), P4(px, py, px, py))


def wrap_geom(x0: V3, x1: V3, gpos: V3, gq: Q4, radius: Float32, cylinder: Bool, hasside: Bool, sidepos: V3) -> Tuple[Float32, V3, V3]:
    """MuJoCo mju_wrap (engine_util_misc.c:281): wrap the straight path x0-x1 around a sphere or cylinder (geom frame gpos, gq,
    radius): returns (wrap length or -1, the two tangent points in world coordinates)."""
    var gi = Q4(gq[0], -gq[1], -gq[2], -gq[3])
    var p0 = rot(gi, x0 - gpos); var p1 = rot(gi, x1 - gpos)
    var zero = V3(0)
    var n0 = sqrt((p0 * p0).reduce_add()); var n1 = sqrt((p1 * p1).reduce_add())
    if n0 < MINVAL or n1 < MINVAL:
        return (Float32(-1), zero, zero)
    var axis0 = V3(1, 0, 0, 0); var axis1 = V3(0, 1, 0, 0)
    if not cylinder:
        axis0 = p0 / n0
        var normal = cross(p0, p1)
        var nrm = sqrt((normal * normal).reduce_add())
        if nrm >= MINVAL:
            normal = normal / nrm
        else:
            var i = 0
            if abs(axis0[1]) > abs(axis0[0]) and abs(axis0[1]) > abs(axis0[2]):
                i = 1
            if abs(axis0[2]) > abs(axis0[0]) and abs(axis0[2]) > abs(axis0[1]):
                i = 2
            var a1 = V3(1, 1, 1, 0)
            a1[i] = 0
            normal = cross(axis0, a1)
            var nn = sqrt((normal * normal).reduce_add())
            normal = normal / nn if nn > MINVAL else V3(1, 0, 0, 0)
        axis1 = cross(normal, axis0)
        var an = sqrt((axis1 * axis1).reduce_add())
        axis1 = axis1 / an if an > MINVAL else V3(0, 1, 0, 0)
    var d = P4((p0 * axis0).reduce_add(), (p0 * axis1).reduce_add(), (p1 * axis0).reduce_add(), (p1 * axis1).reduce_add())
    var sd = P4(0)
    var s = V3(0)
    if hasside:
        s = rot(gi, sidepos - gpos)
        var sx = (s * axis0).reduce_add(); var sy = (s * axis1).reduce_add()
        var sn = sqrt(sx * sx + sy * sy)
        if sn > MINVAL:
            sx = sx / sn
            sy = sy / sn
        else:
            sx = 1
            sy = 0
        sd = P4(radius * sx, radius * sy, 0, 0)
    var res: Tuple[Float32, P4]
    if hasside and sqrt((s * s).reduce_add()) < radius:
        res = wrap_inside(d, radius)
    else:
        res = wrap_circle(d, hasside, sd, radius)
    var wlen = res[0]
    if wlen < 0:
        return (Float32(-1), zero, zero)
    var pnt = res[1]
    var r0 = axis0 * pnt[0] + axis1 * pnt[1]
    var r1 = axis0 * pnt[2] + axis1 * pnt[3]
    if cylinder:
        var L0 = sqrt((p0[0] - r0[0]) * (p0[0] - r0[0]) + (p0[1] - r0[1]) * (p0[1] - r0[1]))
        var L1 = sqrt((p1[0] - r1[0]) * (p1[0] - r1[0]) + (p1[1] - r1[1]) * (p1[1] - r1[1]))
        r0[2] = p0[2] + (p1[2] - p0[2]) * L0 / (L0 + wlen + L1)
        r1[2] = p0[2] + (p1[2] - p0[2]) * (L0 + wlen) / (L0 + wlen + L1)
        var height = abs(r1[2] - r0[2])
        wlen = sqrt(wlen * wlen + height * height)
    return (wlen, rot(gq, r0) + gpos, rot(gq, r1) + gpos)


@always_inline
def tpoint(mi: TI, mf: TF, qpos: TQ, res: TO, f: Int, bi: Int, local: V3, frame: V3) -> V3:
    """A body-fixed point in the tendon's frame (the floating origin of its path): xpos + R local, shifted by the body's tree origin minus the frame's."""
    var b = body(res, f, bi)
    return rot(b[1], local) + b[0] + (origin(mi, qpos, f, bi) - frame)


def tjac(mi: TI, qpos: TQ, res: TO, f: Int, row: Int, bi: Int, point: V3, dir: V3, scale: Float32, frame: V3):
    """J[row] += scale jacp(body, point)' dir over the body's dofs (mj_jacDifPair's endpoint Jacobian under the chain rule of
    mj_tendon): jacp = cdof_lin + cdof_ang x (point - subtree_com of the body's root)."""
    var r = 3 * Int(mi[ROOT + bi])
    var com = V3(get(res, f, OSC + r), get(res, f, OSC + r + 1), get(res, f, OSC + r + 2), 0) + (origin(mi, qpos, f, bi) - frame)
    var off = point - com
    var b = bi
    while b > 0:
        for d in range(Int(mi[BDOFADR + b]), Int(mi[BDOFADR + b]) + Int(mi[BDOFNUM + b])):
            var cd = OCD + 6 * d
            var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
            var lin = V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0)
            res[(row + d) * F + f] += scale * ((lin + cross(ang, off)) * dir).reduce_add()
        b = Int(mi[PARENT + b])


def tseg(mi: TI, qpos: TQ, res: TO, f: Int, row: Int, ba: Int, pa: V3, bb: Int, pb: V3, divisor: Float32, frame: V3):
    """One path segment between points in different bodies: J += (jac(bb, pb) - jac(ba, pa))' unit(pb - pa) / divisor."""
    if ba == bb:
        return
    var dif = pb - pa
    var n = sqrt((dif * dif).reduce_add())
    dif = dif / n if n > MINVAL else V3(1, 0, 0, 0)
    tjac(mi, qpos, res, f, row, bb, pb, dif, 1 / divisor, frame)
    tjac(mi, qpos, res, f, row, ba, pa, dif, -1 / divisor, frame)


def spatial(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_tendon (engine_core_smooth.c:922) for every tendon, fixed and spatial: ten_length, ten_J (dense rows OTJ) and
    ten_velocity = J qvel. A spatial path is a run of sites with sphere or cylinder wraps (mju_wrap, sidesite) and pulleys
    (divisor); a segment adds its endpoint Jacobian difference along its direction where its two ends sit in different bodies."""
    for t in range(NTEN):
        var adr = Int(mi[TNI + t]); var num = Int(mi[TNI + NTEN + t])
        var jr = OTJ + t * NV
        for v in range(NV):
            res[(jr + v) * F + f] = 0
        var L: Float32 = 0
        if Int(mi[CSP + adr]) == 1:  # fixed (joint wraps): length = sum coef q, J = coef
            for w in range(adr, adr + num):
                L += mf[TNF + w] * qpos[Int(mi[TNI + 2 * NTEN + NWRAP + w]) * F + f]
                res[(jr + Int(mi[TNI + 2 * NTEN + w])) * F + f] += mf[TNF + w]
        else:
            var frame = origin(mi, qpos, f, Int(mi[CSP + 3 * NWRAP + t]))
            var divisor: Float32 = 1
            var j = 0
            while j < num - 1:
                var w0 = adr + j
                var type0 = Int(mi[CSP + w0]); var type1 = Int(mi[CSP + w0 + 1])
                if type0 == 2 or type1 == 2:  # pulley
                    if type0 == 2:
                        divisor = mf[TNF + w0]
                    j += 1
                    continue
                var b0 = Int(mi[CSP + NWRAP + w0])
                var p0 = tpoint(mi, mf, qpos, res, f, b0, m3(mf, CSPF + 11 * w0), frame)
                var geom = type1 == 4 or type1 == 5
                var w1 = w0 + 2 if geom else w0 + 1
                var b1 = Int(mi[CSP + NWRAP + w1])
                var p1 = tpoint(mi, mf, qpos, res, f, b1, m3(mf, CSPF + 11 * w1), frame)
                var wlen: Float32 = -1
                var wa = V3(0); var wb = V3(0)
                var gb = 0
                if geom:
                    var wg = w0 + 1
                    gb = Int(mi[CSP + NWRAP + wg])
                    var gB = body(res, f, gb)
                    var gpos = rot(gB[1], m3(mf, CSPF + 11 * wg)) + gB[0] + (origin(mi, qpos, f, gb) - frame)
                    var gq = mul(gB[1], m4(mf, CSPF + 11 * wg + 3))
                    var sb = Int(mi[CSP + 2 * NWRAP + wg])
                    var sidepos = V3(0)
                    if sb >= 0:
                        sidepos = tpoint(mi, mf, qpos, res, f, sb, m3(mf, CSPF + 11 * wg + 8), frame)
                    var wr = wrap_geom(p0, p1, gpos, gq, mf[CSPF + 11 * wg + 7], type1 == 5, sb >= 0, sidepos)
                    wlen = wr[0]
                    wa = wr[1]
                    wb = wr[2]
                if wlen < 0:
                    var d = p1 - p0
                    L += sqrt((d * d).reduce_add()) / divisor
                    tseg(mi, qpos, res, f, jr, b0, p0, b1, p1, divisor, frame)
                else:
                    var d0 = wa - p0; var d1 = p1 - wb
                    L += (sqrt((d0 * d0).reduce_add()) + wlen + sqrt((d1 * d1).reduce_add())) / divisor
                    tseg(mi, qpos, res, f, jr, b0, p0, gb, wa, divisor, frame)
                    tseg(mi, qpos, res, f, jr, gb, wb, b1, p1, divisor, frame)
                j += 2 if geom else 1
        res[(OTLEN + t) * F + f] = L
        var vel: Float32 = 0
        for v in range(NV):
            vel += get(res, f, jr + v) * qvel[v * F + f]
        res[(OTVEL + t) * F + f] = vel


@always_inline
def tendon(mi: TI, mf: TF, qpos: TQ, qvel: TV, f: Int, t: Int) -> Tuple[Float32, Float32]:
    """MuJoCo mj_tendon for a fixed tendon (engine_core_smooth.c): length = sum coef qpos, velocity = sum coef qvel."""
    var length: Float32 = 0
    var vel: Float32 = 0
    var a = Int(mi[TNI + t])
    for w in range(a, a + Int(mi[TNI + NTEN + t])):
        var c = mf[TNF + w]
        length += c * qpos[Int(mi[TNI + 2 * NTEN + NWRAP + w]) * F + f]
        vel += c * qvel[Int(mi[TNI + 2 * NTEN + w]) * F + f]
    return (length, vel)


def tendon_passive(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_springdamper tendon part (engine_passive.c:775): spring about [lower, upper] of lengthspring, linear
    damper, along the constant tendon Jacobian (the wrap coefs) into qfrc_passive."""
    for t in range(NTEN):
        var lv = tendon(mi, mf, qpos, qvel, f, t)
        comptime if SPATIAL:  # spatial() has them, with the moment rows
            lv = (get(res, f, OTLEN + t), get(res, f, OTVEL + t))
        var lo = mf[TNF + NWRAP + 2 * NTEN + 2 * t]
        var hi = mf[TNF + NWRAP + 2 * NTEN + 2 * t + 1]
        var x = lv[0] - hi if lv[0] > hi else (lv[0] - lo if lv[0] < lo else Float32(0))
        var frc = -x * mf[TNF + NWRAP + t] - lv[1] * mf[TNF + NWRAP + NTEN + t]
        comptime if SPATIAL:  # qfrc += J' frc over the tendon's moment row
            for v in range(NV):
                res[(OPAS + v) * F + f] += get(res, f, OTJ + t * NV + v) * frc
            continue
        var a = Int(mi[TNI + t])
        for w in range(a, a + Int(mi[TNI + NTEN + t])):
            res[(OPAS + Int(mi[TNI + 2 * NTEN + w])) * F + f] += mf[TNF + w] * frc


def gravcomp(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_gravcomp (engine_passive.c:817) and its hand-off (:1012): per body with body_gravcomp the force
    -gravity mass gravcomp at xipos through mj_applyFT's point Jacobian (cdof_lin + cdof_ang x (xipos - subtree_com of the
    root), over the body's own and its ancestors' dofs) into qfrc_gravcomp (rows OGC); the dofs of joints without
    actgravcomp are added to qfrc_passive, the others wait for actuation() (engine_forward.c:770)."""
    for v in range(NV):
        res[(OGC + v) * F + f] = 0
    for i in range(1, NB):
        var gc = mf[CGF + i]
        if gc == 0:
            continue
        var force = m3(mf, GRAV) * (-(mf[MASS + i] * gc))
        var r = 3 * Int(mi[ROOT + i])
        var off = V3(get(res, f, OXI + 3 * (i - 1)), get(res, f, OXI + 3 * (i - 1) + 1), get(res, f, OXI + 3 * (i - 1) + 2), 0) - V3(get(res, f, OSC + r), get(res, f, OSC + r + 1), get(res, f, OSC + r + 2), 0)
        var b = i
        while b > 0:
            for d in range(Int(mi[BDOFADR + b]), Int(mi[BDOFADR + b]) + Int(mi[BDOFNUM + b])):
                var cd = OCD + 6 * d
                var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
                var lin = V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0)
                res[(OGC + d) * F + f] += ((lin + cross(ang, off)) * force).reduce_add()
            b = Int(mi[PARENT + b])
    for j in range(NJ):
        if mi[CGC + j] != 0:
            continue
        var t = Int(mi[JTYPE + j])
        var da = Int(mi[DOFADR + j])
        for d in range(da, da + (6 if t == 0 else (3 if t == 1 else 1))):
            res[(OPAS + d) * F + f] += get(res, f, OGC + d)


def velocity(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_comVel, mj_passive (joint springs, dof dampers), mj_rne(flg_acc = 0) into qfrc_bias."""
    # mj_comVel: cvel = cvel_parent + cdof qvel, cdof_dot = cvel x cdof with the velocity before the joint
    for i in range(1, NB):
        var p = Int(mi[PARENT + i])
        var cvel = get6(res, f, OCV + 6 * (p - 1)) if p > 0 else S6(0)
        for j in range(Int(mi[JNTADR + i]), Int(mi[JNTADR + i]) + Int(mi[JNTNUM + i])):
            var t = Int(mi[JTYPE + j])
            var da = Int(mi[DOFADR + j])
            if t == 0:  # free: translations have cdof_dot 0, then rotations as a ball
                for k in range(3):
                    put6(res, f, OCDD + 6 * (da + k), S6(0))
                    cvel += get6(res, f, OCD + 6 * (da + k)) * qvel[(da + k) * F + f]
                da += 3
            var n = 3 if t <= 1 else 1
            for k in range(n):
                put6(res, f, OCDD + 6 * (da + k), cross_motion(cvel, get6(res, f, OCD + 6 * (da + k))))
            for k in range(n):
                cvel += get6(res, f, OCD + 6 * (da + k)) * qvel[(da + k) * F + f]
        put6(res, f, OCV + 6 * (i - 1), cvel)
    # mj_passive: qfrc_passive = qfrc_spring + qfrc_damper
    for v in range(NV):
        res[(OPAS + v) * F + f] = -mf[DAMP + v] * qvel[v * F + f]
    for j in range(NJ):
        var k = mf[STIFF + j]
        if k == 0:
            continue
        var a = Int(mi[QADR + j])
        var da = Int(mi[DOFADR + j])
        var t = Int(mi[JTYPE + j])
        if t == 0:
            for c in range(3):
                res[(OPAS + da + c) * F + f] += -k * (qpos[(a + c) * F + f] - mf[QSPRING + a + c])
            a += 3
            da += 3
        if t <= 1:  # mju_subQuat(dif, unit(q), q_spring): neg(q_spring) q to axis-angle
            var q = unit(Q4(qpos[a * F + f], qpos[(a + 1) * F + f], qpos[(a + 2) * F + f], qpos[(a + 3) * F + f]))
            var qs = m4(mf, QSPRING + a)
            var d = mul(Q4(qs[0], -qs[1], -qs[2], -qs[3]), q)
            var axis = V3(d[1], d[2], d[3], 0)
            var s = sqrt((axis * axis).reduce_add())
            axis = axis / s if s >= 1e-15 else basis(0)
            var speed = 2 * atan2p(s, d[0])
            if speed > 3.14159265358979:
                speed -= 2 * 3.14159265358979
            for c in range(3):
                res[(OPAS + da + c) * F + f] += -k * axis[c] * speed
        else:  # slide, hinge
            res[(OPAS + da) * F + f] += -k * (qpos[a * F + f] - mf[QSPRING + a])
    comptime if SPATIAL:
        spatial(mi, mf, qpos, qvel, res, f)
    comptime if NTEN > 0:
        tendon_passive(mi, mf, qpos, qvel, res, f)
    comptime if NFLUID:
        fluid(mi, mf, res, f)
    comptime if GRAVCOMP:
        gravcomp(mi, mf, res, f)
    # mj_rne: cacc = cacc_parent + cdof_dot qvel (world: -gravity), cfrc = cinert cacc + cvel x* (cinert cvel)
    put6(res, f, OCACC, S6(0, 0, 0, -mf[GRAV], -mf[GRAV + 1], -mf[GRAV + 2], 0, 0))
    for i in range(1, NB):
        var acc = get6(res, f, OCACC + 6 * Int(mi[PARENT + i]))
        var bda = Int(mi[BDOFADR + i])
        for k in range(Int(mi[BDOFNUM + i])):
            acc += get6(res, f, OCDD + 6 * (bda + k)) * qvel[(bda + k) * F + f]
        put6(res, f, OCACC + 6 * i, acc)
        var cvel = get6(res, f, OCV + 6 * (i - 1))
        var ci = OCI + 10 * (i - 1)
        put6(res, f, OCFRC + 6 * i, mul_inert(res, f, ci, acc) + cross_force(cvel, mul_inert(res, f, ci, cvel)))
    for i in range(NB - 1, 0, -1):
        var p = Int(mi[PARENT + i])
        if p > 0:
            put6(res, f, OCFRC + 6 * p, get6(res, f, OCFRC + 6 * p) + get6(res, f, OCFRC + 6 * i))
    for v in range(NV):
        var prod = get6(res, f, OCD + 6 * v) * get6(res, f, OCFRC + 6 * Int(mi[DBODY + v]))
        res[(OBIAS + v) * F + f] = prod.reduce_add()


@always_inline
def jdot(mi: TI, res: TO, f: Int, jadr: Int, nch: Int, choff: Int, X: Int, ch: Int = CCHAIN) -> Float32:
    """Sparse Jacobian row (chain order) dotted with the dof vector in rows X.."""
    var acc: Float32 = 0
    for c in range(nch):
        acc += get(res, f, jadr + c) * get(res, f, X + Int(mi[ch + choff + c]))
    return acc


@always_inline
def sbase(mi: TI, res: TO, f: Int, p: Int) -> Int:
    comptime if NCV > 0:
        if Int(mi[CKIND + p]) > 0:  # pooled pair: this step's allocation
            return Int(get(res, f, OSD + 3 * p))
    return Int(mi[CPH + 11 * p + 8])


@always_inline
def nslot(mi: TI, res: TO, f: Int, p: Int) -> Int:
    comptime if NCV > 0:
        if Int(mi[CKIND + p]) > 0:
            return Int(get(res, f, OSD + 3 * p + 1))
    return Int(mi[CPH + 11 * p + 9])


@always_inline
def jbase(mi: TI, res: TO, f: Int, p: Int) -> Int:
    comptime if NCV > 0:
        if Int(mi[CKIND + p]) > 0:
            return Int(get(res, f, OSD + 3 * p + 2))
    return Int(mi[CPH + 11 * p + 10])


@always_inline
def powi(a: Float32, b: Float32) -> Float32:
    """MuJoCo power() for a whole exponent (export refuses others): amdgpu links no powf here."""
    var r: Float32 = 1
    for _ in range(Int(b)):
        r *= a
    return r


@always_inline
def impedance(s0: Float32, s1: Float32, s2: Float32, s3: Float32, s4: Float32, pos: Float32) -> Float32:
    """MuJoCo getimpedance (engine_core_constraint.c:2017), solimp already clamped, pos = violation minus margin."""
    if s0 == s1 or s2 <= 1e-15:
        return 0.5 * (s0 + s1)
    var x = abs(pos / s2)
    if x >= 1 or x <= 0:
        return s1 if x >= 1 else s0
    var yy: Float32
    if s4 == 1:
        yy = x
    elif x <= s3:
        yy = powi(x, s4) / powi(s3, s4 - 1)
    else:
        yy = 1 - powi(1 - x, s4) / powi(1 - s3, s4 - 1)
    return s0 + yy * (s1 - s0)


@always_inline
def terrain(mf: TF) -> PW:
    var p = PW()
    for k in range(NP):
        p[k] = mf[HFO + k]
    return p


@always_inline
def qsmooth(res: TO, f: Int, v: Int) -> Float32:
    """qfrc_passive - qfrc_bias, + qfrc_applied when NXF: MuJoCo qfrc_smooth before actuation and constraints."""
    var q = get(res, f, OPAS + v) - get(res, f, OBIAS + v)
    comptime if NXF > 0:
        q += get(res, f, OQAPP + v)
    return q


def xfrc(mi: TI, ctrl: TU, res: TO, f: Int):
    """MuJoCo mj_xfrcAccumulate: each slot's force and torque at its body's xipos through mj_applyFT into qfrc_applied,
    qfrc[d] = cdof_lin . force + cdof_ang . (torque + (xipos - subtree_com[root]) x force) over the body's dof chain (frames
    relative to the tree's floating origin, as both rows are before absolute()). Body id 0: an empty slot."""
    for v in range(NV):
        res[(OQAPP + v) * F + f] = 0
    for k in range(NXF):
        var row = NU + 7 * NMOCAP + 7 * k
        var b = Int(ctrl[row * F + f])
        if b <= 0:
            continue
        var fo = V3(ctrl[(row + 1) * F + f], ctrl[(row + 2) * F + f], ctrl[(row + 3) * F + f], 0)
        var r = 3 * Int(mi[ROOT + b])
        var off = V3(get(res, f, OXI + 3 * (b - 1)), get(res, f, OXI + 3 * (b - 1) + 1), get(res, f, OXI + 3 * (b - 1) + 2), 0) - V3(get(res, f, OSC + r), get(res, f, OSC + r + 1), get(res, f, OSC + r + 2), 0)
        var tq = V3(ctrl[(row + 4) * F + f], ctrl[(row + 5) * F + f], ctrl[(row + 6) * F + f], 0) + cross(off, fo)
        var bb = b
        while bb > 0:
            for d in range(Int(mi[BDOFADR + bb]), Int(mi[BDOFADR + bb]) + Int(mi[BDOFNUM + bb])):
                var cd = OCD + 6 * d
                res[(OQAPP + d) * F + f] += (V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0) * tq).reduce_add() + (V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0) * fo).reduce_add()
            bb = Int(mi[PARENT + bb])


@always_inline
def side_com(mi: TI, qpos: TQ, res: TO, f: Int, g1: Int, b: Int, com: V3) -> V3:
    """subtree_com of geom1's root in the frame of b's tree (mj_jacDifPair: each side about its own root); com itself
    when geom1 is a world geom or shares b's root."""
    var gb1 = Int(mi[GBODY + g1])
    if gb1 == 0 or Int(mi[ROOT + gb1]) == Int(mi[ROOT + b]):
        return com
    var r = 3 * Int(mi[ROOT + gb1])
    return V3(get(res, f, OSC + r), get(res, f, OSC + r + 1), get(res, f, OSC + r + 2), 0) + (origin(mi, qpos, f, gb1) - origin(mi, qpos, f, b))


@always_inline
def emit_contact(mi: TI, mf: TF, qvel: TV, res: TO, f: Int, p: Int, count: Int, cpos: V3, dist: Float32, normal: V3, y: V3, z: V3, com: V3, com1: V3):
    """The rest of mj_instantiateContact for contact `count` of pair p at cpos with the frame (normal, y, z): the pyramidal or
    elliptic Jacobian rows over the pair's dof chain, getsolparam / getimpedance / makeImpedance, referenceConstraint.
    com is geom2's side and com1 geom1's side, both in geom2's tree frame."""
    var h = mf[TSTEP]
    var impratio = mf[COPT]
    var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
    var pf = CPF + 14 * p
    var margin = mf[pf]; var gap = mf[pf + 1]; var f0 = mf[pf + 2]; var f1 = mf[pf + 3]
    var slot = sbase(mi, res, f, p) + count
    var imargin = margin  # MuJoCo 3.9 mj_collideGeoms: the narrowphase sees margin + gap, the row (includemargin) is the margin itself
    if dist >= imargin:  # mj_setContact (engine_collision_driver.c:1780): a contact in the gap band carries no row
        res[(OVALID + slot) * F + f] = 0
        return
    res[(OVALID + slot) * F + f] = Float32(p + 1) if ELL else Float32(1)  # elliptic: the pair index + 1, for the cone's mu and friction
    comptime if DIMX > 3:
        emit_rows(mi, mf, qvel, res, f, p, count, cpos, dist - imargin, normal, y, z, com, com1)
        return
    var frictionless = False  # condim 1 (mjCNSTR_CONTACT_FRICTIONLESS): one normal row, the pyramid's other three rows empty
    comptime if NCV > 0:
        frictionless = mi[CCND + p] == 1
    # mj_jacDifPair over the chain: jacp = cdof_lin + cdof_ang x (pos - subtree_com[root]), signed (geom2's body
    # minus geom1's; a dof of both cancels exactly); rows to the contact frame
    var off2 = cpos - com
    var off1 = cpos - com1
    for c in range(nch):
        var cd = OCD + 6 * Int(mi[CCHAIN + choff + c])
        var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
        var s = Int(mi[CSIGN + choff + c])
        var jp = (V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0) + cross(ang, off1 if s < 0 else off2)) * Float32(s)
        var jn = (normal * jp).reduce_add(); var j1 = (y * jp).reduce_add(); var j2 = (z * jp).reduce_add()
        var jadr = OJ + jbase(mi, res, f, p) + RS * count * nch + c
        comptime if ELL:  # mj_instantiateContact elliptic: the contact-frame rows themselves
            res[jadr * F + f] = jn
            res[(jadr + nch) * F + f] = j1
            res[(jadr + 2 * nch) * F + f] = j2
        else:
            if frictionless:
                res[jadr * F + f] = jn
                res[(jadr + nch) * F + f] = 0
                res[(jadr + 2 * nch) * F + f] = 0
                res[(jadr + 3 * nch) * F + f] = 0
            else:
                res[jadr * F + f] = jn + f0 * j1
                res[(jadr + nch) * F + f] = jn - f0 * j1
                res[(jadr + 2 * nch) * F + f] = jn + f1 * j2
                res[(jadr + 3 * nch) * F + f] = jn - f1 * j2
    # getsolparam, getimpedance, makeImpedance, referenceConstraint
    var ref0 = mf[pf + 4]; var ref1 = mf[pf + 5]
    if ref0 > 0:  # refsafe on the standard format only; non-positive = direct (stiffness, damping)
        ref0 = max(ref0, 2 * h)
    var s0 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 6])); var s1 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 7]))
    var s2 = max(Float32(0), mf[pf + 8]); var s3 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 9])); var s4 = max(Float32(1), mf[pf + 10])
    var imp = impedance(s0, s1, s2, s3, s4, dist - imargin)
    var tran = mf[pf + 11]
    var K = 1 / max(Float32(1e-15), s1 * s1 * ref0 * ref0 * ref1 * ref1) if ref0 > 0 else -ref0 / max(Float32(1e-15), s1 * s1)  # direct: -solref[0] / d_width^2
    var B = 2 / max(Float32(1e-15), s1 * ref0) if ref1 > 0 else -ref1 / max(Float32(1e-15), s1)  # direct: -solref[1] / d_width
    comptime if ELL:  # mj_diagApprox: tran on all rows; R1 = R0 / impratio, R2 R1 f0^2 / f1^2; friction rows have K = 0
        var R0 = max(Float32(1e-15), (1 - imp) * tran / imp)
        var R1 = R0 / max(Float32(1e-15), impratio)
        var R2 = R1 * f0 * f0 / (f1 * f1)
        res[(OD + RS * slot) * F + f] = 1 / R0
        res[(OD + RS * slot + 1) * F + f] = 1 / R1
        res[(OD + RS * slot + 2) * F + f] = 1 / R2
    else:
        if frictionless:  # mj_diagApprox: tran alone; R = (1 - imp) tran / imp
            res[(OD + RS * slot) * F + f] = 1 / max(Float32(1e-15), (1 - imp) * tran / imp)
            for k in range(1, RS):
                res[(OD + RS * slot + k) * F + f] = 0
        else:
            var R0 = max(Float32(1e-15), (1 - imp) * (tran + f0 * f0 * tran) / imp)
            var R1 = R0 / max(Float32(1e-15), impratio)
            var mu = f0 * sqrt(R1 / R0)
            var D = 1 / (2 * mu * mu * R0)
            for k in range(RS):
                res[(OD + RS * slot + k) * F + f] = D
    for k in range(NR):
        var row = RS * slot + k
        var vel: Float32 = 0
        for c in range(nch):
            vel += get(res, f, OJ + jbase(mi, res, f, p) + (RS * count + k) * nch + c) * qvel[Int(mi[CCHAIN + choff + c]) * F + f]
        comptime if ELL:
            res[(OAREF + row) * F + f] = -B * vel - (K * imp * (dist - imargin) if k == 0 else Float32(0))
        else:
            res[(OAREF + row) * F + f] = 0 if frictionless and k > 0 else -B * vel - K * imp * (dist - imargin)



comptime U8 = SIMD[DType.float32, 8]  # one contact's rows in the contact frame (condim up to 6)


@always_inline
def pdim(mi: TI, p: Int) -> Int:
    """Pair p's condim: the DIMX block, else 1 for a frictionless convex pair, else 3."""
    comptime if DIMX > 3:
        return Int(mi[CDIM + p])
    comptime if NCV > 0:
        return 1 if mi[CCND + p] == 1 else 3
    return 3


@always_inline
def pscale(mf: TF, p: Int) -> U8:
    """Pair p's dual-cone scaling diag(mu, friction 0..4) (mj_constraintUpdate_impl U = jar * this): mu = friction 0 / sqrt(impratio)."""
    var f0 = mf[CPF + 14 * p + 2]
    var sc = U8(0)
    sc[0] = f0 / sqrt(max(Float32(1e-15), mf[COPT]))
    sc[1] = f0
    sc[2] = mf[CPF + 14 * p + 3]
    comptime if DIMX > 3:
        for k in range(3):
            sc[3 + k] = mf[CFR + 3 * p + k]
    return sc


def emit_rows(mi: TI, mf: TF, qvel: TV, res: TO, f: Int, p: Int, count: Int, cpos: V3, pos: Float32, normal: V3, y: V3, z: V3, com: V3, com1: V3):
    """emit_contact for any condim (DIMX > 3; mj_instantiateContact, engine_core_constraint.c:1428): contact-frame rows
    translation (normal, tangent 1, 2), then rotation (torsion about the normal, rolling about the tangents) up to the
    pair's condim; pyramidal: edge pairs J0 +- friction[k-1] Jk, elliptic: the rows themselves. mj_diagApprox and
    mj_makeImpedance: pyramidal R = 2 mu^2 R0 on every edge, elliptic R0, R1 = R0 / impratio, R(j+1) = R1 f0^2 / f(j)^2.
    Rows past the pair's own count are emptied (J, D, aref 0), as a pool slot may have held a wider pair."""
    var h = mf[TSTEP]
    var impratio = mf[COPT]
    var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
    var pf = CPF + 14 * p
    var sc = pscale(mf, p)
    var f0 = sc[1]
    var slot = sbase(mi, res, f, p) + count
    var dim = pdim(mi, p)
    var nrow = 1 if dim == 1 else (dim if ELL else 2 * (dim - 1))
    var off2 = cpos - com
    var off1 = cpos - com1
    for c in range(nch):
        var cd = OCD + 6 * Int(mi[CCHAIN + choff + c])
        var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
        var sg = Float32(Int(mi[CSIGN + choff + c]))
        var jp = (V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0) + cross(ang, off1 if sg < 0 else off2)) * sg
        var jr = ang * sg
        var jc = U8(0)
        jc[0] = (normal * jp).reduce_add(); jc[1] = (y * jp).reduce_add(); jc[2] = (z * jp).reduce_add()
        jc[3] = (normal * jr).reduce_add(); jc[4] = (y * jr).reduce_add(); jc[5] = (z * jr).reduce_add()
        var jadr = OJ + jbase(mi, res, f, p) + RS * count * nch + c
        for k in range(RS):
            var v: Float32 = 0
            if k < nrow:
                comptime if ELL:
                    v = jc[k]
                else:
                    if dim == 1:
                        v = jc[0]
                    else:
                        var kk = k // 2 + 1
                        v = jc[0] + (sc[kk] if k % 2 == 0 else -sc[kk]) * jc[kk]
            res[(jadr + k * nch) * F + f] = v
    var ref0 = mf[pf + 4]; var ref1 = mf[pf + 5]
    if ref0 > 0:
        ref0 = max(ref0, 2 * h)
    var s0 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 6])); var s1 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 7]))
    var s2 = max(Float32(0), mf[pf + 8]); var s3 = min(Float32(0.9999), max(Float32(1e-4), mf[pf + 9])); var s4 = max(Float32(1), mf[pf + 10])
    var imp = impedance(s0, s1, s2, s3, s4, pos)
    var tran = mf[pf + 11]
    var K = 1 / max(Float32(1e-15), s1 * s1 * ref0 * ref0 * ref1 * ref1) if ref0 > 0 else -ref0 / max(Float32(1e-15), s1 * s1)
    var B = 2 / max(Float32(1e-15), s1 * ref0) if ref1 > 0 else -ref1 / max(Float32(1e-15), s1)
    var D = U8(0)  # elliptic: per row (condim <= 6); pyramidal: Dp on every edge (up to 10 rows)
    var Dp: Float32 = 0
    comptime if ELL:
        var R0 = max(Float32(1e-15), (1 - imp) * tran / imp)
        var R1 = R0 / max(Float32(1e-15), impratio)
        D[0] = 1 / R0
        D[1] = 1 / R1
        for j in range(1, dim - 1):
            D[j + 1] = 1 / (R1 * f0 * f0 / (sc[j + 1] * sc[j + 1]))
    else:
        if dim == 1:
            D[0] = 1 / max(Float32(1e-15), (1 - imp) * tran / imp)
        else:
            var R0 = max(Float32(1e-15), (1 - imp) * (tran + f0 * f0 * tran) / imp)
            var R1 = R0 / max(Float32(1e-15), impratio)
            var mu = f0 * sqrt(R1 / R0)
            Dp = 1 / (2 * mu * mu * R0)
    for k in range(RS):
        var row = RS * slot + k
        var dk = Dp
        comptime if ELL:
            dk = D[k]
        else:
            if dim == 1:
                dk = D[0]
        res[(OD + row) * F + f] = dk if k < nrow else Float32(0)
        if k >= nrow:
            res[(OAREF + row) * F + f] = 0
            continue
        var vel: Float32 = 0
        for c in range(nch):
            vel += get(res, f, OJ + jbase(mi, res, f, p) + (RS * count + k) * nch + c) * qvel[Int(mi[CCHAIN + choff + c]) * F + f]
        comptime if ELL:
            res[(OAREF + row) * F + f] = -B * vel - (K * imp * pos if k == 0 else Float32(0))
        else:
            res[(OAREF + row) * F + f] = -B * vel - K * imp * pos


def rows3(q: Q4) -> Tuple[V3, V3, V3]:
    """Rotation matrix rows of the quaternion (convex.mojo's Shape takes rows)."""
    var c0 = col(q, 0)
    var c1 = col(q, 1)
    var c2 = col(q, 2)
    return (V3(c0[0], c1[0], c2[0], 0), V3(c0[1], c1[1], c2[1], 0), V3(c0[2], c1[2], c2[2], 0))


def pool_alloc(mi: TI, res: TO, f: Int, p: Int, want: Int) -> Int:
    """Take up to `want` contact slots (and their Jacobian entries) of the animal's pool for pooled pair p this step; returns
    how many it got (fewer only when the pool is exhausted: those contacts are dropped, the gate's ncon shows it)."""
    var nch = Int(mi[CPH + 11 * p + 6])
    var used = Int(get(res, f, OPP))
    var jused = Int(get(res, f, OPP + 1))
    var n = want
    while n > 0 and (used + n > NPOOL or jused + RS * n * nch > JPOOL):
        n -= 1
    res[(OSD + 3 * p) * F + f] = Float32(NSTATIC + used)
    res[(OSD + 3 * p + 1) * F + f] = Float32(n)
    res[(OSD + 3 * p + 2) * F + f] = Float32(JSTATIC + jused)
    res[OPP * F + f] = Float32(used + n)
    res[(OPP + 1) * F + f] = Float32(jused + RS * n * nch)
    return n


struct TerrainH(Height):
    """The world's terrain function sampled at the virtual grid's nodes (x = dx c - size0, y = dx r - size1; the hfield
    geom sits at world (0, 0, lo)): the node heights MuJoCo's hfield holds, from world.mojo's function instead of a table."""
    var params: PW
    var lo: Float32
    var dx: Float32
    var size0: Float32
    var size1: Float32

    def __init__(out self, params: PW, lo: Float32, dx: Float32, size0: Float32, size1: Float32):
        self.params = params
        self.lo = lo
        self.dx = dx
        self.size0 = size0
        self.size1 = size1

    def h(self, r: Int, c: Int) -> CS:
        return (height_grad(self.dx * Float32(c) - self.size0, self.dx * Float32(r) - self.size1, self.params)[0] - self.lo).cast[CT]()


struct TerrainTab[o: MutOrigin](Height):
    """HTAB: the grid's node heights read from mf (MuJoCo's hfield_data x size[2]); an app may rewrite them and the
    table centre between launches to stream terrain around an animal (living-world's walk)."""
    var t: Pointer[Float32, Self.o]  # the first node height

    def __init__(out self, t: Pointer[Float32, Self.o]):
        self.t = t

    def h(self, r: Int, c: Int) -> CS:
        return self.t[r * HFNC + c].cast[CT]()


def hfield_pair(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int, p: Int, q2: Q4, pos2: V3, b: Int):
    """Port of mjc_ConvexHField (engine_collision_convex.c:1120), in convex.mojo: one GJK + EPA per cell-triangle prism under
    the mesh, on world.mojo's terrain function at the grid nodes (the grid is MuJoCo's, extended without limit). The mesh
    pose goes to the hfield frame in float64 (animal frame + floating origin, minus the hfield's origin), the contacts
    come back to the animal frame. At most nslot contacts, in prism order; MuJoCo keeps up to 50 (check.sh terrain_exact reports it)."""
    var g2 = Int(mi[CPH + 11 * p + 1])
    var voff = Int(mi[CPH + 11 * p + 4]); var eoff = Int(mi[CPH + 11 * p + 5])
    var pf = CPF + 14 * p
    var margin = mf[pf] + mf[pf + 1]
    var rb2 = mf[pf + 12]
    var o = origin(mi, qpos, f, b)
    var size0 = mf[CCV + 1]; var size1 = mf[CCV + 2]; var size2 = mf[CCV + 3]; var size3 = mf[CCV + 4]; var lo = mf[CCV + 5]
    var dx = 2 * size0 / Float32(HFNC - 1)
    var mesh = Sp(mf.ptr.unsafe_offset(CVERT + 3 * voff), mi.ptr.unsafe_offset(CFIRST + eoff), mi.ptr.unsafe_offset(CEDGE))
    mesh.kind = MESH
    var org = (o.cast[CT]() + CVEC(0, 0, -lo.cast[CT]()[0], 0))  # animal frame -> hfield frame
    mesh.pos = pos2.cast[CT]() + org
    var c = rows3(q2)
    mesh.r0 = c[0].cast[CT](); mesh.r1 = c[1].cast[CT](); mesh.r2 = c[2].cast[CT]()
    var cs: HCons
    comptime if HTAB:
        comptime assert CHT == NFL, "HTAB: model.py puts the table last in mf (CHT == NFL)"
        mesh.pos = mesh.pos - CVEC(mf[CHT].cast[CT]()[0], mf[CHT + 1].cast[CT]()[0], 0, 0)  # world -> the table's centre
        cs = collide_hfield(TerrainTab[MutAnyOrigin](mf.ptr.unsafe_offset(CHT + 2)), mesh, HFNR, HFNC, size0.cast[CT]()[0], size1.cast[CT]()[0], size2.cast[CT]()[0],
                            size3.cast[CT]()[0], margin.cast[CT]()[0], mf[CCV].cast[CT]()[0], rb2.cast[CT]()[0], False, CVEC(0))
    else:
        cs = collide_hfield(TerrainH(terrain(mf), lo, dx, size0, size1), mesh, HFNR, HFNC, size0.cast[CT]()[0], size1.cast[CT]()[0], size2.cast[CT]()[0],
                            size3.cast[CT]()[0], margin.cast[CT]()[0], mf[CCV].cast[CT]()[0], rb2.cast[CT]()[0], True, CVEC(0))
    var com = V3(get(res, f, OSC + 3 * Int(mi[ROOT + b])), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 1), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 2), 0)
    var count = 0
    var take = pool_alloc(mi, res, f, p, min(cs.n, Int(mi[CPH + 11 * p + 9])))
    while count < take:
        var normal = cs.normal[count].cast[DType.float32]()
        var y = basis(1) if normal[1] < 0.5 and normal[1] > -0.5 else basis(2)  # mju_makeFrame
        y = y - normal * (normal * y).reduce_add()
        y = y / sqrt((y * y).reduce_add())
        var z = cross(normal, y)
        emit_contact(mi, mf, qvel, res, f, p, count, (cs.pos[count] - org).cast[DType.float32](), cs.dist[count].cast[DType.float32](), normal, y, z, com, com)
        count += 1


@always_inline
def polys(mi: TI, mf: TF, mut s: Sp, voff: Int):
    """A mesh shape's polygon pointers (PCLIP), its hull starting at hull vertex voff."""
    s.pmv = mi.ptr.unsafe_offset(CPMV + 2 * voff)
    s.pmap = mi.ptr.unsafe_offset(CPMV + 2 * NHV)
    s.ppoly = mi.ptr.unsafe_offset(CPMV + 2 * NHV + NPMAP)
    s.pvert = mi.ptr.unsafe_offset(CPMV + 2 * NHV + NPMAP + 2 * NPOLY)
    s.pnorm = mf.ptr.unsafe_offset(CPN)
    s.npoly = NPOLY


def convex_pair(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int, p: Int, kind: Int, q1: Q4, pos1_: V3, q2: Q4, pos2: V3, b: Int):
    """Port of mjc_Convex for a box-mesh (kind 2, up to 5 contacts) or ellipsoid-mesh (kind 1, one) pair, ported in convex.mojo (GJK +
    EPA, perturbation search); the contacts take the shared mj_instantiateContact tail with their own frame. A world geom1
    is moved into the animal's frame. The bounding spheres skip pairs that cannot be within the margin (MuJoCo's pair
    loop applies the same bound); the box is tested as its exact distance to the mesh's bounding sphere."""
    var g1 = Int(mi[CPH + 11 * p]); var g2 = Int(mi[CPH + 11 * p + 1])
    var nvert = Int(mi[CPH + 11 * p + 3]); var voff = Int(mi[CPH + 11 * p + 4]); var eoff = Int(mi[CPH + 11 * p + 5])
    var pf = CPF + 14 * p
    var margin = mf[pf] + mf[pf + 1]
    var rb2 = mf[pf + 12]; var rb1 = mf[pf + 13]
    var pos1 = pos1_
    if Int(mi[GBODY + g1]) == 0:
        pos1 = pos1 - origin(mi, qpos, f, b)
    elif Int(mi[ROOT + Int(mi[GBODY + g1])]) != Int(mi[ROOT + b]):  # geom1 on another tree: its frame to b's
        pos1 = pos1 + (origin(mi, qpos, f, Int(mi[GBODY + g1])) - origin(mi, qpos, f, b))
    var far = False
    var d12 = pos2 - pos1
    if kind == 1 or kind >= 4:
        far = sqrt((d12 * d12).reduce_add()) > rb1 + rb2 + margin
    else:  # distance from the mesh centre to the box (in the box frame), against the mesh's bounding radius
        var r1 = rows3(q1)
        var sz = m3(mf, CCV + 6 + 3 * p)
        var l = V3((r1[0] * d12).reduce_add(), (r1[1] * d12).reduce_add(), (r1[2] * d12).reduce_add(), 0)
        var e = V3(max(abs(l[0]) - sz[0], 0), max(abs(l[1]) - sz[1], 0), max(abs(l[2]) - sz[2], 0), 0)
        far = sqrt((e * e).reduce_add()) > rb2 + margin
    var count = 0
    res[(OSD + 3 * p + 1) * F + f] = 0  # no slots unless the pair takes some below
    if not far:
        var cs = ConSet()
        if kind >= 5:  # analytic pairs (engine_collision_primitive.c): mjraw_CapsuleCapsule, mjraw_SphereCapsule, mjc_SphereCylinder
            var a = rows3(q1); var c = rows3(q2)
            cs = primitive_pair(kind, pos1.cast[CT](), V3(a[0][2], a[1][2], a[2][2], 0).cast[CT](), m3(mf, CCV + 6 + 3 * p).cast[CT](),
                                pos2.cast[CT](), V3(c[0][2], c[1][2], c[2][2], 0).cast[CT](), m3(mf, CCV + 6 + 3 * NCP + 3 * p).cast[CT](), margin.cast[CT]())
        else:
            var s1 = Sp(mf.ptr.unsafe_offset(CVERT + 3 * voff), mi.ptr.unsafe_offset(CFIRST + eoff), mi.ptr.unsafe_offset(CEDGE))
            var s2 = Sp(mf.ptr.unsafe_offset(CVERT + 3 * voff), mi.ptr.unsafe_offset(CFIRST + eoff), mi.ptr.unsafe_offset(CEDGE))
            if kind == 4:  # two primitives (sphere, capsule, ellipsoid, cylinder): mjc_Convex, the perturbation search on capsule/cylinder pairs (up to 5)
                s1.kind = Int(mi[CGT + 2 * p]); s2.kind = Int(mi[CGT + 2 * p + 1])
                s2.size = m3(mf, CCV + 6 + 3 * NCP + 3 * p).cast[CT]()
                comptime if NMM:
                    if s1.kind == MESH:  # mesh-mesh (mjc_Convex): geom1's own hull, s1 was built on geom2's
                        s1 = Sp(mf.ptr.unsafe_offset(CVERT + 3 * Int(mi[CMM + 2 * p])), mi.ptr.unsafe_offset(CFIRST + Int(mi[CMM + 2 * p + 1])), mi.ptr.unsafe_offset(CEDGE))
                        s1.kind = MESH
            else:
                s1.kind = BOX if kind == 2 else ELLIPSOID
                s2.kind = MESH
            s1.size = m3(mf, CCV + 6 + 3 * p).cast[CT]()
            s1.pos = pos1.cast[CT]()
            var a = rows3(q1)
            s1.r0 = a[0].cast[CT](); s1.r1 = a[1].cast[CT](); s1.r2 = a[2].cast[CT]()
            s2.pos = pos2.cast[CT]()
            var c = rows3(q2)
            s2.r0 = c[0].cast[CT](); s2.r1 = c[1].cast[CT](); s2.r2 = c[2].cast[CT]()
            comptime if PCLIP:  # the meshes' polygons for multicontact (margin 0)
                if s2.kind == MESH:
                    polys(mi, mf, s2, voff)
                comptime if NMM:
                    if s1.kind == MESH:
                        polys(mi, mf, s1, Int(mi[CMM + 2 * p]))
            cs = collide(s1, s2, margin.cast[CT](), mf[CCV].cast[CT](), min(rb1, rb2).cast[CT]())
        var com = V3(get(res, f, OSC + 3 * Int(mi[ROOT + b])), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 1), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 2), 0)
        var com1 = side_com(mi, qpos, res, f, g1, b, com)
        var take = pool_alloc(mi, res, f, p, min(cs.n, Int(mi[CPH + 11 * p + 9])))
        while count < take:
            var normal = cs.normal[count].cast[DType.float32]()
            var y = basis(1) if normal[1] < 0.5 and normal[1] > -0.5 else basis(2)  # mju_makeFrame
            y = y - normal * (normal * y).reduce_add()
            y = y / sqrt((y * y).reduce_add())
            var z = cross(normal, y)
            emit_contact(mi, mf, qvel, res, f, p, count, cs.pos[count].cast[DType.float32](), cs.dist[count].cast[DType.float32](), normal, y, z, com, com1)
            count += 1


def contacts(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mjc_PlaneConvex (support vertex + hull neighbours below the margin, at most 3, none within 0.3 rbound of
    the first), mj_instantiateContact (pyramidal condim 3, Jacobian over the body's dof chain), mj_diagApprox,
    mj_makeImpedance and mj_referenceConstraint, per exported plane-mesh pair; mjc_SphereSphere (one contact on the centre line, a two-body Jacobian) per sphere pair. A heightfield ground (HFGEOM) is not
    mjc_ConvexHField (GJK+EPA on cell prisms): the plane is the tangent plane of world.mojo's terrain function under
    the mesh centre, exact on linear terrain, wrong where the terrain bends inside one mesh."""
    comptime if NCV > 0:  # the pool of the convex and exact-heightfield pairs starts empty each step
        res[OPP * F + f] = 0
        res[(OPP + 1) * F + f] = 0
        for sp in range(NPOOL):
            res[(OVALID + NSTATIC + sp) * F + f] = 0
    for p in range(NCP):
        var g1 = Int(mi[CPH + 11 * p]); var g2 = Int(mi[CPH + 11 * p + 1]); var b = Int(mi[CPH + 11 * p + 2])
        var nvert = Int(mi[CPH + 11 * p + 3]); var voff = Int(mi[CPH + 11 * p + 4]); var eoff = Int(mi[CPH + 11 * p + 5])
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        var pf = CPF + 14 * p
        var margin = mf[pf] + mf[pf + 1]; var f0 = mf[pf + 2]; var f1 = mf[pf + 3]  # the narrowphase margin is margin + gap (mj_collideGeoms); emit_contact keeps the row within the margin
        var b1 = body(res, f, Int(mi[GBODY + g1]))
        var q1 = mul(b1[1], m4(mf, GQUAT + 4 * g1))
        var pos1 = V3(get(res, f, OG + 3 * g1), get(res, f, OG + 3 * g1 + 1), get(res, f, OG + 3 * g1 + 2), 0)
        var b2 = body(res, f, b)
        var q2 = mul(b2[1], m4(mf, GQUAT + 4 * g2))
        var pos2 = V3(get(res, f, OG + 3 * g2), get(res, f, OG + 3 * g2 + 1), get(res, f, OG + 3 * g2 + 2), 0)
        comptime if NCV > 0:
            var kind = Int(mi[CKIND + p])
            if kind == 3:
                hfield_pair(mi, mf, qpos, qvel, res, f, p, q2, pos2, b)
                continue
            if kind > 0:
                convex_pair(mi, mf, qpos, qvel, res, f, p, kind, q1, pos1, q2, pos2, b)
                continue
        var normal: V3
        var pnt: V3
        var dist: Float32
        var locdir = V3(0)
        var best = 0
        if nvert == 0:  # mjraw_SphereSphere: one contact on the centre line, normal from geom1 to geom2
            var r1 = mf[pf + 13]; var r2 = mf[pf + 12]
            if Int(mi[GBODY + g1]) == 0:  # a world geom (static sphere) is in world coordinates, the animal's frame is shifted by its floating origin
                pos1 = pos1 - origin(mi, qpos, f, b)
            elif Int(mi[ROOT + Int(mi[GBODY + g1])]) != Int(mi[ROOT + b]):  # geom1 on another tree: its frame to b's
                pos1 = pos1 + (origin(mi, qpos, f, Int(mi[GBODY + g1])) - origin(mi, qpos, f, b))
            var dif = pos2 - pos1
            var cd2 = (dif * dif).reduce_add()
            if cd2 > (margin + r1 + r2) * (margin + r1 + r2):
                for c in range(nslot(mi, res, f, p)):
                    res[(OVALID + sbase(mi, res, f, p) + c) * F + f] = 0
                continue
            var len = sqrt(cd2)
            dist = len - r1 - r2
            if len < 1e-15:  # same centre: the cross product of the z axes
                normal = cross(col(q1, 2), col(q2, 2))
                var nl = sqrt((normal * normal).reduce_add())
                normal = normal / nl if nl > 1e-15 else V3(1, 0, 0, 0)  # parallel z axes: mju_normalize3's fallback
            else:
                normal = dif / len
            pnt = pos1 + normal * (r1 + dist)  # the loop's contact point pnt - dist/2 normal is MuJoCo's pos
        else:
            var o = origin(mi, qpos, f, b)  # the plane (a world geom, world coordinates) in the animal's frame
            pos1 = pos1 - o
            normal = rot(q1, basis(2))
            comptime if HFG >= 0:
                if g1 == HFG:
                    var w = pos2 + o
                    var hg = height_grad(w[0], w[1], terrain(mf))
                    normal = V3(-hg[1], -hg[2], 1, 0) / sqrt(hg[1] * hg[1] + hg[2] * hg[2] + 1)
                    pos1 = V3(pos2[0], pos2[1], hg[0] - o[2], 0)  # pos2's own x, y: no round trip through world coordinates
            if (normal * (pos2 - pos1)).reduce_add() - mf[pf + 12] > margin:  # bounding sphere clear of the plane: the
                for c in range(nslot(mi, res, f, p)):                                        # support vertex would be too (no contact)
                    res[(OVALID + sbase(mi, res, f, p) + c) * F + f] = 0
                continue
            locdir = rot(Q4(q2[0], -q2[1], -q2[2], -q2[3]), -normal)
            var bdot: Float32 = -3.4e38
            for v in range(nvert):
                var vd = (locdir * m3(mf, CVERT + 3 * (voff + v))).reduce_add()
                if vd > bdot:
                    bdot = vd
                    best = v
            pnt = rot(q2, m3(mf, CVERT + 3 * (voff + best))) + pos2
            dist = (normal * (pnt - pos1)).reduce_add()
        # contact frame (mju_makeFrame from the normal), shared by the pair's contacts
        var y = basis(1) if normal[1] < 0.5 and normal[1] > -0.5 else basis(2)
        y = y - normal * (normal * y).reduce_add()
        y = y / sqrt((y * y).reduce_add())
        var z = cross(normal, y)
        var com = V3(get(res, f, OSC + 3 * Int(mi[ROOT + b])), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 1), get(res, f, OSC + 3 * Int(mi[ROOT + b]) + 2), 0)
        var com1 = side_com(mi, qpos, res, f, g1, b, com)
        var first = pnt - normal * (0.5 * dist)
        var threshold = (normal * (pos2 - pos1)).reduce_add() - margin
        var count = 0
        var e = Int(mi[CFIRST + eoff + best])
        while count < nslot(mi, res, f, p):
            if count > 0:  # hull neighbours of the support vertex
                var loc = Int(mi[CEDGE + e])
                if loc < 0:
                    break
                e += 1
                if (locdir * m3(mf, CVERT + 3 * (voff + loc))).reduce_add() <= threshold:
                    continue
                pnt = rot(q2, m3(mf, CVERT + 3 * (voff + loc))) + pos2
                var dd = pnt - first
                if sqrt((dd * dd).reduce_add()) < 0.3 * mf[pf + 12]:
                    continue
                dist = (normal * (pnt - pos1)).reduce_add()
            elif dist > margin:
                break
            var cpos = pnt - normal * (0.5 * dist)
            emit_contact(mi, mf, qvel, res, f, p, count, cpos, dist, normal, y, z, com, com1)
            count += 1
        for c in range(count, nslot(mi, res, f, p)):
            res[(OVALID + sbase(mi, res, f, p) + c) * F + f] = 0


def actuation(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO, f: Int):
    """MuJoCo mj_transmission + mj_fwdActuation for the ported kinds. Joint (hinge, slide): moment = gear on the
    joint's dof, length = qpos gear. Body (adhesion): moment = -(mean contact normal Jacobian) over the body's
    contacts (pyramid rows weighted 0.5 / (dim - 1), which sums to the normal row), length 0. Force = gain ctrl
    (ctrl clamped to ctrlrange) + bias0 + bias1 length + bias2 velocity, clamped to forcerange; qfrc = moment' force, plus
    actuator gravcomp, then clamped per joint to jnt_actfrcrange (engine_forward.c:788)."""
    for v in range(NV):
        res[(OQACT + v) * F + f] = 0
    comptime if GAME:  # a dead animal has no servo or adhesion effort (game.actuator_scale)
        if get(res, f, OGAME + 3) != 0:
            for u in range(NU):
                res[(OAF + u) * F + f] = 0
            return
    for u in range(NU):
        var a = CACT + 5 * u
        var af = CACF + 9 * u
        var c = ctrl[u * F + f]
        if mi[a + 2] != 0:
            c = min(max(c, mf[af + 4]), mf[af + 5])
        var gear = mf[af + 8]
        if mi[a] == 0:  # joint
            var dof = Int(mi[a + 1])
            var length = qpos[Int(mi[a + 4]) * F + f] * gear
            var vel = gear * qvel[dof * F + f]
            var force = mf[af] * c + mf[af + 1] + mf[af + 2] * length + mf[af + 3] * vel
            if mi[a + 3] != 0:
                force = min(max(force, mf[af + 6]), mf[af + 7])
            res[(OAF + u) * F + f] = force
            res[(OQACT + dof) * F + f] += gear * force
        elif mi[a] == 3:  # fixed tendon: moment = gear x the wrap coefs, length and velocity = gear x the tendon's
            var t = Int(mi[a + 1])
            var lv = tendon(mi, mf, qpos, qvel, f, t)
            comptime if SPATIAL:
                lv = (get(res, f, OTLEN + t), get(res, f, OTVEL + t))
            var force = mf[af] * c + mf[af + 1] + mf[af + 2] * (lv[0] * gear) + mf[af + 3] * (lv[1] * gear)
            if mi[a + 3] != 0:
                force = min(max(force, mf[af + 6]), mf[af + 7])
            res[(OAF + u) * F + f] = force
            comptime if SPATIAL:
                for v in range(NV):
                    res[(OQACT + v) * F + f] += gear * get(res, f, OTJ + t * NV + v) * force
                continue
            var w0 = Int(mi[TNI + t])
            for w in range(w0, w0 + Int(mi[TNI + NTEN + t])):
                res[(OQACT + Int(mi[TNI + 2 * NTEN + w])) * F + f] += gear * mf[TNF + w] * force
        elif NMUSCLE > 0 and mi[a] == 4:  # compile-time false for legacy models; state lives in persistent output rows
            var t = Int(mi[a + 1])
            var m = Int(mi[a + 4])
            var p = CMUSF + 11 * m
            var lv = tendon(mi, mf, qpos, qvel, f, t)
            comptime if SPATIAL:
                lv = (get(res, f, OTLEN + t), get(res, f, OTVEL + t))
            var x = (lv[0] - mf[p + 1]) / mf[p + 2]
            var fl = exp(-(x * x))
            var vx = lv[1] / mf[p + 3]
            var fv: Float32 = 0
            if vx <= -1.0:
                fv = 0
            elif vx < 0.0:
                fv = (1.0 + vx) / (1.0 - vx / mf[p + 4])
            else:
                fv = min(mf[p + 4], 1.0 + (mf[p + 4] - 1.0) * vx / (1.0 + vx))
            var state = OMUS + 2 * m
            var activation0 = get(res, f, state)
            var fatigue0 = get(res, f, state + 1)
            var tau = mf[p + 7] if c > activation0 else mf[p + 8]
            var activation1 = c + (activation0 - c) * exp(-mf[TSTEP] / tau)
            var output = activation0 * (1.0 - fatigue0) * fl * fv
            output = 0.0 if output < 0.0 else (1.0 if output > 1.0 else output)
            var fatigue_target = output
            tau = mf[p + 9] if fatigue_target > fatigue0 else mf[p + 10]
            var fatigue1 = fatigue_target + (fatigue0 - fatigue_target) * exp(-mf[TSTEP] / tau)
            res[state * F + f] = activation1
            res[(state + 1) * F + f] = fatigue1
            var force = activation1 * (1.0 - fatigue0) * mf[p] * fl * fv
            var stretch = lv[0] - mf[p + 6]
            stretch = stretch if stretch > 0.0 else 0.0
            force += mf[p + 5] * stretch * stretch
            res[(OAF + u) * F + f] = force
            comptime if SPATIAL:
                for v in range(NV):
                    res[(OQACT + v) * F + f] += get(res, f, OTJ + t * NV + v) * force
                continue
            var w0 = Int(mi[TNI + t])
            for w in range(w0, w0 + Int(mi[TNI + NTEN + t])):
                res[(OQACT + Int(mi[TNI + 2 * NTEN + w])) * F + f] += mf[TNF + w] * force
        else:  # body: adhesion over the body's contacts
            var body = Int(mi[a + 1])
            var counter = 0
            for ip in range(npairs(res, f)):
                var p = pick(res, f, ip)
                if p < 0:
                    continue
                if Int(mi[CPH + 11 * p + 2]) != body and Int(mi[GBODY + Int(mi[CPH + 11 * p])]) != body:  # either side (mj_transmission)
                    continue
                for sl in range(nslot(mi, res, f, p)):
                    counter += 1 if get(res, f, OVALID + sbase(mi, res, f, p) + sl) != 0 else 0
            var force = mf[af] * c + mf[af + 1]  # length and velocity of a body transmission are 0
            if mi[a + 3] != 0:
                force = min(max(force, mf[af + 6]), mf[af + 7])
            res[(OAF + u) * F + f] = force
            if counter == 0:
                continue
            var scale = -force / Float32(counter)
            for ip in range(npairs(res, f)):
                var p = pick(res, f, ip)
                if p < 0:
                    continue
                if Int(mi[CPH + 11 * p + 2]) != body and Int(mi[GBODY + Int(mi[CPH + 11 * p])]) != body:
                    continue
                var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
                var one = ELL  # condim 1 or elliptic: the normal is row 0 alone
                comptime if NCV > 0:
                    one = one or Int(mi[CCND + p]) == 1
                for sl in range(nslot(mi, res, f, p)):
                    if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                        continue
                    var jadr = OJ + jbase(mi, res, f, p) + RS * sl * nch
                    for k in range(nch):  # normal row = mean of the pyramid's first pair
                        var jn = get(res, f, jadr + k) if one else 0.5 * (get(res, f, jadr + k) + get(res, f, jadr + nch + k))
                        res[(OQACT + Int(mi[CCHAIN + choff + k])) * F + f] += scale * jn

    comptime if GRAVCOMP and NU > 0:  # actuator-level gravcomp (engine_forward.c:770), only where mj_fwdActuation runs (nu > 0)
        for j in range(NJ):
            if mi[CGC + j] == 0:
                continue
            var t = Int(mi[JTYPE + j])
            var da = Int(mi[DOFADR + j])
            for d in range(da, da + (6 if t == 0 else (3 if t == 1 else 1))):
                res[(OQACT + d) * F + f] += get(res, f, OGC + d)

    comptime if NAFRC > 0:  # clamp qfrc_actuator to joint-level actuator force limits (clampVec, engine_forward.c:788): the joint's first dof
        for i in range(NAFRC):
            var j = Int(mi[CAFR + i])
            res[(OQACT + j) * F + f] = min(max(get(res, f, OQACT + j), mf[AFRF + 2 * i]), mf[AFRF + 2 * i + 1])


def equality(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO, f: Int):
    """MuJoCo mj_instantiateEquality (engine_core_constraint.c:513) for connect, weld, joint and tendon constraints, mj_diagApprox
    (:1637), mj_makeImpedance (:2069) and mj_referenceConstraint (:3048) with its mj_Jdotv correction (:973): per constraint
    the rows' residual, Jacobian over the constraint's chain (mj_jacDifPair, anchors 0 minus 1, weld rotation corrected
    by 0.5 neg(q1) J q0 relpose), D and aref. Positions are relative to the floating origin of the animal (a body of the
    world tree is shifted into it); the impedance of a connect or weld is a function of the norm of its residual."""
    var h = mf[TSTEP]
    for e in range(NEQ):
        var eh = CEQH + EQH * e
        var typ = Int(mi[eh]); var b0 = Int(mi[eh + 1]); var b1 = Int(mi[eh + 2]); var size = Int(mi[eh + 3])
        var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
        var ef = CEQF + EQF * e
        var cpos = S6(0)
        var jdv = S6(0)  # Jdot qvel of the rows, subtracted from aref
        var posn: Float32
        if typ == 6:  # dof friction loss (mj_instantiateFriction, engine_core_constraint.c:1207): J = 1 on the dof, pos 0 (so K drops out)
            var dof = Int(mi[eh + 8])
            for c in range(nch):
                res[(jb + c) * F + f] = Float32(1) if Int(mi[CEQC + choff + c]) == dof else Float32(0)
            posn = 0
        elif typ == 7:  # tendon friction loss (mj_instantiateFriction, engine_core_constraint.c:1240): J = ten_J, pos 0
            var t = Int(mi[eh + 8])
            for c in range(nch):
                res[(jb + c) * F + f] = ten_j(mi, mf, res, f, t, Int(mi[CEQC + choff + c]))
            posn = 0
        elif typ == 4:  # joint limit (mj_instantiateLimit): dist = side (range - qpos), a row while dist < margin, J = -side on the joint's dof
            var dof = Int(mi[eh + 8]); var side = Float32(Int(mi[eh + 9]))
            var dist = side * (mf[ef + 24] - qpos[Int(mi[eh + 10]) * F + f])
            var on = dist < mf[ef + 25]
            for c in range(nch):
                res[(jb + c) * F + f] = -side if on and Int(mi[CEQC + choff + c]) == dof else Float32(0)
            if not on:
                res[(OEQD + rbase) * F + f] = 0
                res[(OEQA + rbase) * F + f] = 0
                continue
            cpos[0] = dist - mf[ef + 25]
            posn = cpos[0]
        elif typ == 3:  # tendon limit (mj_instantiateLimit, engine_core_constraint.c:1277): dist = side (range - length), a row while dist < margin, J = -side ten_J
            var t = Int(mi[eh + 8]); var side = Float32(Int(mi[eh + 9]))
            var L: Float32
            comptime if SPATIAL:
                L = get(res, f, OTLEN + t)
            else:
                L = tendon(mi, mf, qpos, qvel, f, t)[0]
            var dist = side * (mf[ef + 24] - L)
            var on = dist < mf[ef + 25]
            for c in range(nch):
                res[(jb + c) * F + f] = -side * ten_j(mi, mf, res, f, t, Int(mi[CEQC + choff + c])) if on else Float32(0)
            if not on:
                res[(OEQD + rbase) * F + f] = 0
                res[(OEQA + rbase) * F + f] = 0
                continue
            cpos[0] = dist - mf[ef + 25]
            posn = cpos[0]
        elif typ == 5:  # tendon (mjEQ_TENDON, engine_core_constraint.c:645): L0 - length0_0 - c0 - poly(L1 - length0_1), J = J0 - deriv J1 (mju_addToScl)
            var t0 = Int(mi[eh + 8]); var t1 = Int(mi[eh + 9])
            cpos[0] = ten_len(mi, mf, qpos, qvel, res, f, t0) - mf[ef + 29] - mf[ef + 24]
            var deriv: Float32 = 0
            if t1 >= 0:
                var dif = ten_len(mi, mf, qpos, qvel, res, f, t1) - mf[ef + 30]
                cpos[0] -= mf[ef + 25] * dif + mf[ef + 26] * dif * dif + mf[ef + 27] * dif * dif * dif + mf[ef + 28] * dif * dif * dif * dif
                deriv = mf[ef + 25] + 2 * mf[ef + 26] * dif + 3 * mf[ef + 27] * dif * dif + 4 * mf[ef + 28] * dif * dif * dif
            for c in range(nch):
                var d = Int(mi[CEQC + choff + c])
                var jt = ten_j(mi, mf, res, f, t0, d)
                if t1 >= 0:
                    jt += -deriv * ten_j(mi, mf, res, f, t1, d)
                res[(jb + c) * F + f] = jt
            posn = cpos[0]
        elif typ == 2:  # joint: q0 - qpos0 - c0 - poly(q1 - qpos1_0) (engine_core_constraint.c:808)
            var d0 = Int(mi[eh + 8]); var d1 = Int(mi[eh + 9])
            cpos[0] = qpos[Int(mi[eh + 10]) * F + f] - mf[ef + 29] - mf[ef + 24]
            var deriv: Float32 = 0
            if d1 >= 0:
                var dif = qpos[Int(mi[eh + 11]) * F + f] - mf[ef + 30]
                cpos[0] -= mf[ef + 25] * dif + mf[ef + 26] * dif * dif + mf[ef + 27] * dif * dif * dif + mf[ef + 28] * dif * dif * dif * dif
                deriv = mf[ef + 25] + 2 * mf[ef + 26] * dif + 3 * mf[ef + 27] * dif * dif + 4 * mf[ef + 28] * dif * dif * dif
            for c in range(nch):
                var d = Int(mi[CEQC + choff + c])
                res[(jb + c) * F + f] = (Float32(1) if d == d0 else Float32(0)) - (deriv if d == d1 else Float32(0))
            posn = cpos[0]
        else:
            var weld = typ == 1
            var o0 = origin(mi, qpos, f, b0); var o1 = origin(mi, qpos, f, b1)
            var frame = origin(mi, qpos, f, Int(mi[eh + 8]))
            var sh0 = o0 - frame; var sh1 = o1 - frame
            var B0 = body(res, f, b0); var B1 = body(res, f, b1)
            var p0 = rot(B0[1], m3(mf, ef)) + B0[0] + sh0
            var p1 = rot(B1[1], m3(mf, ef + 3)) + B1[0] + sh1
            var r0 = 3 * Int(mi[ROOT + b0]); var r1 = 3 * Int(mi[ROOT + b1])
            var off0 = p0 - V3(get(res, f, OSC + r0), get(res, f, OSC + r0 + 1), get(res, f, OSC + r0 + 2), 0) - sh0
            var off1 = p1 - V3(get(res, f, OSC + r1), get(res, f, OSC + r1 + 1), get(res, f, OSC + r1 + 2), 0) - sh1
            var d01 = p0 - p1
            for k in range(3):
                cpos[k] = d01[k]
            var qa = mul(B0[1], m4(mf, ef + 6)); var qb = mul(B1[1], m4(mf, ef + 10))
            var nq1 = Q4(qb[0], -qb[1], -qb[2], -qb[3])
            var ts = mf[ef + 14]
            if weld:  # neg(q1) q0 relpose, axis components scaled by torquescale
                var qq = mul(nq1, qa)
                for k in range(3):
                    cpos[3 + k] = qq[1 + k] * ts
            var cv0 = get6(res, f, OCV + 6 * (b0 - 1)) if b0 > 0 else S6(0)
            var cv1 = get6(res, f, OCV + 6 * (b1 - 1)) if b1 > 0 else S6(0)
            var om0 = V3(cv0[0], cv0[1], cv0[2], 0); var om1 = V3(cv1[0], cv1[1], cv1[2], 0)
            # point velocity (mju_transformSpatial from the root's com to the anchor)
            var pv0 = V3(cv0[3], cv0[4], cv0[5], 0) - cross(off0, om0)
            var pv1 = V3(cv1[3], cv1[4], cv1[5], 0) - cross(off1, om1)
            var jd0 = V3(0); var jd1 = V3(0); var jr0 = V3(0); var jr1 = V3(0)
            for c in range(nch):
                var d = Int(mi[CEQC + choff + c]); var mk = Int(mi[CEQM + choff + c])
                var cd = OCD + 6 * d
                var ang = V3(get(res, f, cd), get(res, f, cd + 1), get(res, f, cd + 2), 0)
                var lin = V3(get(res, f, cd + 3), get(res, f, cd + 4), get(res, f, cd + 5), 0)
                var jp = V3(0); var jr = V3(0)
                var qv = qvel[d * F + f]
                var cdot = get6(res, f, OCDD + 6 * d)
                if mk & 4 != 0:  # mj_jacDot: a quaternion dof takes its body's final velocity
                    cdot = cross_motion(get6(res, f, OCV + 6 * (Int(mi[DBODY + d]) - 1)), get6(res, f, cd))
                var cda = V3(cdot[0], cdot[1], cdot[2], 0); var cdl = V3(cdot[3], cdot[4], cdot[5], 0)
                if mk & 1 != 0:
                    jp += lin + cross(ang, off0)
                    jr += ang
                    jd0 += (cdl + cross(cda, off0) + cross(ang, pv0)) * qv
                    jr0 += cda * qv
                if mk & 2 != 0:
                    jp -= lin + cross(ang, off1)
                    jr -= ang
                    jd1 += (cdl + cross(cda, off1) + cross(ang, pv1)) * qv
                    jr1 += cda * qv
                for k in range(3):
                    res[(jb + k * nch + c) * F + f] = jp[k]
                if weld:  # 0.5 neg(q1) (J0 - J1) q0 relpose, axis components
                    var q3 = mul(mul(nq1, Q4(0, jr[0], jr[1], jr[2])), qa)
                    for k in range(3):
                        res[(jb + (3 + k) * nch + c) * F + f] = 0.5 * q3[1 + k] * ts
            for k in range(3):
                jdv[k] = jd0[k] - jd1[k]
            if weld:  # mj_Jdotv rotational part: product rule on 0.5 neg(q1) (omega0 - omega1) q0 relpose
                var dom = om0 - om1
                var qd0 = Q4(0.5 * (-om0[0] * qa[1] - om0[1] * qa[2] - om0[2] * qa[3]), 0.5 * (om0[0] * qa[0] + om0[1] * qa[3] - om0[2] * qa[2]),
                             0.5 * (-om0[0] * qa[3] + om0[1] * qa[0] + om0[2] * qa[1]), 0.5 * (om0[0] * qa[2] - om0[1] * qa[1] + om0[2] * qa[0]))
                var qd1 = Q4(0.5 * (-om1[0] * qb[1] - om1[1] * qb[2] - om1[2] * qb[3]), 0.5 * (om1[0] * qb[0] + om1[1] * qb[3] - om1[2] * qb[2]),
                             0.5 * (-om1[0] * qb[3] + om1[1] * qb[0] + om1[2] * qb[1]), 0.5 * (om1[0] * qb[2] - om1[1] * qb[1] + om1[2] * qb[0]))
                var dj = jr0 - jr1
                var t1 = mul(mul(Q4(qd1[0], -qd1[1], -qd1[2], -qd1[3]), Q4(0, dom[0], dom[1], dom[2])), qa)
                var t2 = mul(mul(nq1, Q4(0, dj[0], dj[1], dj[2])), qa)
                var t3 = mul(mul(nq1, Q4(0, dom[0], dom[1], dom[2])), qd0)
                for k in range(3):
                    jdv[3 + k] = 0.5 * (t1[1 + k] + t2[1 + k] + t3[1 + k]) * ts
            var sq = cpos[0] * cpos[0] + cpos[1] * cpos[1] + cpos[2] * cpos[2]
            if weld:
                sq += cpos[3] * cpos[3] + cpos[4] * cpos[4] + cpos[5] * cpos[5]
            posn = sqrt(sq)
        # mj_diagApprox, getsolparam, getimpedance, makeImpedance, referenceConstraint
        # getsolparam: refsafe on the standard format only (a mixed format arrives as the default from the export); non-positive = direct
        var ref0 = mf[ef + 15] if mf[ef + 15] <= 0 else max(mf[ef + 15], 2 * h); var ref1 = mf[ef + 16]
        var s0 = min(Float32(0.9999), max(Float32(1e-4), mf[ef + 17])); var s1 = min(Float32(0.9999), max(Float32(1e-4), mf[ef + 18]))
        var s2 = max(Float32(0), mf[ef + 19]); var s3 = min(Float32(0.9999), max(Float32(1e-4), mf[ef + 20])); var s4 = max(Float32(1), mf[ef + 21])
        var imp = impedance(s0, s1, s2, s3, s4, posn)
        var K: Float32; var B: Float32
        if ref0 > 0:  # mj_makeImpedance standard: K = 1 / (d_width^2 timeconst^2 dampratio^2)
            K = 1 / max(Float32(1e-15), s1 * s1 * ref0 * ref0 * ref1 * ref1)
        else:  # direct: K = -solref[0] / d_width^2
            K = -ref0 / max(Float32(1e-15), s1 * s1)
        if ref1 > 0:  # standard: B = 2 / (d_width timeconst)
            B = 2 / max(Float32(1e-15), s1 * ref0)
        else:  # direct: B = -solref[1] / d_width
            B = -ref1 / max(Float32(1e-15), s1)
        var on = Float32(Int(mi[eh + 12]))  # MuJoCo eq_active0: d->eq_active's reset value, the state without -DEQACT
        comptime if EQACT:  # an inactive equality's rows get D = 0: no force, no Hessian term, no cost (MuJoCo drops them); the host's row wins
            if e < NEQE:  # one host row per MuJoCo equality (m.neq), none for friction or limit rows
                on = ctrl[(NU + 7 * NMOCAP + 7 * NXF + e) * F + f]
        for k in range(size):
            var row = rbase + k
            var dA = mf[ef + 23] if typ == 1 and k >= 3 else mf[ef + 22]
            res[(OEQD + row) * F + f] = on / max(Float32(1e-15), (1 - imp) * dA / imp)
            var vel: Float32 = 0
            for c in range(nch):
                vel += get(res, f, jb + k * nch + c) * qvel[Int(mi[CEQC + choff + c]) * F + f]
            res[(OEQA + row) * F + f] = -B * vel - K * imp * cpos[k] - jdv[k]


@always_inline
def cone_fric(mf: TF, p: Int) -> V3:
    """Elliptic cone of pair p: mu of the regularized cone = f0 sqrt(R1/R0) = f0 / sqrt(impratio) (mj_makeImpedance), f0, f1."""
    var f0 = mf[CPF + 14 * p + 2]
    return V3(f0 / sqrt(max(Float32(1e-15), mf[COPT])), f0, mf[CPF + 14 * p + 3], 0)


@always_inline
def floss(mf: TF, r: Int) -> Float32:
    """Friction-loss row r (NEQB <= r < NEQL): its dof or tendon frictionloss; with R = 1 / D the row is quadratic while -R floss < jar < R floss."""
    return mf[CEQF + EQF * (NEQFE + r - NEQB) + 24]


@always_inline
def fquad(mf: TF, res: TO, f: Int, r: Int, jar: Float32) -> Bool:
    """MuJoCo mjCNSTRSTATE_QUADRATIC of friction row r at jar (mj_constraintUpdate_impl, engine_core_constraint.c:3100)."""
    var Rf = floss(mf, r) / get(res, f, OEQD + r)
    return -Rf < jar and jar < Rf


def update_cone_n(mi: TI, mf: TF, res: TO, f: Int) -> Float32:
    """update_cone for any condim (DIMX > 3; mj_constraintUpdate_impl, engine_core_constraint.c:3155): U = jar diag(mu, friction),
    N = U0, T = |U1..U(dim-1)|, the same three zones; the middle zone's force -Dm (N - mu T) mu on the normal, -f0 / T U_j friction(j-1)."""
    var cost: Float32 = 0
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        var sc = pscale(mf, p)
        var mu = sc[0]
        var dim = pdim(mi, p)
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            var r = RS * (sbase(mi, res, f, p) + sl)
            var jar = U8(0); var D = U8(0)
            for k in range(dim):
                jar[k] = get(res, f, OJAR + r + k)
                D[k] = get(res, f, OD + r + k)
            var U = jar * sc
            var N = U[0]
            var Ut = U
            Ut[0] = 0
            var T = sqrt((Ut * Ut).reduce_add())
            var force = U8(0)
            var zone: Float32
            if N >= mu * T or (T <= 0 and N >= 0):
                zone = 0
            elif mu * N + T <= 0 or (T <= 0 and N < 0):
                zone = 1
                force = -D * jar
                cost += 0.5 * (D * jar * jar).reduce_add()
            else:
                zone = 2
                var Dm = D[0] / (mu * mu * (1 + mu * mu))
                var NmT = N - mu * T
                cost += 0.5 * Dm * NmT * NmT
                var fnorm = -Dm * NmT * mu
                force = -fnorm / T * U * sc
                force[0] = fnorm
            for k in range(NR):
                res[(OST + r + k) * F + f] = zone if k < dim else Float32(0)
                res[(OFRC + r + k) * F + f] = force[k]
                if force[k] != 0:
                    var jadr = OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch
                    for c in range(nch):
                        res[(OQFC + Int(mi[CCHAIN + choff + c])) * F + f] += get(res, f, jadr + c) * force[k]
    return cost


def peval_cone_n(mi: TI, mf: TF, res: TO, f: Int, alpha: Float32) -> U8:
    """peval's elliptic terms for any condim (DIMX > 3; PrimalEval, engine_solver.c:1360): bottom-zone quadratic q0 q1 q2, then
    the middle zone's cost and its two derivatives, over this lane's valid slots."""
    var t = U8(0)
    for a in range(lane(), Int(get(res, f, OACT)), LN):
        var sl = Int(get(res, f, OACT + 1 + a))
        var r = RS * sl
        var p = Int(get(res, f, OVALID + sl)) - 1
        var sc = pscale(mf, p)
        var mu = sc[0]
        var dim = pdim(mi, p)
        var jar = U8(0); var jv = U8(0); var D = U8(0)
        for k in range(dim):
            jar[k] = get(res, f, OJAR + r + k); jv[k] = get(res, f, OJV + r + k); D[k] = get(res, f, OD + r + k)
        var u = jar * sc; var v = jv * sc
        var U0 = u[0]; var V0 = v[0]
        u[0] = 0; v[0] = 0
        var UU = (u * u).reduce_add(); var UV = (u * v).reduce_add(); var VV = (v * v).reduce_add()
        var N = U0 + alpha * V0
        var Tsqr = UU + alpha * (2 * UV + alpha * VV)
        var bottom = False
        var T = sqrt(max(Tsqr, Float32(0)))
        if Tsqr <= 0:
            bottom = N < 0
        else:
            if N >= mu * T:
                pass
            elif mu * N + T <= 0:
                bottom = True
            else:
                var Dm = D[0] / (mu * mu * (1 + mu * mu))
                var T1 = (UV + alpha * VV) / T
                var T2 = VV / T - (UV + alpha * VV) * T1 / (T * T)
                t[3] += 0.5 * Dm * (N - mu * T) * (N - mu * T)
                t[4] += Dm * (N - mu * T) * (V0 - mu * T1)
                t[5] += Dm * ((V0 - mu * T1) * (V0 - mu * T1) + (N - mu * T) * (-mu * T2))
        if bottom:
            t[0] += 0.5 * (D * jar * jar).reduce_add()
            t[1] += (D * jar * jv).reduce_add()
            t[2] += 0.5 * (D * jv * jv).reduce_add()
    return t


def hessian_cone_n(mi: TI, mf: TF, res: TO, f: Int, p: Int, sl: Int):
    """hessian's cone block for any condim (DIMX > 3; HessianCone, engine_core_constraint.c:3208): for a middle-zone contact
    H = Dm diag(mu, friction) h diag(mu, friction), h = [1, -mu/T U'; -mu/T U, mu N/T^3 U U' + (mu^2 - mu N/T) I], added J' H J."""
    var r0 = RS * (sbase(mi, res, f, p) + sl)
    if get(res, f, OST + r0) != 2:
        return
    var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
    var sc = pscale(mf, p)
    var mu = sc[0]
    var dim = pdim(mi, p)
    var U = U8(0)
    for k in range(dim):
        U[k] = get(res, f, OJAR + r0 + k) * sc[k]
    var N = U[0]
    var Ut = U
    Ut[0] = 0
    var T = sqrt((Ut * Ut).reduce_add())
    var Dm = get(res, f, OD + r0) / (mu * mu * (1 + mu * mu))
    var H = SIMD[DType.float32, 64](0)
    for k in range(dim):
        for j in range(dim):
            var hk: Float32
            if k == 0 and j == 0:
                hk = 1
            elif k == 0 or j == 0:
                hk = -mu / T * U[k + j]
            else:
                hk = mu * N / (T * T * T) * U[k] * U[j] + (mu * mu - mu * N / T if k == j else Float32(0))
            H[8 * k + j] = Dm * sc[k] * sc[j] * hk
    var jb = OJ + jbase(mi, res, f, p) + RS * sl * nch
    for a in range(nch):
        var d = Int(mi[CCHAIN + choff + a])
        if d % LN != lane():
            continue
        var ra = Int(mi[HADR + d])
        var w = U8(0)
        for k in range(dim):
            var ja = get(res, f, jb + k * nch + a)
            for j in range(dim):
                w[j] += ja * H[8 * k + j]
        for bb in range(a + 1):
            var jbv = U8(0)
            for j in range(dim):
                jbv[j] = get(res, f, jb + j * nch + bb)
            res[(OHN + ra + bb) * F + f] += (w * jbv).reduce_add()


def update_cone(mi: TI, mf: TF, res: TO, f: Int) -> Float32:
    """MuJoCo mj_constraintUpdate_impl, elliptic contacts (engine_core_constraint.c:3155): U = (mu jar0, f0 jar1, f1 jar2), N = U0,
    T = |U1 U2|; top zone (N >= mu T): no force, state 0; bottom zone (mu N + T <= 0): quadratic, state 1; middle: cost
    0.5 Dm (N - mu T)^2 with Dm = D0 / (mu^2 (1 + mu^2)), state 2. Forces to OFRC, qfrc_constraint += J' force, cost returned."""
    comptime if DIMX > 3:
        return update_cone_n(mi, mf, res, f)
    var cost: Float32 = 0
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        var fr = cone_fric(mf, p)
        var mu = fr[0]
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            var r = RS * (sbase(mi, res, f, p) + sl)
            var jar = V3(get(res, f, OJAR + r), get(res, f, OJAR + r + 1), get(res, f, OJAR + r + 2), 0)
            var D = V3(get(res, f, OD + r), get(res, f, OD + r + 1), get(res, f, OD + r + 2), 0)
            var U = V3(jar[0] * mu, jar[1] * fr[1], jar[2] * fr[2], 0)
            var N = U[0]
            var T = sqrt(U[1] * U[1] + U[2] * U[2])
            var force = V3(0)
            var zone: Float32
            if N >= mu * T or (T <= 0 and N >= 0):
                zone = 0
            elif mu * N + T <= 0 or (T <= 0 and N < 0):
                zone = 1
                force = -D * jar
                cost += 0.5 * (D * jar * jar).reduce_add()
            else:
                zone = 2
                var Dm = D[0] / (mu * mu * (1 + mu * mu))
                var NmT = N - mu * T
                cost += 0.5 * Dm * NmT * NmT
                var f0 = -Dm * NmT * mu
                force = V3(f0, -f0 / T * U[1] * fr[1], -f0 / T * U[2] * fr[2], 0)
            for k in range(3):
                res[(OST + r + k) * F + f] = zone
                res[(OFRC + r + k) * F + f] = force[k]
                if force[k] != 0:
                    var jadr = OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch
                    for c in range(nch):
                        res[(OQFC + Int(mi[CCHAIN + choff + c])) * F + f] += get(res, f, jadr + c) * force[k]
    return cost


def update(mi: TI, mf: TF, res: TO, f: Int) -> Float32:
    """MuJoCo PrimalUpdateConstraint, pyramidal rows: state = jar < 0, force = -D jar, qfrc_constraint = J' force,
    returns cost = sum 0.5 D jar^2 over active rows + Gauss."""
    var cost: Float32 = 0
    for v in range(NV):
        res[(OQFC + v) * F + f] = 0
    comptime if ELL:
        cost = update_cone(mi, mf, res, f)
    for ip in range(0 if ELL else npairs(res, f)):
        var p = pick(res, f, ip)
        if p < 0:
            continue
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(RS):
                var row = RS * (sbase(mi, res, f, p) + sl) + k
                var jar = get(res, f, OJAR + row)
                var act = jar < 0
                res[(OST + row) * F + f] = 1 if act else 0
                if act:
                    var D = get(res, f, OD + row)
                    cost += 0.5 * D * jar * jar
                    var force = -D * jar
                    var jadr = OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch
                    for c in range(nch):
                        var v = Int(mi[CCHAIN + choff + c])
                        res[(OQFC + v) * F + f] += get(res, f, jadr + c) * force
    comptime if NEQ > 0:  # equality rows are always active (quadratic); a limit row only while jar < 0; a friction row linear outside +-R floss
        for e in range(NEQ):
            var eh = CEQH + EQH * e
            var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
            for k in range(Int(mi[eh + 3])):
                var r = rbase + k
                var jar = get(res, f, OEQJAR + r)
                if r >= NEQL and jar >= 0:
                    continue
                var D = get(res, f, OEQD + r)
                var force = -D * jar
                if r >= NEQB and r < NEQL and not fquad(mf, res, f, r, jar):  # linear negative (jar <= -R floss) or positive
                    var fl = floss(mf, r)
                    var R = 1 / D
                    force = fl if jar < 0 else -fl
                    cost += -0.5 * R * fl * fl + (-fl * jar if jar < 0 else fl * jar)
                else:
                    cost += 0.5 * D * jar * jar
                for c in range(nch):
                    res[(OQFC + Int(mi[CEQC + choff + c])) * F + f] += get(res, f, jb + k * nch + c) * force
    for v in range(NV):
        cost += 0.5 * (get(res, f, OMA + v) - get(res, f, OQSM + v)) * (get(res, f, OCQACC + v) - get(res, f, OQACCS + v))
    return cost


def hessian(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo MakeHessian + FactorizeHessian: H = M + J' D J over active rows, in H's pattern (model.py hpattern: M's
    plus the fill of pairs coupling two limbs; a pair's dof chain is exactly the pattern of its deepest row),
    factored L'DL with mj_factorI."""
    res[(ODIAG + 2) * F + f] += 1
    for c in range(NH):
        res[(OHN + c) * F + f] = 0
    for c in range(NC):
        res[(OHN + Int(mi[MTOH + c])) * F + f] = get(res, f, OM + c)
    sync()
    for ip in range(npairs(res, f)):
        var p = pick(res, f, ip)
        if p < 0:
            continue
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            comptime if ELL and DIMX > 3:
                hessian_cone_n(mi, mf, res, f, p, sl)
            comptime if ELL and DIMX <= 3:  # middle zone: the cone Hessian block (HessianCone, engine_core_constraint.c:3208)
                var r0 = RS * (sbase(mi, res, f, p) + sl)
                if get(res, f, OST + r0) == 2:
                    var fr = cone_fric(mf, p)
                    var mu = fr[0]
                    var U = V3(get(res, f, OJAR + r0) * mu, get(res, f, OJAR + r0 + 1) * fr[1], get(res, f, OJAR + r0 + 2) * fr[2], 0)
                    var N = U[0]
                    var T = sqrt(U[1] * U[1] + U[2] * U[2])
                    var Dm = get(res, f, OD + r0) / (mu * mu * (1 + mu * mu))
                    var sc = mu * N / (T * T * T)
                    var dg = mu * mu - mu * N / T
                    var H00 = Dm * mu * mu
                    var H01 = Dm * mu * fr[1] * (-mu / T * U[1]); var H02 = Dm * mu * fr[2] * (-mu / T * U[2])
                    var H11 = Dm * fr[1] * fr[1] * (sc * U[1] * U[1] + dg); var H22 = Dm * fr[2] * fr[2] * (sc * U[2] * U[2] + dg)
                    var H12 = Dm * fr[1] * fr[2] * (sc * U[1] * U[2])
                    var jb = OJ + jbase(mi, res, f, p) + RS * sl * nch
                    for a in range(nch):
                        var d = Int(mi[CCHAIN + choff + a])
                        if d % LN != lane():
                            continue
                        var ra = Int(mi[HADR + d])
                        var a0 = get(res, f, jb + a); var a1 = get(res, f, jb + nch + a); var a2 = get(res, f, jb + 2 * nch + a)
                        for bb in range(a + 1):
                            var b0 = get(res, f, jb + bb); var b1 = get(res, f, jb + nch + bb); var b2 = get(res, f, jb + 2 * nch + bb)
                            res[(OHN + ra + bb) * F + f] += (a0 * (H00 * b0 + H01 * b1 + H02 * b2) + a1 * (H01 * b0 + H11 * b1 + H12 * b2)
                                                            + a2 * (H02 * b0 + H12 * b1 + H22 * b2))
            for k in range(NR):
                var row = RS * (sbase(mi, res, f, p) + sl) + k
                if get(res, f, OST + row) == 0:
                    continue
                comptime if ELL:  # only the bottom zone is a plain quadratic
                    if get(res, f, OST + row) != 1:
                        continue
                var D = get(res, f, OD + row)
                var jadr = OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch
                for a in range(nch):  # lane owns H row d (d % LN): every row the same lane, no race across pairs
                    var d = Int(mi[CCHAIN + choff + a])
                    if d % LN != lane():
                        continue
                    var ja = get(res, f, jadr + a) * D
                    var ra = Int(mi[HADR + d])
                    for bb in range(a + 1):
                        res[(OHN + ra + bb) * F + f] += ja * get(res, f, jadr + bb)
    comptime if NEQ > 0:  # equality rows: always in H; a limit row while jar < 0; a friction row while quadratic
        for e in range(NEQ):
            var eh = CEQH + EQH * e
            var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
            for k in range(Int(mi[eh + 3])):
                var r = rbase + k
                if r >= NEQL and get(res, f, OEQJAR + r) >= 0:
                    continue
                if r >= NEQB and r < NEQL and not fquad(mf, res, f, r, get(res, f, OEQJAR + r)):
                    continue
                var D = get(res, f, OEQD + rbase + k)
                var jadr = jb + k * nch
                for a in range(nch):
                    var d = Int(mi[CEQC + choff + a])
                    if d % LN != lane():
                        continue
                    var ja = get(res, f, jadr + a) * D
                    var ra = Int(mi[HADR + d])
                    for bb in range(a + 1):
                        res[(OHN + ra + bb) * F + f] += ja * get(res, f, jadr + bb)
    sync()
    factorI(mi, res, f, OHN, OHNI, HADR)


def lowrank(mi: TI, res: TO, f: Int) -> Bool:
    """Active-set change without refactoring: before update rewrites OST, every row turning active (inactive) is a
    rank-1 update (downdate) H +- D j j' of the factored L'DL in OHN (Gill-Golub-Murray-Saunders C1 in elimination
    order, k from the row's deepest dof back to the root). j lives on the pair's chain, which is the pattern of its
    deepest row, so the walk stays on that row's elimination path: no fill. w in OMV (free until the next search).
    False when a downdate's pivot collapses (float32): the caller rebuilds H from scratch."""
    for ip in range(npairs(res, f)):
        var p = pick(res, f, ip)
        if p < 0:
            continue
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        var deep = 0
        for c in range(nch):
            deep = max(deep, Int(mi[CCHAIN + choff + c]))
        var ds = Int(mi[HADR + deep]); var dn = Int(mi[HADR + NV + deep])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(RS):
                var row = RS * (sbase(mi, res, f, p) + sl) + k
                var now = get(res, f, OJAR + row) < 0
                if now == (get(res, f, OST + row) != 0):
                    continue
                for t in range(dn):
                    res[(OMV + Int(mi[HADR + 2 * NV + ds + t])) * F + f] = 0
                var jadr = OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch
                for c in range(nch):
                    res[(OMV + Int(mi[CCHAIN + choff + c])) * F + f] = get(res, f, jadr + c)
                var a = get(res, f, OD + row) if now else -get(res, f, OD + row)
                for t in range(dn - 1, -1, -1):
                    var kk = Int(mi[HADR + 2 * NV + ds + t])
                    var pk = get(res, f, OMV + kk)
                    if pk == 0:
                        continue
                    var start = Int(mi[HADR + kk]); var diag = Int(mi[HADR + NV + kk]) - 1
                    var dk = get(res, f, OHN + start + diag)
                    var dnew = dk + a * pk * pk
                    if dnew <= 1e-4 * dk:
                        return False
                    var beta = a * pk / dnew
                    a = a * dk / dnew
                    res[(OHN + start + diag) * F + f] = dnew
                    res[(OHNI + kk) * F + f] = 1 / dnew
                    for c in range(diag):
                        var i = Int(mi[HADR + 2 * NV + start + c])
                        var wi = get(res, f, OMV + i) - pk * get(res, f, OHN + start + c)
                        res[(OMV + i) * F + f] = wi
                        res[(OHN + start + c) * F + f] = get(res, f, OHN + start + c) + beta * wi
    return True


def gradient(mi: TI, res: TO, f: Int):
    """MuJoCo PrimalUpdateGradient (Newton): grad = M qacc - qfrc_smooth - qfrc_constraint, search = -H^-1 grad."""
    for v in range(NV):
        var g = get(res, f, OMA + v) - get(res, f, OQSM + v) - get(res, f, OQFC + v)
        res[(OGRAD + v) * F + f] = g
        res[(OSRCH + v) * F + f] = g
    solveLD(mi, res, f, OSRCH, OHN, OHNI, HADR)
    for v in range(NV):
        res[(OSRCH + v) * F + f] = -get(res, f, OSRCH + v)


def gradient_cg(mi: TI, res: TO, f: Int, first: Bool):
    """MuJoCo PrimalUpdateGradient (CG, engine_solver.c:1201): grad = M qacc - qfrc_smooth - qfrc_constraint, preconditioned with the
    plain M factor, Mgrad = M^-1 grad (mj_solveLD with qLD). The previous grad and Mgrad are kept for the Polak-Ribiere beta; the
    first call also starts the search along -Mgrad."""
    for v in range(NV):
        if not first:
            res[(OCGO + v) * F + f] = get(res, f, OGRAD + v)
            res[(OCGMO + v) * F + f] = get(res, f, OCGM + v)
        var g = get(res, f, OMA + v) - get(res, f, OQSM + v) - get(res, f, OQFC + v)
        res[(OGRAD + v) * F + f] = g
        res[(OCGM + v) * F + f] = g
    solveLD(mi, res, f, OCGM, OLD, OLDI)
    if first:
        for v in range(NV):
            res[(OSRCH + v) * F + f] = -get(res, f, OCGM + v)


comptime PNT = SIMD[DType.float32, 4]  # line-search point: alpha, cost, deriv0, deriv1


def peval(mi: TI, mf: TF, res: TO, f: Int, gauss: PNT, alpha: Float32) -> PNT:
    """MuJoCo PrimalEval for pyramidal rows: Gauss quadratic plus every row active at jar + alpha jv (valid rows split
    over lanes, partial sums in OPART)."""
    var q0: Float32 = 0; var q1: Float32 = 0; var q2: Float32 = 0
    var cc: Float32 = 0; var e0: Float32 = 0; var e1: Float32 = 0  # elliptic middle zone: cost and derivatives added directly
    comptime if ELL:
        comptime if DIMX > 3:
            var t = peval_cone_n(mi, mf, res, f, alpha)
            q0 = t[0]; q1 = t[1]; q2 = t[2]; cc = t[3]; e0 = t[4]; e1 = t[5]
        for a in range(lane(), 0 if DIMX > 3 else Int(get(res, f, OACT)), LN):
            var sl = Int(get(res, f, OACT + 1 + a))
            var r = RS * sl
            var fr = cone_fric(mf, Int(get(res, f, OVALID + sl)) - 1)
            var mu = fr[0]
            var jar = V3(get(res, f, OJAR + r), get(res, f, OJAR + r + 1), get(res, f, OJAR + r + 2), 0)
            var jv = V3(get(res, f, OJV + r), get(res, f, OJV + r + 1), get(res, f, OJV + r + 2), 0)
            var D = V3(get(res, f, OD + r), get(res, f, OD + r + 1), get(res, f, OD + r + 2), 0)
            var U0 = jar[0] * mu; var V0 = jv[0] * mu
            var u1 = jar[1] * fr[1]; var u2 = jar[2] * fr[2]; var v1 = jv[1] * fr[1]; var v2 = jv[2] * fr[2]
            var UU = u1 * u1 + u2 * u2; var UV = u1 * v1 + u2 * v2; var VV = v1 * v1 + v2 * v2
            var N = U0 + alpha * V0
            var Tsqr = UU + alpha * (2 * UV + alpha * VV)
            var bottom = False
            var T = sqrt(max(Tsqr, Float32(0)))
            if Tsqr <= 0:
                bottom = N < 0
            else:
                if N >= mu * T:
                    pass
                elif mu * N + T <= 0:
                    bottom = True
                else:
                    var Dm = D[0] / (mu * mu * (1 + mu * mu))
                    var N1 = V0
                    var T1 = (UV + alpha * VV) / T
                    var T2 = VV / T - (UV + alpha * VV) * T1 / (T * T)
                    cc += 0.5 * Dm * (N - mu * T) * (N - mu * T)
                    e0 += Dm * (N - mu * T) * (N1 - mu * T1)
                    e1 += Dm * ((N1 - mu * T1) * (N1 - mu * T1) + (N - mu * T) * (-mu * T2))
            if bottom:
                q0 += 0.5 * (D * jar * jar).reduce_add()
                q1 += (D * jar * jv).reduce_add()
                q2 += 0.5 * (D * jv * jv).reduce_add()
    else:
        for a in range(lane(), RS * nslots(res, f), LN):
            var sl = slot(res, f, a // RS)
            if sl < 0:
                continue
            var r = RS * sl + a % RS
            var jar = get(res, f, OJAR + r); var jv = get(res, f, OJV + r)
            if jar + alpha * jv < 0:
                var D = get(res, f, OD + r)
                q0 += 0.5 * D * jar * jar
                q1 += jv * D * jar
                q2 += 0.5 * jv * D * jv
    comptime if NEQ > 0:  # equality rows: active at every alpha; a limit row where jar + alpha jv < 0; a friction row linear outside +-R floss
        for r in range(lane(), NEQR, LN):
            var jar = get(res, f, OEQJAR + r); var jv = get(res, f, OEQJV + r); var D = get(res, f, OEQD + r)
            if r >= NEQL and jar + alpha * jv >= 0:
                continue
            if r >= NEQB and r < NEQL and not fquad(mf, res, f, r, jar + alpha * jv):  # PrimalEval (engine_solver.c:1335)
                var fl = floss(mf, r)
                var Rf = fl / D
                if jar + alpha * jv <= -Rf:
                    q0 += fl * (-0.5 * Rf - jar)
                    q1 += -fl * jv
                else:
                    q0 += fl * (-0.5 * Rf + jar)
                    q1 += fl * jv
                continue
            q0 += 0.5 * D * jar * jar
            q1 += jv * D * jar
            q2 += 0.5 * jv * D * jv
    comptime if WAVE:
        res[(OPART + lane()) * F + f] = q0
        res[(OPART + LN + lane()) * F + f] = q1
        res[(OPART + 2 * LN + lane()) * F + f] = q2
        comptime if ELL:
            res[(OPART + 3 * LN + lane()) * F + f] = cc
            res[(OPART + 4 * LN + lane()) * F + f] = e0
            res[(OPART + 5 * LN + lane()) * F + f] = e1
        sync()
        q0 = 0; q1 = 0; q2 = 0; cc = 0; e0 = 0; e1 = 0
        for l in range(LN):
            q0 += get(res, f, OPART + l); q1 += get(res, f, OPART + LN + l); q2 += get(res, f, OPART + 2 * LN + l)
            comptime if ELL:
                cc += get(res, f, OPART + 3 * LN + l); e0 += get(res, f, OPART + 4 * LN + l); e1 += get(res, f, OPART + 5 * LN + l)
        sync()
    q0 += gauss[0]; q1 += gauss[1]; q2 += gauss[2]
    res[(ODIAG + 1) * F + f] += 1
    var d1 = 2 * q2 + e1
    return PNT(alpha, alpha * alpha * q2 + alpha * q1 + q0 + cc, 2 * alpha * q2 + q1 + e0, d1 if d1 > 0 else Float32(1e-15))


@always_inline
def bracket(mut p: PNT, c0: PNT, c1: PNT, c2: PNT) -> Bool:
    """MuJoCo updateBracket, the point update; the caller evaluates the next Newton point."""
    var flag = False
    for i in range(3):
        var c = c0 if i == 0 else (c1 if i == 1 else c2)
        if p[2] < 0 and c[2] < 0 and p[2] < c[2]:
            p = c
            flag = True
        elif p[2] > 0 and c[2] > 0 and p[2] > c[2]:
            p = c
            flag = True
    return flag


def search(mi: TI, mf: TF, res: TO, f: Int, scale: Float32, tolerance: Float32, lsiter: Int) -> Float32:
    """MuJoCo PrimalSearch: Newton steps along the search direction, then a bracketed search; returns alpha."""
    var snorm: Float32 = 0
    for v in range(NV):
        snorm += get(res, f, OSRCH + v) * get(res, f, OSRCH + v)
    snorm = sqrt(snorm)
    if snorm < 1e-15:
        return 0
    var gtol = tolerance * snorm / scale
    mulM(mi, res, f, OMV, OSRCH)
    for ip in range(npairs(res, f)):
        var p = pick(res, f, ip)
        if p < 0:
            continue
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(NR):
                res[(OJV + RS * (sbase(mi, res, f, p) + sl) + k) * F + f] = jdot(mi, res, f, OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch, nch, choff, OSRCH)
    comptime if NEQ > 0:
        for e in range(NEQ):
            var eh = CEQH + EQH * e
            var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
            for k in range(Int(mi[eh + 3])):
                res[(OEQJV + rbase + k) * F + f] = jdot(mi, res, f, jb + k * nch, nch, choff, OSRCH, CEQC)
    var g1: Float32 = 0; var g2: Float32 = 0; var g0: Float32 = 0
    for v in range(NV):
        var sv = get(res, f, OSRCH + v)
        g1 += sv * (get(res, f, OMA + v) - get(res, f, OQSM + v))
        g2 += 0.5 * sv * get(res, f, OMV + v)
        g0 += 0.5 * (get(res, f, OMA + v) - get(res, f, OQSM + v)) * (get(res, f, OCQACC + v) - get(res, f, OQACCS + v))
    var gauss = PNT(g0, g1, g2, 0)
    var it = 0
    var p0 = peval(mi, mf, res, f, gauss, 0); it += 1
    gtol = max(gtol, LSFLOOR * abs(p0[2]))  # float32: the slope cannot fall below ~1e-7 of its start
    var p1 = peval(mi, mf, res, f, gauss, p0[0] - p0[2] / p0[3]); it += 1
    if abs(p1[2]) < gtol:
        return p1[0]
    var dir: Float32 = 1 if p1[2] < 0 else -1
    var p2 = p0
    while p1[2] * dir <= -gtol and it < lsiter:
        p2 = p1
        p1 = peval(mi, mf, res, f, gauss, p1[0] - p1[2] / p1[3]); it += 1
        if abs(p1[2]) < gtol:
            return p1[0]
    if it >= lsiter:
        return p1[0]
    var p2next = p1
    var p1next = peval(mi, mf, res, f, gauss, p1[0] - p1[2] / p1[3]); it += 1
    var pmid = p0
    while it < lsiter:
        pmid = peval(mi, mf, res, f, gauss, 0.5 * (p1[0] + p2[0])); it += 1
        var bestcost: Float32 = 0
        var bestind = -1
        for i in range(3):
            var c = p1next if i == 0 else (p2next if i == 1 else pmid)
            if abs(c[2]) < gtol and (bestind == -1 or c[1] < bestcost):
                bestcost = c[1]
                bestind = i
        if bestind >= 0:
            return p1next[0] if bestind == 0 else (p2next[0] if bestind == 1 else pmid[0])
        var c0 = p1next; var c1 = p2next; var c2 = pmid
        var b1 = bracket(p1, c0, c1, c2)
        if b1:
            p1next = peval(mi, mf, res, f, gauss, p1[0] - p1[2] / p1[3]); it += 1
        var b2 = bracket(p2, c0, c1, c2)
        if b2:
            p2next = peval(mi, mf, res, f, gauss, p2[0] - p2[2] / p2[3]); it += 1
        if not b1 and not b2:
            return pmid[0]
    if p1[1] <= p2[1] and p1[1] < p0[1]:
        return p1[0]
    if p2[1] <= p1[1] and p2[1] < p0[1]:
        return p2[0]
    return 0


def start(mi: TI, mf: TF, res: TO, f: Int, src: Int) -> Float32:
    """Newton's starting point: qacc = rows src.., M qacc, jar on the valid rows, then PrimalUpdateConstraint's cost."""
    for v in range(NV):
        res[(OCQACC + v) * F + f] = get(res, f, src + v)
    mulM(mi, res, f, OMA, OCQACC)
    for ip in range(npairs(res, f)):
        var p = pick(res, f, ip)
        if p < 0:
            continue
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(NR):
                var row = RS * (sbase(mi, res, f, p) + sl) + k
                res[(OJAR + row) * F + f] = jdot(mi, res, f, OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch, nch, choff, OCQACC) - get(res, f, OAREF + row)
    comptime if NEQ > 0:
        for e in range(NEQ):
            var eh = CEQH + EQH * e
            var nch = Int(mi[eh + 4]); var choff = Int(mi[eh + 5]); var rbase = Int(mi[eh + 6]); var jb = OEQJ + Int(mi[eh + 7])
            for k in range(Int(mi[eh + 3])):
                res[(OEQJAR + rbase + k) * F + f] = jdot(mi, res, f, jb + k * nch, nch, choff, OCQACC, CEQC) - get(res, f, OEQA + rbase + k)
    return update(mi, mf, res, f)


@always_inline
def rowpair(mi: TI, res: TO, f: Int, r: Int) -> Int:
    """The exported pair a constraint row belongs to: its slot r // RS lies in [sbase, sbase + nslot) of exactly one pair."""
    for ip in range(Int(get(res, f, OPA))):
        var q = Int(get(res, f, OPA + 1 + ip))
        if r // RS >= sbase(mi, res, f, q) and r // RS < sbase(mi, res, f, q) + nslot(mi, res, f, q):
            return q
    return 0


@always_inline
def pcg32(mut state: UInt64) -> UInt32:
    """MuJoCo pcg32_next (engine_solver.c:240), inc = 1."""
    var old = state
    state = old * 6364136223846793005 + 1
    var xorshifted = UInt32(((old >> 18) ^ old) >> 27)
    var rot = UInt32(old >> 59)
    return (xorshifted >> rot) | (xorshifted << ((UInt32(0) - rot) & 31))


@always_inline
def eqof(mi: TI, k: Int) -> Int:
    """The equality-block entry holding block row k (entries' rows ascend)."""
    for e in range(NEQ):
        if k < Int(mi[CEQH + EQH * e + 6]) + Int(mi[CEQH + EQH * e + 3]):
            return e
    return NEQ - 1


def prow(mi: TI, res: TO, f: Int, R: Int) -> Tuple[Int, Int, Int, Int]:
    """PGS row R (contact rows below NEFC, then the equality block's): its J address, w address (L^-T J', laid out as J),
    chain length and chain address."""
    if R < NEFC:
        var p = rowpair(mi, res, f, R)
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        var jo = jbase(mi, res, f, p) + (RS * (R // RS - sbase(mi, res, f, p)) + R % RS) * nch
        return (OJ + jo, OWC + jo, nch, CCHAIN + choff)
    var k = R - NEFC
    var eh = CEQH + EQH * eqof(mi, k)
    var nch = Int(mi[eh + 4])
    var jo = Int(mi[eh + 7]) + (k - Int(mi[eh + 6])) * nch
    return (OEQJ + jo, OGWE + jo, nch, CEQC + Int(mi[eh + 5]))


@always_inline
def pget(res: TO, f: Int, R: Int, C: Int, E: Int) -> Float32:
    """Row R's value in the contact array C or the equality-block array E (OD/OEQD, OJAR/OEQJAR, ...)."""
    return get(res, f, C + R) if R < NEFC else get(res, f, E + R - NEFC)


@always_inline
def pwz(mi: TI, res: TO, f: Int, a: Int, b: Int) -> Float32:
    """w_a' D^-1 w_b over row a's chain (rows a and b of one block share it)."""
    var A = prow(mi, res, f, a); var B = prow(mi, res, f, b)
    var s: Float32 = 0
    for c in range(A[2]):
        s += get(res, f, A[1] + c) * get(res, f, OLDI + Int(mi[A[3] + c])) * get(res, f, B[1] + c)
    return s


@always_inline
def pres(mi: TI, res: TO, f: Int, R: Int) -> Float32:
    """MuJoCo residual: b + AR f for row R, matrix-free: jar(qacc_smooth) + w' D^-1 z + R f."""
    var A = prow(mi, res, f, R)
    var s = pget(res, f, R, OJAR, OEQJAR)
    for c in range(A[2]):
        var i = Int(mi[A[3] + c])
        s += get(res, f, A[1] + c) * get(res, f, OLDI + i) * get(res, f, OGZ + i)
    return s + get(res, f, OGF + R) / pget(res, f, R, OD, OEQD)


@always_inline
def pmove(mi: TI, res: TO, f: Int, R: Int, delta: Float32):
    """force[R] += delta, z += w_R delta."""
    if delta == 0:
        return
    res[(OGF + R) * F + f] = get(res, f, OGF + R) + delta
    var A = prow(mi, res, f, R)
    for c in range(A[2]):
        res[(OGZ + Int(mi[A[3] + c])) * F + f] += get(res, f, A[1] + c) * delta


@always_inline
def ufind(res: TO, f: Int, b: Int) -> Int:
    var r = b
    while Int(get(res, f, OGU + r)) != r:
        r = Int(get(res, f, OGU + r))
    return r


def pgs(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_solPGS (engine_solver.c:416) per island: dual projected Gauss-Seidel on AR = J M^-1 J' + R over contact rows
    and the equality block's (equality unclamped, friction loss within +-frictionloss, limit and pyramidal rows >= 0, an
    elliptic contact one block of 3: normal or ray update, then the friction pair by mju_QCQP2 on the raw friction), each sweep
    in the order MuJoCo's fixed-seed pcg32 Fisher-Yates shuffle gives, improvement from costChange, stop below opt.tolerance.
    Rows with D = 0 (an inactive equality, a frictionless contact's empty rows) do not exist for MuJoCo and are skipped.
    Islands (mj_island): a union-find over kinematic trees joining every row's nonzero Jacobian entries; each island has its
    own shuffle and stop test, rows in efc order (equality block, then contacts). Matrix-free as noslip: w = L^-T J' per row,
    A(i, j) = w_i' D^-1 w_j, z = L^-T J' f kept current. Warmstart (engine_forward.c:833): mj_constraintUpdate's forces at
    qacc_warmstart, zero when their dual cost f.b + f.AR.f / 2 is positive. Then mj_dualFinish; the result is left in
    Newton's representation (OST, OJAR = -f / D, OQFC = J' f, OCQACC) so noslip continues from it unchanged."""
    comptime NPR = NEFC + NEQR
    for v in range(NV):
        res[(OQSM + v) * F + f] = qsmooth(res, f, v)
        comptime if STAGE >= 7:
            res[(OQSM + v) * F + f] += get(res, f, OQACT + v)
        res[(OQACCS + v) * F + f] = get(res, f, OQSM + v)
    solveLD(mi, res, f, OQACCS, OLD, OLDI)
    res[ONCON * F + f] = get(res, f, OACT)
    res[ODIAG * F + f] = 0
    res[(ODIAG + 1) * F + f] = 0
    res[(ODIAG + 2) * F + f] = 0
    _ = start(mi, mf, res, f, OQACCS)  # OJAR, OEQJAR = J qacc_smooth - aref = efc_b
    for v in range(NV):
        res[(OGZ + v) * F + f] = 0
    for b in range(NB):
        res[(OGU + b) * F + f] = Float32(b)
    # the live rows (efc order: equality block, then contacts; an elliptic contact by its first row), w = L^-T J' of each
    var nb = 0
    for k in range(NEQR):
        if get(res, f, OEQD + k) != 0:
            res[(OGP + nb) * F + f] = Float32(NEFC + k)
            nb += 1
    for ip in range(Int(get(res, f, OPA))):  # contacts in pair order (MuJoCo's efc order; pooled slots are not in slot order)
        var p = Int(get(res, f, OPA + 1 + ip))
        for c in range(nslot(mi, res, f, p)):
            var sl = sbase(mi, res, f, p) + c
            if get(res, f, OVALID + sl) == 0:
                continue
            for k in range(1 if ELL else RS):
                if get(res, f, OD + RS * sl + k) != 0:
                    res[(OGP + nb) * F + f] = Float32(RS * sl + k)
                    nb += 1
    for bi in range(nb):
        var R0 = Int(get(res, f, OGP + bi))
        var first = -1  # the block's first tree: every other tree it touches joins it
        for kk in range(3 if ELL and R0 < NEFC else 1):
            var R = R0 + kk
            var A = prow(mi, res, f, R)
            ltsolve(mi, res, f, OW, A[0], A[2], A[3], 0)
            var a: Float32 = 0
            for c in range(A[2]):
                var i = Int(mi[A[3] + c])
                var w = get(res, f, OW + i)
                res[(A[1] + c) * F + f] = w
                a += w * w * get(res, f, OLDI + i)
                if get(res, f, A[0] + c) != 0:
                    var t = ufind(res, f, Int(mi[ROOT + Int(mi[DBODY + i])]))
                    if first < 0:
                        first = t
                    elif t != first:
                        res[(OGU + t) * F + f] = Float32(first)
            res[(OGA + R) * F + f] = 1 / (a + 1 / pget(res, f, R, OD, OEQD))
    # warmstart: mj_constraintUpdate at qacc_warmstart (zero at K = 0, the last step's qacc otherwise)
    for R in range(NPR):
        res[(OGF + R) * F + f] = 0
    for bi in range(nb):
        var R = Int(get(res, f, OGP + bi))
        var jw = SIMD[DType.float32, 4](0)
        for kk in range(3 if ELL and R < NEFC else 1):
            comptime if K > 0:
                var A = prow(mi, res, f, R + kk)
                for c in range(A[2]):
                    jw[kk] += get(res, f, A[0] + c) * get(res, f, OWARM + Int(mi[A[3] + c]))
            jw[kk] -= pget(res, f, R + kk, OAREF, OEQA)
        var fr = SIMD[DType.float32, 4](0)
        var D = pget(res, f, R, OD, OEQD)
        if R >= NEFC and R - NEFC < NEQB:
            fr[0] = -D * jw[0]
        elif R >= NEFC and R - NEFC < NEQL:
            var fl = floss(mf, R - NEFC)
            fr[0] = -D * jw[0] if fquad(mf, res, f, R - NEFC, jw[0]) else (fl if jw[0] < 0 else -fl)
        elif R < NEFC and ELL:  # the cone (update_cone)
            var p = rowpair(mi, res, f, R)
            var fc = cone_fric(mf, p)
            var mu = fc[0]
            var U = V3(jw[0] * mu, jw[1] * fc[1], jw[2] * fc[2], 0)
            var T = sqrt(U[1] * U[1] + U[2] * U[2])
            if U[0] >= mu * T or (T <= 0 and U[0] >= 0):
                pass
            elif mu * U[0] + T <= 0 or (T <= 0 and U[0] < 0):
                for kk in range(3):
                    fr[kk] = -get(res, f, OD + R + kk) * jw[kk]
            else:
                var Dm = D / (mu * mu * (1 + mu * mu))
                fr[0] = -Dm * (U[0] - mu * T) * mu
                fr[1] = -fr[0] / T * U[1] * fc[1]
                fr[2] = -fr[0] / T * U[2] * fc[2]
        else:
            fr[0] = -D * jw[0] if jw[0] < 0 else Float32(0)
        for kk in range(3 if ELL and R < NEFC else 1):
            pmove(mi, res, f, R + kk, fr[kk])
    var cost: Float32 = 0  # f.b + f.AR.f / 2, AR f = w' D^-1 z + R f
    for bi in range(nb):
        var R = Int(get(res, f, OGP + bi))
        for kk in range(3 if ELL and R < NEFC else 1):
            var fr = get(res, f, OGF + R + kk)
            if fr != 0:
                var b = pget(res, f, R + kk, OJAR, OEQJAR)
                cost += fr * b + 0.5 * fr * (pres(mi, res, f, R + kk) - b)
    if cost > 0:
        for R in range(NPR):
            res[(OGF + R) * F + f] = 0
        for v in range(NV):
            res[(OGZ + v) * F + f] = 0
    var scale = 1 / (mf[COPT + 4] * Float32(NV))
    var tol = mf[COPT + 1]
    var order = nb  # the island's visiting list, after the live rows in OGP
    for troot in range(1, NB):
        if mi[PARENT + troot] != 0 or ufind(res, f, troot) != troot:
            continue
        var ni = 0
        for bi in range(nb):
            var R = Int(get(res, f, OGP + bi))
            var A = prow(mi, res, f, R)
            var t = -1
            for c in range(A[2]):
                if get(res, f, A[0] + c) != 0:
                    t = ufind(res, f, Int(mi[ROOT + Int(mi[DBODY + Int(mi[A[3] + c])])]))
                    break
            if t == troot:
                res[(OGP + order + ni) * F + f] = Float32(R)
                ni += 1
        if ni == 0:
            continue
        var rng: UInt64 = 0
        _ = pcg32(rng)
        for _ in range(Int(mi[CITER])):
            res[ODIAG * F + f] += 1
            for i in range(ni - 1, 0, -1):  # Fisher-Yates over the persistent order
                var j = Int(pcg32(rng) % UInt32(i + 1))
                var t = get(res, f, OGP + order + i)
                res[(OGP + order + i) * F + f] = get(res, f, OGP + order + j)
                res[(OGP + order + j) * F + f] = t
            var improvement: Float32 = 0
            for bi in range(ni):
                var R = Int(get(res, f, OGP + order + bi))
                if ELL and R < NEFC:
                    improvement -= pgs_cone(mi, mf, res, f, R)
                    continue
                var rs = pres(mi, res, f, R)
                var old = get(res, f, OGF + R)
                var new = old - rs * get(res, f, OGA + R)
                if R >= NEFC and R - NEFC < NEQB:
                    pass
                elif R >= NEFC and R - NEFC < NEQL:
                    var fl = floss(mf, R - NEFC)
                    new = min(max(new, -fl), fl)
                else:
                    new = max(new, Float32(0))
                var delta = new - old
                var change = 0.5 * delta * delta / get(res, f, OGA + R) + delta * rs
                if change > 1e-10:
                    continue
                improvement -= change
                pmove(mi, res, f, R, delta)
            if improvement * scale < tol:
                break
    for v in range(NV):  # mj_dualFinish: qfrc_constraint = J' f, qacc = qacc_smooth + M^-1 qfrc_constraint
        res[(OQFC + v) * F + f] = 0
    for R in range(NPR):
        var fr = get(res, f, OGF + R)
        var D = pget(res, f, R, OD, OEQD)
        var bil = R >= NEFC and R - NEFC < NEQL  # equality and friction rows: any sign
        if R < NEFC:
            res[(OST + R) * F + f] = 1 if fr > 0 else 0
            res[(OJAR + R) * F + f] = -fr / D if fr > 0 else Float32(1)
            comptime if ELL:
                res[(OFRC + R) * F + f] = fr
        elif D != 0:
            res[(OEQJAR + R - NEFC) * F + f] = -fr / D if fr > 0 or bil else Float32(1)
        if fr == 0:
            continue
        var A = prow(mi, res, f, R)
        for c in range(A[2]):
            res[(OQFC + Int(mi[A[3] + c])) * F + f] += get(res, f, A[0] + c) * fr
    for v in range(NV):
        res[(OW + v) * F + f] = get(res, f, OQFC + v)
    solveLD(mi, res, f, OW, OLD, OLDI)
    for v in range(NV):
        res[(OCQACC + v) * F + f] = get(res, f, OQACCS + v) + get(res, f, OW + v)


def pgs_cone(mi: TI, mf: TF, res: TO, f: Int, R: Int) -> Float32:
    """One elliptic contact (rows R..R+2) of mj_solPGS: residual, the 3 x 3 block of AR, a normal update while the normal
    force is below mjMINVAL else a ray update along the current force, then the friction pair by mju_QCQP2 with the normal
    fixed (bc = res - Ac f_old + A(.,0) df0), costChange on the block (restored when positive). Returns the change."""
    var p = rowpair(mi, res, f, R)
    var fc = cone_fric(mf, p)
    var rs = V3(pres(mi, res, f, R), pres(mi, res, f, R + 1), pres(mi, res, f, R + 2), 0)
    var o = V3(get(res, f, OGF + R), get(res, f, OGF + R + 1), get(res, f, OGF + R + 2), 0)
    var A00 = pwz(mi, res, f, R, R) + 1 / get(res, f, OD + R)
    var A01 = pwz(mi, res, f, R, R + 1); var A02 = pwz(mi, res, f, R, R + 2)
    var A11 = pwz(mi, res, f, R + 1, R + 1) + 1 / get(res, f, OD + R + 1)
    var A12 = pwz(mi, res, f, R + 1, R + 2)
    var A22 = pwz(mi, res, f, R + 2, R + 2) + 1 / get(res, f, OD + R + 2)
    var n = o
    if o[0] < MINVAL:  # normal update
        n[0] = max(o[0] - rs[0] / A00, Float32(0))
        n[1] = 0; n[2] = 0
    else:  # ray update
        var v1 = V3(A00 * o[0] + A01 * o[1] + A02 * o[2], A01 * o[0] + A11 * o[1] + A12 * o[2], A02 * o[0] + A12 * o[1] + A22 * o[2], 0)
        var denom = (o * v1).reduce_add()
        if denom >= MINVAL:
            var x = -(o * rs).reduce_add() / denom
            if o[0] + x * o[0] < 0:
                x = -o[0] / o[0]
            n = o + x * o
    var b1 = rs[1] - (A11 * o[1] + A12 * o[2]) + A01 * (n[0] - o[0])
    var b2 = rs[2] - (A12 * o[1] + A22 * o[2]) + A02 * (n[0] - o[0])
    if n[0] < MINVAL:
        n[1] = 0; n[2] = 0
    else:
        var q = qcqp2(A11, A12, A22, b1, b2, fc[1], fc[2], n[0])
        n[1] = q[0]; n[2] = q[1]
    var d = n - o
    var Ad = V3(A00 * d[0] + A01 * d[1] + A02 * d[2], A01 * d[0] + A11 * d[1] + A12 * d[2], A02 * d[0] + A12 * d[1] + A22 * d[2], 0)
    var change = 0.5 * (d * Ad).reduce_add() + (d * rs).reduce_add()
    if change > 1e-10:
        return 0
    for kk in range(3):
        pmove(mi, res, f, R + kk, d[kk])
    return change


def newton(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_solPrimal (Newton), one island: start at qacc_smooth, line search, Hessian rebuilt when the
    active set changes (MuJoCo updates it incrementally; same matrix)."""
    for v in range(NV):
        res[(OQSM + v) * F + f] = qsmooth(res, f, v)
        comptime if STAGE >= 7:
            res[(OQSM + v) * F + f] += get(res, f, OQACT + v)
        res[(OQACCS + v) * F + f] = get(res, f, OQSM + v)
    solveLD(mi, res, f, OQACCS, OLD, OLDI)
    var scale: Float32 = 0
    for v in range(NV):
        scale += get(res, f, OM + Int(mi[MADR + v]) + Int(mi[MNNZ + v]) - 1)
    scale = 1 / scale
    res[ONCON * F + f] = get(res, f, OACT)
    var tol = mf[COPT + 1]
    res[ODIAG * F + f] = 0
    res[(ODIAG + 1) * F + f] = 0
    res[(ODIAG + 2) * F + f] = 0
    var cost = start(mi, mf, res, f, OQACCS)
    comptime if K > 0:  # mj warmstart: the last step's qacc unless qacc_smooth costs less
        var cw = start(mi, mf, res, f, OWARM)
        cost = start(mi, mf, res, f, OQACCS) if cw > cost else cw
    comptime if SOLVER != 1:
        hessian(mi, mf, res, f)
    comptime if STOP == 9:
        return
    comptime if SOLVER == 1:
        gradient_cg(mi, res, f, True)
    else:
        gradient(mi, res, f)
    comptime if STOP == 8:
        return
    for _ in range(Int(mi[CITER])):
        res[ODIAG * F + f] += 1
        var alpha = search(mi, mf, res, f, scale, tol * mf[COPT + 2], Int(mi[CITER + 1]))
        if alpha == 0:
            break
        for v in range(NV):
            res[(OCQACC + v) * F + f] += alpha * get(res, f, OSRCH + v)
            res[(OMA + v) * F + f] += alpha * get(res, f, OMV + v)
        var changed = False
        var eqflip = False
        for a in range(NR * nslots(res, f)):
            var sl = slot(res, f, a // NR)
            if sl < 0:
                continue
            var r = RS * sl + a % NR
            res[(OJAR + r) * F + f] += alpha * get(res, f, OJV + r)
            if (get(res, f, OJAR + r) < 0) != (get(res, f, OST + r) != 0):
                changed = True
        comptime if NEQ > 0:
            for r in range(NEQR):
                var was = get(res, f, OEQJAR + r) < 0
                var wasq = False
                comptime if NEQL > NEQB:
                    wasq = r >= NEQB and r < NEQL and fquad(mf, res, f, r, get(res, f, OEQJAR + r))
                res[(OEQJAR + r) * F + f] += alpha * get(res, f, OEQJV + r)
                if r >= NEQL and (get(res, f, OEQJAR + r) < 0) != was:  # a limit row entered or left H: lowrank covers contacts only
                    eqflip = True
                comptime if NEQL > NEQB:
                    if r >= NEQB and r < NEQL and fquad(mf, res, f, r, get(res, f, OEQJAR + r)) != wasq:  # a friction row entered or left H
                        eqflip = True
        var cheap = True
        comptime if ELL:  # the cone Hessian moves with jar: rebuilt every iteration, no rank-1 shortcut
            changed = True
            cheap = False
        if changed and not ELL and SOLVER != 1:
            comptime if WAVE:  # one sequential walk per animal on lane 0; its verdict shared so all lanes agree
                sync()
                if lane() == 0:
                    res[OPART * F + f] = 1 if lowrank(mi, res, f) else 0
                sync()
                cheap = get(res, f, OPART) != 0
                sync()
            else:
                cheap = lowrank(mi, res, f)
        var oldcost = cost
        cost = update(mi, mf, res, f)
        comptime if SOLVER != 1:
            if (changed and not cheap) or eqflip:
                hessian(mi, mf, res, f)
        comptime if SOLVER == 1:
            gradient_cg(mi, res, f, False)
        else:
            gradient(mi, res, f)
        var gn: Float32 = 0
        for v in range(NV):
            gn += get(res, f, OGRAD + v) * get(res, f, OGRAD + v)
        comptime if SOLVER != 1:  # CG runs on: MuJoCo's two tests end it after ~80 float64 iterations, in float32 they fire at ~37 (rounding in cost and
            # gradient) with qacc 2e-3 off; the line search ends it when alpha is 0, the iteration cap otherwise
            if scale * (oldcost - cost) < tol or scale * sqrt(gn) < tol:
                break
        # float32 stop: MuJoCo's 1e-8 tests sit below float32 noise, so a few animals ran all 100 iterations and
        # held their wave. Stop once a step moves qacc by under 1e-6 of its size (max norms): further steps only
        # move it within float32 rounding. (Stopping when a full step kept the active set was wrong: with stiff
        # contacts H is ill-conditioned, so one float32 Newton step is not exact.)
        comptime if SOLVER != 1:  # CG has no such stop: its steps are small by construction, MuJoCo's own tests end it
            var dq: Float32 = 0
            var qa: Float32 = 0
            for v in range(NV):
                dq = max(dq, abs(alpha * get(res, f, OSRCH + v)))
                qa = max(qa, abs(get(res, f, OCQACC + v)))
            if dq <= STEPTOL * qa:
                break
        comptime if SOLVER == 1:  # Polak-Ribiere: beta = grad . (Mgrad - Mgradold) / gradold . Mgradold, reset when negative
            var num: Float32 = 0
            var den: Float32 = 0
            for v in range(NV):
                num += get(res, f, OGRAD + v) * (get(res, f, OCGM + v) - get(res, f, OCGMO + v))
                den += get(res, f, OCGO + v) * get(res, f, OCGMO + v)
            var beta = max(num / max(den, Float32(1e-15)), Float32(0))
            if alpha < 0:  # the search was not a descent direction (float32 after convergence): restart, else the direction grows ~3.8x
                beta = 0  # per two steps with alternating alpha until it overflows (free_pair elliptic, state 43); MuJoCo's tests stop first
            for v in range(NV):
                res[(OSRCH + v) * F + f] = -get(res, f, OCGM + v) + beta * get(res, f, OSRCH + v)


@always_inline
def ltsolve(mi: TI, res: TO, f: Int, W: Int, jadr: Int, nch: Int, choff: Int, ch: Int = CCHAIN):
    """One constraint row's w = L^-T J' (mj_solveLD's first half on M's factor): nonzero on the row's chain only,
    which holds every M ancestor of its dofs, so rows W.. outside the chain are neither read nor written. ch: the chain
    table (contact chains CCHAIN, equality chains CEQC; 0 with an absolute choff)."""
    for c in range(nch):
        res[(W + Int(mi[ch + choff + c])) * F + f] = get(res, f, jadr + c)
    for c in range(nch - 1, -1, -1):
        var i = Int(mi[ch + choff + c])
        var start = Int(mi[MADR + i])
        var x_i = get(res, f, W + i)
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            res[(W + Int(mi[MCOL + adr])) * F + f] -= get(res, f, OLD + adr) * x_i


def qcqp2(a00: Float32, a01: Float32, a11: Float32, b0: Float32, b1: Float32, d0: Float32, d1: Float32, r: Float32) -> V3:
    """MuJoCo mju_QCQP2 (engine_util_solve.c) plus solveQCQP's projection (engine_solver.c:379): min 0.5 x'Ax + x'b over the
    ellipse (x0 / d0)^2 + (x1 / d1)^2 <= r^2, scaled to the unit-radius disc, Newton on the multiplier la of (A + la I) v = -b;
    an active solution is put exactly on the ellipse. Returns x0, x1."""
    var A0 = a00 * d0 * d0; var A1 = a01 * d0 * d1; var A2 = a11 * d1 * d1
    var B0 = b0 * d0; var B1 = b1 * d1
    var la: Float32 = 0
    var v0: Float32 = 0; var v1: Float32 = 0
    for _ in range(20):
        var det = (A0 + la) * (A2 + la) - A1 * A1
        if det < 1e-15:
            return V3(0)
        var P00 = (A2 + la) / det; var P11 = (A0 + la) / det; var P01 = -A1 / det
        v0 = -(P00 * B0 + P01 * B1); v1 = -(P01 * B0 + P11 * B1)
        var val = v0 * v0 + v1 * v1 - r * r
        if val < 1e-8 * r * r:
            break
        var deriv = -2 * (v0 * (P00 * v0 + P01 * v1) + v1 * (P01 * v0 + P11 * v1))
        var delta = -val / deriv
        if delta < 1e-15:
            break
        la += delta
    var x0 = v0 * d0; var x1 = v1 * d1
    if la != 0:
        var sq = x0 * x0 / (d0 * d0) + x1 * x1 / (d1 * d1)
        var sc = sqrt(r * r / max(Float32(1e-15), sq))
        x0 *= sc; x1 *= sc
    return V3(x0, x1, 0, 0)


def noslip_cone(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_solNoSlip for elliptic contacts (engine_solver.c:646), no dry friction: after Newton, per contact the 2 friction
    rows are re-solved with the normal force held, a QCQP over the friction ellipse (mu = f0, f1) against their 2x2 block
    A = J M^-1 J' and the residual of all other rows; a step that does not lower the cost is dropped. Same matrix-free
    scheme as noslip() (w = L^-T J', z = L^-T J' f kept current). Then mj_dualFinish as a change on Newton's result."""
    var nit = Int(mi[CITER + 2])
    if nit == 0:
        return
    var imp0: Float32 = 0
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(3):
                var r = RS * (sbase(mi, res, f, p) + sl) + k
                var fr = get(res, f, OFRC + r)
                imp0 += 0.5 * fr * fr / get(res, f, OD + r)
                res[(OFRN + r) * F + f] = fr
    for v in range(NV):
        res[(OZ + v) * F + f] = get(res, f, OQFC + v)
    for i in range(NV - 1, -1, -1):  # z = L^-T J' f
        var start = Int(mi[MADR + i])
        var x_i = get(res, f, OZ + i)
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            res[(OZ + Int(mi[MCOL + adr])) * F + f] -= get(res, f, OLD + adr) * x_i
    for ip in range(Int(get(res, f, OPA))):  # w, A and J qacc_smooth - aref of the friction rows: fixed during the sweeps
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            var slot = sbase(mi, res, f, p) + sl
            var jo = jbase(mi, res, f, p) + (RS * sl + 1) * nch
            ltsolve(mi, res, f, OW, OJ + jo, nch, choff)
            ltsolve(mi, res, f, OW + NV, OJ + jo + nch, nch, choff)
            var a00: Float32 = 0; var a01: Float32 = 0; var a11: Float32 = 0
            for c in range(nch):
                var i = Int(mi[CCHAIN + choff + c])
                var w0 = get(res, f, OW + i); var w1 = get(res, f, OW + NV + i); var di = get(res, f, OLDI + i)
                a00 += w0 * w0 * di; a01 += w0 * w1 * di; a11 += w1 * w1 * di
                res[(OWC + jo + c) * F + f] = w0
                res[(OWC + jo + nch + c) * F + f] = w1
            var ac = OAC + 5 * slot
            res[ac * F + f] = max(a00, Float32(1e-10))
            res[(ac + 1) * F + f] = a01
            res[(ac + 2) * F + f] = max(a11, Float32(1e-10))
            res[(ac + 3) * F + f] = jdot(mi, res, f, OJ + jo, nch, choff, OQACCS) - get(res, f, OAREF + RS * slot + 1)
            res[(ac + 4) * F + f] = jdot(mi, res, f, OJ + jo + nch, nch, choff, OQACCS) - get(res, f, OAREF + RS * slot + 2)
    var scale = 1 / (mf[COPT + 4] * Float32(NV))
    for it in range(nit):
        var improvement: Float32 = imp0 if it == 0 else 0
        for ip in range(Int(get(res, f, OPA))):
            var p = Int(get(res, f, OPA + 1 + ip))
            var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
            var fr = cone_fric(mf, p)
            for sl in range(nslot(mi, res, f, p)):
                if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                    continue
                var slot = sbase(mi, res, f, p) + sl
                var r = RS * slot
                var w = OWC + jbase(mi, res, f, p) + (RS * sl + 1) * nch
                var ac = OAC + 5 * slot
                var a00 = get(res, f, ac); var a01 = get(res, f, ac + 1); var a11 = get(res, f, ac + 2)
                var res0 = get(res, f, ac + 3); var res1 = get(res, f, ac + 4)
                for c in range(nch):
                    var i = Int(mi[CCHAIN + choff + c])
                    var zd = get(res, f, OZ + i) * get(res, f, OLDI + i)
                    res0 += get(res, f, w + c) * zd; res1 += get(res, f, w + nch + c) * zd
                var fnorm = get(res, f, OFRN + r)
                var o1 = get(res, f, OFRN + r + 1); var o2 = get(res, f, OFRN + r + 2)
                var n = V3(0)
                if fnorm >= 1e-15:
                    n = qcqp2(a00, a01, a11, res0 - (a00 * o1 + a01 * o2), res1 - (a01 * o1 + a11 * o2), fr[1], fr[2], fnorm)
                var d1 = n[0] - o1; var d2 = n[1] - o2
                var change = 0.5 * (a00 * d1 * d1 + 2 * a01 * d1 * d2 + a11 * d2 * d2) + d1 * res0 + d2 * res1
                if change > 1e-10:  # no improvement: keep the old forces
                    continue
                improvement -= change
                res[(OFRN + r + 1) * F + f] = n[0]
                res[(OFRN + r + 2) * F + f] = n[1]
                for c in range(nch):
                    var i = Int(mi[CCHAIN + choff + c])
                    res[(OZ + i) * F + f] += get(res, f, w + c) * d1 + get(res, f, w + nch + c) * d2
        if improvement * scale < mf[COPT + 3]:
            break
    for v in range(NV):  # mj_dualFinish as a change on Newton's result: qfrc_constraint += J' (f - f_newton), qacc += M^-1 of it
        res[(OW + v) * F + f] = 0
    for ip in range(Int(get(res, f, OPA))):
        var p = Int(get(res, f, OPA + 1 + ip))
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(1, 3):
                var row = RS * (sbase(mi, res, f, p) + sl) + k
                var df = get(res, f, OFRN + row) - get(res, f, OFRC + row)
                var jadr = OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch
                for c in range(nch):
                    res[(OW + Int(mi[CCHAIN + choff + c])) * F + f] += get(res, f, jadr + c) * df
    for v in range(NV):
        res[(OQFC + v) * F + f] += get(res, f, OW + v)
    solveLD(mi, res, f, OW, OLD, OLDI)
    for v in range(NV):
        res[(OCQACC + v) * F + f] += get(res, f, OW + v)


def noslip(mi: TI, mf: TF, res: TO, f: Int):
    """MuJoCo mj_solNoSlip for pyramidal contacts (no dry friction ported): after Newton, projected Gauss-Seidel over
    each contact's opposing pyramid-edge row pairs (their sum, what the normal sees, kept), noslip_iterations sweeps
    or until improvement / (meaninertia nv) < noslip_tolerance; then mj_dualFinish: qfrc_constraint = J' f, qacc =
    qacc_smooth + M^-1 qfrc_constraint. Matrix-free with M = L'DL: A(k, l) = J_k M^-1 J_l' = w_k' D^-1 w_l, w = L^-T J'
    on the row's chain, and the residual's J_k M^-1 J' f = w_k' D^-1 z with z = L^-T J' f kept current."""
    comptime if ELL:
        noslip_cone(mi, mf, res, f)
        return
    var nit = Int(mi[CITER + 2])
    if nit == 0:
        return
    var imp0: Float32 = 0
    for a in range(RS * nslots(res, f)):  # Newton's forces: -D jar on active rows
        var sl = slot(res, f, a // RS)
        if sl < 0:
            continue
        var r = RS * sl + a % RS
        var fr: Float32 = 0
        if get(res, f, OST + r) != 0:
            var D = get(res, f, OD + r)
            fr = -D * get(res, f, OJAR + r)
            imp0 += 0.5 * fr * fr / D
        res[(OFRC + r) * F + f] = fr
    comptime if NEQ > 0:  # solNoSlip's iteration-0 correction sums every row, equality rows included (a tendon limit row while jar < 0)
        for r in range(NEQR):
            if r >= NEQL and get(res, f, OEQJAR + r) >= 0:
                continue
            imp0 += 0.5 * get(res, f, OEQD + r) * get(res, f, OEQJAR + r) * get(res, f, OEQJAR + r)
    for v in range(NV):
        res[(OZ + v) * F + f] = get(res, f, OQFC + v)
    for i in range(NV - 1, -1, -1):  # z = L^-T J' f
        var start = Int(mi[MADR + i])
        var x_i = get(res, f, OZ + i)
        for adr in range(start, start + Int(mi[MNNZ + i]) - 1):
            res[(OZ + Int(mi[MCOL + adr])) * F + f] -= get(res, f, OLD + adr) * x_i
    for ip in range(npairs(res, f)):
        var p = pick(res, f, ip)  # w, A and J qacc_smooth - aref are fixed during the sweeps: once per step
        if p < 0:
            continue
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for t in range(2):
                var r0 = RS * (sbase(mi, res, f, p) + sl) + 2 * t
                var jo = jbase(mi, res, f, p) + (RS * sl + 2 * t) * nch
                ltsolve(mi, res, f, OW, OJ + jo, nch, choff)
                ltsolve(mi, res, f, OW + NV, OJ + jo + nch, nch, choff)
                var a00: Float32 = 0; var a01: Float32 = 0; var a11: Float32 = 0
                for c in range(nch):
                    var i = Int(mi[CCHAIN + choff + c])
                    var w0 = get(res, f, OW + i); var w1 = get(res, f, OW + NV + i); var di = get(res, f, OLDI + i)
                    a00 += w0 * w0 * di; a01 += w0 * w1 * di; a11 += w1 * w1 * di
                    res[(OWC + jo + c) * F + f] = w0
                    res[(OWC + jo + nch + c) * F + f] = w1
                var ac = OAC + 5 * (r0 // 2)
                res[ac * F + f] = max(a00, Float32(1e-10))
                res[(ac + 1) * F + f] = a01
                res[(ac + 2) * F + f] = max(a11, Float32(1e-10))
                res[(ac + 3) * F + f] = jdot(mi, res, f, OJ + jo, nch, choff, OQACCS) - get(res, f, OAREF + r0)
                res[(ac + 4) * F + f] = jdot(mi, res, f, OJ + jo + nch, nch, choff, OQACCS) - get(res, f, OAREF + r0 + 1)
    var scale = 1 / (mf[COPT + 4] * Float32(NV))
    for it in range(nit):
        var improvement: Float32 = imp0 if it == 0 else 0
        for ip in range(npairs(res, f)):
            var p = pick(res, f, ip)
            if p < 0:
                continue
            comptime if NCV > 0:  # a condim-1 pair has no friction rows
                if mi[CCND + p] == 1:
                    continue
            var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
            for sl in range(nslot(mi, res, f, p)):
                if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                    continue
                for t in range(2):
                    var r0 = RS * (sbase(mi, res, f, p) + sl) + 2 * t
                    var w = OWC + jbase(mi, res, f, p) + (RS * sl + 2 * t) * nch
                    var ac = OAC + 5 * (r0 // 2)
                    var a00 = get(res, f, ac); var a01 = get(res, f, ac + 1); var a11 = get(res, f, ac + 2)
                    var res0 = get(res, f, ac + 3); var res1 = get(res, f, ac + 4)
                    for c in range(nch):
                        var i = Int(mi[CCHAIN + choff + c])
                        var zd = get(res, f, OZ + i) * get(res, f, OLDI + i)
                        res0 += get(res, f, w + c) * zd; res1 += get(res, f, w + nch + c) * zd
                    var o0 = get(res, f, OFRC + r0); var o1 = get(res, f, OFRC + r0 + 1)
                    var mid = 0.5 * (o0 + o1)
                    var k1 = a00 + a11 - 2 * a01
                    var k0 = mid * (a00 - a11) + res0 - (a00 * o0 + a01 * o1) - res1 + (a01 * o0 + a11 * o1)
                    var n0 = mid; var n1 = mid
                    if k1 >= 1e-15:
                        var y = -k0 / k1
                        if y < -mid:
                            n0 = 0; n1 = 2 * mid
                        elif y > mid:
                            n0 = 2 * mid; n1 = 0
                        else:
                            n0 = mid + y; n1 = mid - y
                    var d0 = n0 - o0; var d1 = n1 - o1
                    # d1 = -d0 (the pair's sum is kept): the exact form; the general one cancels ulp(f) x |res| ~ 0.2 in float32
                    var change = 0.5 * k1 * d0 * d0 + d0 * (res0 - res1)
                    if change > 1e-10:  # no improvement: keep the old forces
                        continue
                    improvement -= change
                    res[(OFRC + r0) * F + f] = n0
                    res[(OFRC + r0 + 1) * F + f] = n1
                    for c in range(nch):
                        var i = Int(mi[CCHAIN + choff + c])
                        res[(OZ + i) * F + f] += get(res, f, w + c) * d0 + get(res, f, w + nch + c) * d1
        if improvement * scale < mf[COPT + 3]:
            break
    for v in range(NV):  # mj_dualFinish as a change on Newton's result: qfrc_constraint += J' (f - f_newton), qacc +=
        res[(OW + v) * F + f] = 0  # M^-1 of it (qacc_smooth + M^-1 J'f cancels ~140x in float32: qacc_smooth >> qacc)
    for ip in range(npairs(res, f)):
        var p = pick(res, f, ip)
        if p < 0:
            continue
        comptime if NCV > 0:  # a condim-1 pair has no friction rows
            if mi[CCND + p] == 1:
                continue
        var nch = Int(mi[CPH + 11 * p + 6]); var choff = Int(mi[CPH + 11 * p + 7])
        for sl in range(nslot(mi, res, f, p)):
            if get(res, f, OVALID + sbase(mi, res, f, p) + sl) == 0:
                continue
            for k in range(RS):
                var row = RS * (sbase(mi, res, f, p) + sl) + k
                var df = get(res, f, OFRC + row)
                if get(res, f, OST + row) != 0:
                    df += get(res, f, OD + row) * get(res, f, OJAR + row)
                var jadr = OJ + jbase(mi, res, f, p) + (RS * sl + k) * nch
                for c in range(nch):
                    res[(OW + Int(mi[CCHAIN + choff + c])) * F + f] += get(res, f, jadr + c) * df
    for v in range(NV):
        res[(OQFC + v) * F + f] += get(res, f, OW + v)
    solveLD(mi, res, f, OW, OLD, OLDI)
    for v in range(NV):
        res[(OCQACC + v) * F + f] += get(res, f, OW + v)

@always_inline
def add6(res: TO, f: Int, row: Int, v: S6):
    put6(res, f, row, get6(res, f, row) + v)


@always_inline
def rne_acc(mi: TI, qvel: TV, res: TO, f: Int, badr: Int, j: Int):
    """Dcacc of the current body for dof j (engine_derivative.c:596 forward pass): += (D cdof_dot) qvel_j on the ancestor columns, where
    D cdof_dot = crossMotion(Dcvel, cdof_j) (mjd_crossMotion_vel is linear in its first argument), then += cdof_dot_j on dof j's own
    column. Dcvel is read in the state MuJoCo's separate comVel pass had when it set Dcdofdot_j."""
    var jc = Int(mi[MNNZ + j]) - 1  # dof j's position in its body's B row: its ancestor dofs come first
    var cd = get6(res, f, OCD + 6 * j)
    var q = qvel[j * F + f]
    for c in range(jc):
        add6(res, f, ODCA + 6 * (badr + c), q * cross_motion(get6(res, f, ODCV + 6 * (badr + c)), cd))
    add6(res, f, ODCA + 6 * (badr + jc), get6(res, f, OCDD + 6 * j))


@always_inline
def rne_vel_col(mi: TI, res: TO, f: Int, badr: Int, j: Int):
    """Dcvel of the current body += cdof_j on dof j's own column."""
    add6(res, f, ODCV + 6 * (badr + Int(mi[MNNZ + j]) - 1), get6(res, f, OCD + 6 * j))


comptime B66 = SIMD[DType.float32, 64]  # 6 x 6 force-by-velocity block of mjd_passive_vel, B[i * 6 + j] = d force_i / d velocity_j (ang 0..2, lin 3..5)


@always_inline
def skew(x: V3) -> Tuple[V3, V3, V3]:
    """Rows of [x]x, so that [x]x y = cross(x, y): d cross(a, b) / d b = skew(a), d cross(a, b) / d a = -skew(b) (mjd_cross)."""
    return (V3(0, -x[2], x[1], 0), V3(x[2], 0, -x[0], 0), V3(-x[1], x[0], 0, 0))


@always_inline
def badd(mut B: B66, fq: Int, vq: Int, d: Tuple[V3, V3, V3], scale: Float32 = 1, colscale: V3 = V3(1, 1, 1, 0)):
    """MuJoCo addToQuadrant(B, D, col_quad = fq, row_quad = vq): B[3 fq + i][3 vq + j] += scale colscale[j] D[i][j]."""
    for j in range(3):
        B[(3 * fq) * 6 + 3 * vq + j] += scale * colscale[j] * d[0][j]
        B[(3 * fq + 1) * 6 + 3 * vq + j] += scale * colscale[j] * d[1][j]
        B[(3 * fq + 2) * 6 + 3 * vq + j] += scale * colscale[j] * d[2][j]


@always_inline
def on_chain(mi: TI, b: Int, v: Int) -> Bool:
    """Dof v moves body b: its body is b or an ancestor of b."""
    var bb = b
    while bb > 0:
        if Int(mi[DBODY + v]) == bb:
            return True
        bb = Int(mi[PARENT + bb])
    return False


@always_inline
def jcol(res: TO, f: Int, v: Int, dif: V3, qi: Q4) -> S6:
    """Column v of the 6D Jacobian of the frame (p, q) fixed to a body, in that frame (mj_jacGeom / mj_jacBodyCom rotated by the frame,
    as mjd_*Fluid build it): cdof about the root's subtree_com carried to p, rotated into the frame; local_wrench's rule on one dof."""
    var cd = get6(res, f, OCD + 6 * v)
    var ang = V3(cd[0], cd[1], cd[2], 0)
    var lin = V3(cd[3], cd[4], cd[5], 0) - cross(dif, ang)
    var la = rot(qi, ang)
    var ll = rot(qi, lin)
    return S6(la[0], la[1], la[2], ll[0], ll[1], ll[2], 0, 0)


def jtbj(mi: TI, res: TO, f: Int, b: Int, p: V3, q: Q4, B: B66, h: Float32, diag: Bool):
    """qDeriv += J' B J (MuJoCo addJTBJ) for the frame (p, q) on body b, folded into the factor as -h qDeriv: onto qLU over D's pattern
    (full implicit) or onto qH over M's lower pattern (implicitfast, mapD2M; B symmetrized by the caller). Entries outside the
    pattern are dropped, as MuJoCo's sparse add drops them; on a tree every pair of chain dofs is in D. diag: B is diagonal (the
    inertia boxes, 130 of the flight model's 132 fluid bodies), 6 products per pair instead of 36. The chain's J columns are built
    once into the fluid wrench rows (OFLW, free once fluid() has folded them into qfrc_passive): rebuilding them per pair and the
    36-term product cost 88 of 95 ms per launch on 64 flight animals; the row scan for the (k, p) entry most of the next 43 (DPAIR / MPAIR tables)."""
    var r = Int(mi[ROOT + b])
    var dif = p - V3(get(res, f, OSC + 3 * r), get(res, f, OSC + 3 * r + 1), get(res, f, OSC + 3 * r + 2), 0)
    var qi = Q4(q[0], -q[1], -q[2], -q[3])
    var chain = SIMD[DType.int32, 64](0)  # the body's dofs (mj_bodyChain), collected once
    var n = 0
    for v in range(NV):
        if n < 64 and n < NB and on_chain(mi, b, v):
            chain[n] = Int32(v)
            put6(res, f, OFLW + 6 * n, jcol(res, f, v, dif, qi))
            n += 1
    for ki in range(n):
        var k = Int(chain[ki])
        var jk = get6(res, f, OFLW + 6 * ki)
        for pi in range(n):
            var pp = Int(chain[pi])
            var jp = get6(res, f, OFLW + 6 * pi)
            var val: Float32 = 0
            if diag:
                for i in range(6):
                    val += jk[i] * B[i * 7] * jp[i]
            else:
                for i in range(6):
                    var u: Float32 = 0
                    for j in range(6):
                        u += B[i * 6 + j] * jp[j]
                    val += jk[i] * u
            comptime if IMPL:
                var e = Int(mi[DPAIR + k * NV + pp])
                if e >= 0:
                    res[(OQLU + e) * F + f] -= h * val
            else:
                var e = Int(mi[MPAIR + k * NV + pp])
                if e >= 0:
                    res[(OQH + e) * F + f] -= h * val


def ellipsoid_deriv(mi: TI, mf: TF, res: TO, f: Int, row: Int, h: Float32, sym: Bool):
    """MuJoCo mjd_ellipsoidFluid for one geom (engine_derivative.c:1618): B = d(local wrench) / d(local velocity) from mjd_magnus_force,
    mjd_kutta_lift, mjd_viscous_drag, mjd_viscous_torque and mjd_addedMassForces, the same prologue as ellipsoid()."""
    var rho = mf[FLF]
    var mu = mf[FLF + 1]
    var c = FLF + 5 + 15 * row
    var s = m3(mf, c)
    var coef = mf[c + 3]
    if coef == 0:
        return
    var blunt = mf[c + 4]
    var slender = mf[c + 5]
    var angd = mf[c + 6]
    var kutta = mf[c + 7]
    var magnus = mf[c + 8]
    var vm = m3(mf, c + 9)
    var vi = m3(mf, c + 12)
    var b = Int(mi[FLI + 2 * row])
    var g = Int(mi[FLI + 2 * row + 1])
    var bd = body(res, f, b)
    var q = mul(bd[1], m4(mf, GQUAT + 4 * g))
    var p = V3(get(res, f, OG + 3 * g), get(res, f, OG + 3 * g + 1), get(res, f, OG + 3 * g + 2), 0)
    var lv = local_wrench(mi, mf, res, f, b, p, q)
    var w = V3(lv[0], lv[1], lv[2], 0)
    var v = V3(lv[3], lv[4], lv[5], 0)
    var B = B66(0)
    var pi: Float32 = 3.14159265358979
    # mjd_magnus_force: force += mc cross(w, v)
    var mc = magnus * rho * 4.0 / 3.0 * pi * s[0] * s[1] * s[2]
    var sk = skew(v * mc)
    badd(B, 1, 0, (-sk[0], -sk[1], -sk[2]))
    badd(B, 1, 1, skew(w * mc))
    # mjd_kutta_lift
    var a = pow2(s[1] * s[2]); var bb = pow2(s[2] * s[0]); var cc = pow2(s[0] * s[1])
    var aa = a * a; var bb2 = bb * bb; var cc2 = cc * cc
    var x = v[0]; var y = v[1]; var z = v[2]
    var xx = x * x; var yy = y * y; var zz = z * z; var xy = x * y; var yz = y * z; var xz = x * z
    var pden = aa * xx + bb2 * yy + cc2 * zz
    var pnum = a * xx + bb * yy + cc * zz
    var norm2 = xx + yy + zz
    var dfden = pi * kutta * rho / max(Float32(1e-15), sqrt(pden * pnum * norm2))
    var dfx = yy * (a - bb) + zz * (a - cc)
    var dfy = xx * (bb - a) + zz * (bb - cc)
    var dfz = xx * (cc - a) + yy * (cc - bb)
    var pterm = pnum / max(Float32(1e-15), pden)
    var cterm = pnum / max(Float32(1e-15), norm2)
    var inner = V3(aa * pterm - a + cterm, bb2 * pterm - bb + cterm, cc2 * pterm - cc + cterm, 0)
    var d0 = V3(a - a, bb - a, cc - a, 0) * (2 * pnum) + inner * dfx
    var d1 = V3(a - bb, bb - bb, cc - bb, 0) * (2 * pnum) + inner * dfy
    var d2 = V3(a - cc, bb - cc, cc - cc, 0) * (2 * pnum) + inner * dfz
    d0 = d0 * V3(xx, xy, xz, 0); d1 = d1 * V3(xy, yy, yz, 0); d2 = d2 * V3(xz, yz, zz, 0)
    d0[0] -= dfx * pnum; d1[1] -= dfy * pnum; d2[2] -= dfz * pnum
    badd(B, 1, 1, (d0, d1, d2), dfden)
    # mjd_viscous_drag
    var dmax = max(max(s[0], s[1]), s[2])
    var dmin = min(min(s[0], s[1]), s[2])
    var dmid = s[0] + s[1] + s[2] - dmax - dmin
    var dsph = 2.0 / 3.0 * (s[0] + s[1] + s[2])
    var amax = pi * dmax * dmid
    var dacoef = pi / max(Float32(1e-15), sqrt(pnum * pnum * pnum * pden))
    var aproj = pi * sqrt(pden / max(Float32(1e-15), pnum))
    var nrm = sqrt(norm2)
    var invn = 1 / max(Float32(1e-15), nrm)
    var lincoef = mu * 3 * pi * dsph
    var quad = rho * (aproj * blunt + slender * (amax - aproj))
    var apc = rho * nrm * (blunt - slender)
    var dap = V3(apc * dacoef * a * x * (bb * yy * (a - bb) + cc * zz * (a - cc)), apc * dacoef * bb * y * (a * xx * (bb - a) + cc * zz * (bb - cc)),
                 apc * dacoef * cc * z * (a * xx * (cc - a) + bb * yy * (cc - bb)), 0)
    var e0 = V3(xx + norm2, xy, xz, 0) * (-quad * invn) - dap * x
    var e1 = V3(xy, yy + norm2, yz, 0) * (-quad * invn) - dap * y
    var e2 = V3(xz, yz, zz + norm2, 0) * (-quad * invn) - dap * z
    e0[0] -= lincoef; e1[1] -= lincoef; e2[2] -= lincoef
    badd(B, 1, 1, (e0, e1, e2))
    # mjd_viscous_torque
    var imax = 8.0 / 15.0 * pi * dmid * pow4(dmax)
    var momc = V3(0)
    for k in range(3):
        var ii = 8.0 / 15.0 * pi * s[k] * pow4(max(s[(k + 1) % 3], s[(k + 2) % 3]))
        momc[k] = angd * ii + slender * (imax - ii)
    var momv = w * momc
    var dens = rho / max(Float32(1e-15), sqrt((momv * momv).reduce_add()))
    var msq = w * momc * momc * (-dens)
    var diag = (w * msq).reduce_add() - mu * pi * dsph * dsph * dsph
    var t0 = msq * w[0]; var t1 = msq * w[1]; var t2 = msq * w[2]
    t0[0] += diag; t1[1] += diag; t2[2] += diag
    badd(B, 0, 0, (t0, t1, t2))
    # mjd_addedMassForces
    var vam = vi * w * rho
    var vlm = vm * v * rho
    sk = skew(w)
    badd(B, 0, 0, skew(vam))
    badd(B, 0, 0, (-sk[0], -sk[1], -sk[2]), 1, vi * rho)
    sk = skew(v)
    badd(B, 0, 1, skew(vlm))
    badd(B, 0, 1, (-sk[0], -sk[1], -sk[2]), 1, vm * rho)
    sk = skew(w)
    badd(B, 1, 0, skew(vlm))
    badd(B, 1, 1, (-sk[0], -sk[1], -sk[2]), 1, vm * rho)
    B = B * coef
    if sym:  # implicitfast: mju_symmetrize(B)
        var Bt = B66(0)
        for i in range(6):
            for j in range(6):
                Bt[i * 6 + j] = 0.5 * (B[i * 6 + j] + B[j * 6 + i])
        B = Bt
    jtbj(mi, res, f, b, p, q, B, h, False)


def box_deriv(mi: TI, mf: TF, res: TO, f: Int, b: Int, h: Float32, sym: Bool):
    """MuJoCo mjd_inertiaBoxFluid (engine_derivative.c:1724): diagonal B, the same prologue as inertia_box()."""
    var rho = mf[FLF]
    var mu = mf[FLF + 1]
    var m = mf[MASS + b]
    var I = m3(mf, INERTIA + 3 * b)
    var bx = V3(sqrt(max(Float32(1e-15), I[1] + I[2] - I[0]) / m * 6), sqrt(max(Float32(1e-15), I[0] + I[2] - I[1]) / m * 6),
                sqrt(max(Float32(1e-15), I[0] + I[1] - I[2]) / m * 6), 0)
    var bd = body(res, f, b)
    var p = V3(get(res, f, OXI + 3 * (b - 1)), get(res, f, OXI + 3 * (b - 1) + 1), get(res, f, OXI + 3 * (b - 1) + 2), 0)
    var q = mul(bd[1], m4(mf, IQUAT + 4 * b))
    var lv = local_wrench(mi, mf, res, f, b, p, q)
    var B = B66(0)
    var pi: Float32 = 3.14159265358979
    if mu > 0:
        var diam = (bx[0] + bx[1] + bx[2]) / 3
        for k in range(3):
            B[k * 7] = -pi * diam * diam * diam * mu
            B[(3 + k) * 7] = -3 * pi * diam * mu
    if rho > 0:
        B[0] -= rho * bx[0] * (pow4(bx[1]) + pow4(bx[2])) * 2 * abs(lv[0]) / 64
        B[7] -= rho * bx[1] * (pow4(bx[0]) + pow4(bx[2])) * 2 * abs(lv[1]) / 64
        B[14] -= rho * bx[2] * (pow4(bx[0]) + pow4(bx[1])) * 2 * abs(lv[2]) / 64
        B[21] -= rho * bx[1] * bx[2] * abs(lv[3])
        B[28] -= rho * bx[0] * bx[2] * abs(lv[4])
        B[35] -= rho * bx[0] * bx[1] * abs(lv[5])
    jtbj(mi, res, f, b, p, q, B, h, True)


def fluid_deriv(mi: TI, mf: TF, res: TO, f: Int, h: Float32, sym: Bool):
    """mjd_passive_vel's fluid terms (engine_derivative.c:1890): the ellipsoid rows, then the inertia-box bodies, into the implicit factor."""
    for r in range(NFLROW):
        ellipsoid_deriv(mi, mf, res, f, r, h, sym)
    for k in range(NFLBOX):
        box_deriv(mi, mf, res, f, Int(mi[FLI + 2 * NFLROW + k]), h, sym)


@always_inline
def ten_len(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int, t: Int) -> Float32:
    """Tendon t's length: the row spatial() wrote (OTLEN), else the fixed tendon's sum coef qpos."""
    comptime if SPATIAL:
        return get(res, f, OTLEN + t)
    return tendon(mi, mf, qpos, qvel, f, t)[0]


def ten_j(mi: TI, mf: TF, res: TO, f: Int, t: Int, v: Int) -> Float32:
    """Tendon t's moment on dof v: the dense row spatial() wrote (OTJ), else the fixed tendon's wrap coef on that dof (0 if none)."""
    comptime if SPATIAL:
        return get(res, f, OTJ + t * NV + v)
    var a = Int(mi[TNI + t])
    for w in range(a, a + Int(mi[TNI + NTEN + t])):
        if Int(mi[TNI + 2 * NTEN + w]) == v:
            return mf[TNF + w]
    return Float32(0)


def implicit_factor(mi: TI, mf: TF, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_implicitSkip, full implicit (engine_forward.c:1864, INTEGRATOR 2): qLU = M - h qDeriv in MuJoCo's dof-dof sparsity D
    (mju_gatherMasked of M through mapM2D, then mju_addToScl), factored in place by mju_factorLUSparse (engine_util_solve.c:841: reverse
    order, no fill-in on a tree). qDeriv = d(qfrc_actuator + qfrc_passive - qfrc_bias) / d qvel (mjd_smooth_vel, engine_derivative.c:1985):
    the dof dampers on the diagonal, the actuator velocity term as implicitfast's, and mjd_rne_vel (engine_derivative.c:596) for the bias.
    mjd_rne_vel keeps, per body, one 6-vector per B entry (the derivative w.r.t. that dof's velocity) of cvel (Dcvel), cacc (Dcacc) and
    cfrc_body (Dcfrcbody). Its matrix products are linear in the 6-vector, so each is written as the spatial operation it is:
    D(I cacc + cvel x* (I cvel)) x = I Dcacc + Dcvel x* (I cvel) + cvel x* (I Dcvel)."""
    var h = mf[TSTEP]
    for e in range(NDD):
        var m_ = Int(mi[M2D + e])
        res[(OQLU + e) * F + f] = get(res, f, OM + m_) if m_ >= 0 else Float32(0)
    for v in range(NV):
        res[(OQLU + Int(mi[DRA + v]) + Int(mi[DDG + v])) * F + f] += h * mf[DAMP + v]
    for t in range(NTEN):  # mjd_passive_vel tendon damping (engine_derivative.c:1953): qDeriv += J' (-damping) J, onto D's entries only (addJTBJSparse)
        var damp = mf[TNF + NWRAP + NTEN + t]
        if damp == 0:
            continue
        for k in range(NV):
            var jk = ten_j(mi, mf, res, f, t, k)
            if jk == 0:
                continue
            for pp in range(NV):
                var e = Int(mi[DPAIR + k * NV + pp])
                if e >= 0:
                    res[(OQLU + e) * F + f] += h * damp * jk * ten_j(mi, mf, res, f, t, pp)
    comptime if NFLUID:  # mjd_passive_vel fluid terms
        fluid_deriv(mi, mf, res, f, h, False)
    comptime if STAGE >= 7:  # as implicitfast: moment' bias_vel moment per joint actuator, skipped where forcerange clamps
        for u in range(NU):
            var a = CACT + 5 * u
            if mi[a] != 0 and mi[a] != 3:
                continue
            var af = CACF + 9 * u
            var force = get(res, f, OAF + u)
            if mi[a + 3] != 0 and (force <= mf[af + 6] or force >= mf[af + 7]):
                continue
            var dof = Int(mi[a + 1])
            if mi[a] == 3:  # fixed tendon: moment = gear ten_J, so qDeriv += gear^2 biasprm[2] J' J onto the DPAIR pattern (mjd_actuator_vel)
                comptime if NTEN > 0:
                    var g2b = mf[af + 8] * mf[af + 8] * mf[af + 3]
                    if g2b != 0:
                        for k in range(NV):
                            var jk = ten_j(mi, mf, res, f, dof, k)
                            if jk == 0:
                                continue
                            for pp in range(NV):
                                var e = Int(mi[DPAIR + k * NV + pp])
                                if e >= 0:
                                    res[(OQLU + e) * F + f] -= h * g2b * jk * ten_j(mi, mf, res, f, dof, pp)
                continue
            res[(OQLU + Int(mi[DRA + dof]) + Int(mi[DDG + dof])) * F + f] -= h * mf[af + 8] * mf[af + 8] * mf[af + 3]
    # forward pass over bodies: Dcvel, Dcacc, then Dcfrcbody = D(cinert cacc + cvel x (cinert cvel))
    for b in range(1, NB):
        var badr = Int(mi[BRA + b])
        var bnnz = Int(mi[BRN + b])
        var anc = Int(mi[BANC + b])
        var padr = Int(mi[BRA + Int(mi[PARENT + b])])
        for r in range(6 * bnnz):
            res[(ODCV + 6 * badr + r) * F + f] = 0
            res[(ODCA + 6 * badr + r) * F + f] = 0
        for r in range(6 * anc):  # copyFromParent: the ancestor dofs lead the parent's row too
            res[(ODCV + 6 * badr + r) * F + f] = get(res, f, ODCV + 6 * padr + r)
            res[(ODCA + 6 * badr + r) * F + f] = get(res, f, ODCA + 6 * padr + r)
        for jj in range(Int(mi[JNTNUM + b])):
            var jn = Int(mi[JNTADR + b]) + jj
            var t = Int(mi[JTYPE + jn])
            var j = Int(mi[DOFADR + jn])
            if t == 0:  # free: translations (Dcdofdot 0), then the rotations as a ball
                for k in range(3):
                    rne_vel_col(mi, res, f, badr, j + k)
                for k in range(3):
                    add6(res, f, ODCA + 6 * (badr + Int(mi[MNNZ + j + k]) - 1), get6(res, f, OCDD + 6 * (j + k)))
                j += 3
            if t <= 1:  # ball (or the rotations of a free joint): all three Dcdofdot see Dcvel before any of the three columns is added
                for k in range(3):
                    rne_acc(mi, qvel, res, f, badr, j + k)
                for k in range(3):
                    rne_vel_col(mi, res, f, badr, j + k)
            else:  # hinge, slide
                rne_acc(mi, qvel, res, f, badr, j)
                rne_vel_col(mi, res, f, badr, j)
        var cin = OCI + 10 * (b - 1)
        var cv = get6(res, f, OCV + 6 * (b - 1))
        var icv = mul_inert(res, f, cin, cv)
        for c in range(bnnz):
            var x = get6(res, f, ODCV + 6 * (badr + c))
            var y = get6(res, f, ODCA + 6 * (badr + c))
            put6(res, f, ODCF + 6 * (badr + c), mul_inert(res, f, cin, y) + cross_force(x, icv) + cross_force(cv, mul_inert(res, f, cin, x)))
    # backward pass: addToParent, the child's row is a subset of the parent's (both sorted by dof)
    for b in range(NB - 1, 0, -1):
        if Int(mi[BANC + b]) == 0:
            continue
        var p = Int(mi[PARENT + b])
        var i = 0
        var ip = 0
        var cn = Int(mi[BRN + b])
        var pn = Int(mi[BRN + p])
        while i < cn and ip < pn:
            var cc = mi[BCOL + Int(mi[BRA + b]) + i]
            var pc = mi[BCOL + Int(mi[BRA + p]) + ip]
            if cc == pc:
                add6(res, f, ODCF + 6 * (Int(mi[BRA + p]) + ip), get6(res, f, ODCF + 6 * (Int(mi[BRA + b]) + i)))
                i += 1
                ip += 1
            else:
                ip += 1
    # qDeriv -= D(cdof cfrc_body): row j, entry c of its body's B row; qLU = M - h qDeriv adds h times it
    for j in range(NV):
        var b = Int(mi[DBODY + j])
        var cd = get6(res, f, OCD + 6 * j)
        for c in range(Int(mi[BRN + b])):
            var fr = get6(res, f, ODCF + 6 * (Int(mi[BRA + b]) + c))
            res[(OQLU + Int(mi[DRA + j]) + c) * F + f] += h * (fr * cd).reduce_add()
    # mju_factorLUSparse
    for i in range(NV):
        res[(ORM + i) * F + f] = Float32(Int(mi[DRN + i]))
    for r in range(NV - 1, -1, -1):
        var ii = Int(mi[DRA + r]) + Int(get(res, f, ORM + r)) - 1
        res[(ORM + r) * F + f] -= 1
        var lu_ii = get(res, f, OQLU + ii)
        for c in range(r - 1, -1, -1):
            var ji = Int(mi[DRA + c]) + Int(get(res, f, ORM + c)) - 1
            if Int(mi[DCOL + ji]) == r:
                res[(ORM + c) * F + f] -= 1
                var lu_ji = get(res, f, OQLU + ji) / lu_ii
                res[(OQLU + ji) * F + f] = lu_ji
                var icnt = Int(mi[DRA + r])
                var jcnt = Int(mi[DRA + c])
                var jend = Int(mi[DRA + c]) + Int(get(res, f, ORM + c))
                while jcnt < jend:
                    var ci = mi[DCOL + icnt]
                    var cj = mi[DCOL + jcnt]
                    if ci == cj:
                        res[(OQLU + jcnt) * F + f] -= get(res, f, OQLU + icnt) * lu_ji
                        icnt += 1
                        jcnt += 1
                    else:
                        jcnt += 1  # only (j,k) non-zero (MuJoCo errors out when only (i,k) is)


def implicit_solve(mi: TI, res: TO, f: Int, X: Int):
    """MuJoCo mju_solveLUSparse (engine_util_solve.c:929): x <- (M - h qDeriv)^-1 x from the LU in rows OQLU.., in place."""
    for i in range(NV - 1, -1, -1):  # (U + I) x = b, unit diagonal
        var d1 = Int(mi[DDG + i]) + 1
        var adr = Int(mi[DRA + i]) + d1
        var dot: Float32 = 0
        for t in range(Int(mi[DRN + i]) - d1):
            dot += get(res, f, OQLU + adr + t) * get(res, f, X + Int(mi[DCOL + adr + t]))
        res[(X + i) * F + f] -= dot
    for i in range(NV):  # L x = b
        var d = Int(mi[DDG + i])
        var adr = Int(mi[DRA + i])
        var dot: Float32 = 0
        for t in range(d):
            dot += get(res, f, OQLU + adr + t) * get(res, f, X + Int(mi[DCOL + adr + t]))
        res[(X + i) * F + f] = (get(res, f, X + i) - dot) / get(res, f, OQLU + adr + d)


@always_inline
def conj(q: Q4) -> Q4:
    """MuJoCo mju_negQuat: the conjugate (not the negative)."""
    return Q4(q[0], -q[1], -q[2], -q[3])


def solve3(J: SIMD[DType.float32, 16], rhs: V3) -> V3:
    """MuJoCo mju_solve3 (engine_util_solve.c): Gauss-Jordan on a 3x3 system without pivoting; J row-major in lanes 0..8."""
    var M = SIMD[DType.float32, 16](0)
    for i in range(3):
        for j in range(3):
            M[4 * i + j] = J[3 * i + j]
        M[4 * i + 3] = rhs[i]
    for i in range(3):
        var pivot = M[4 * i + i]
        for j in range(i, 4):
            M[4 * i + j] = M[4 * i + j] / pivot
        for k in range(3):
            if k != i:
                var fac = M[4 * k + i]
                for j in range(i, 4):
                    M[4 * k + j] -= fac * M[4 * i + j]
    return V3(M[3], M[7], M[11], 0)


def midpoint_newton(inertia: V3, w: V3, tau: V3, h: Float32) -> V3:
    """MuJoCo midpointNewton (engine_forward.c:1625): w_mid of I (w_new - w) / h = tau - w_mid x (I w_mid), w_mid = (w + w_new) / 2, Newton
    with a halving line search from w_mid = w. Returned as delta = w_mid - w and iterated on delta (w_mid = w + delta in the cross term only),
    because the velocity change is what the engine reports (qacc = (new - old) / h) and two nearby float32 velocities cancel. MuJoCo stops at
    |f| < 1e-13 (1e-6 in single precision) x (1 + i2h |I w|), absolute units that mean nothing at fly scale (I ~ 1e-13): float32 stops at
    5e-7 of the residual's own terms, or when no step lowers |f|."""
    var i2h = 2 / h
    var dI = V3(inertia[2] - inertia[1], inertia[0] - inertia[2], inertia[1] - inertia[0], 0)
    var i2h_I = i2h * inertia
    var delta = V3(0)
    for _ in range(30):
        var wm = w + delta
        var cor = cross(wm, inertia * wm)
        var lin = i2h_I * delta
        var fv = lin + cor - tau
        var fnorm = sqrt((fv * fv).reduce_add())
        if fnorm <= 5e-7 * (sqrt((lin * lin).reduce_add()) + sqrt((tau * tau).reduce_add()) + sqrt((cor * cor).reduce_add())):
            break
        var J = SIMD[DType.float32, 16](0)
        J[0] = i2h_I[0]; J[1] = wm[2] * dI[0]; J[2] = wm[1] * dI[0]
        J[3] = wm[2] * dI[1]; J[4] = i2h_I[1]; J[5] = wm[0] * dI[1]
        J[6] = wm[1] * dI[2]; J[7] = wm[0] * dI[2]; J[8] = i2h_I[2]
        var dx = solve3(J, -fv)
        var step: Float32 = 1
        var taken = False
        for _ in range(20):
            var dt_ = delta + step * dx
            var wt = w + dt_
            var ft = i2h_I * dt_ + cross(wt, inertia * wt) - tau
            if sqrt((ft * ft).reduce_add()) < fnorm:
                delta = dt_
                taken = True
                break
            step *= 0.5
        if not taken:
            break
    return delta


def midpoint(mi: TI, mf: TF, res: TO, f: Int, k: Int, qvel_old: S6, qfrc: S6, h: Float32) -> S6:
    """MuJoCo mj_midpoint (engine_forward.c:1708) for candidate k: implicit midpoint of one standalone free body, returning the CHANGE of
    its velocity (qvel_new - qvel_old; linear in the world frame, angular in the body frame; an aligned body (ipos 0) gets no linear part,
    MuJoCo leaves that to the ordinary step). The change is formed without subtracting nearby velocities: w_new - w = 2 delta, and with
    x = vcom_mid - vcom the linear solve keeps MuJoCo's analytic form with b = F/m + g - w_mid x vcom, v_new - v = 2 (x - delta x r_com);
    the world-frame change rot(xquat_new, u + du) - rot(xquat, u) = rot(xquat, rot(q, u) - u) + rot(xquat_new, du), q the half-step
    rotation, whose rot(q, u) - u is written out (2 s c a x u + 2 s^2 a x (a x u)). qfrc is the external force without the bias
    (qfrc_smooth + qfrc_constraint + qfrc_bias)."""
    var b = Int(mi[CMID + 3 * k])
    var aligned = mi[CMID + 3 * k + 2] != 0
    var mass = mf[MASS + b]
    var inertia = m3(mf, INERTIA + 3 * b)
    var ipos = m3(mf, IPOS + 3 * b)
    var iquat = m4(mf, IQUAT + 4 * b)
    var xquat = body(res, f, b)[1]
    var iquat_neg = conj(iquat)
    var fo = V3(qfrc[0], qfrc[1], qfrc[2], 0)
    var w = rot(iquat_neg, V3(qvel_old[3], qvel_old[4], qvel_old[5], 0))
    var tau = rot(iquat_neg, V3(qfrc[3], qfrc[4], qfrc[5], 0))
    var rot_x2i = mul(iquat_neg, conj(xquat))
    var force = rot(rot_x2i, fo)
    var r_com = rot(iquat_neg, ipos)
    var tau_com = tau
    if not aligned:
        tau_com = tau - cross(r_com, force)
    var delta = midpoint_newton(inertia, w, tau_com, h)
    var w_mid = w + delta
    var dw_body = rot(iquat, 2 * delta)
    var out = S6(0)
    out[3] = dw_body[0]; out[4] = dw_body[1]; out[5] = dw_body[2]
    if aligned:
        return out
    var v = rot(rot_x2i, V3(qvel_old[0], qvel_old[1], qvel_old[2], 0))
    var i2h = 2 / h
    var bx = force / mass + rot(rot_x2i, m3(mf, GRAV)) - cross(w_mid, v + cross(w, r_com))
    var denom = i2h * i2h + (w_mid * w_mid).reduce_add()
    var x = (i2h * bx + ((w_mid * bx).reduce_add() / i2h) * w_mid - cross(w_mid, bx)) / denom
    var du = rot(iquat, 2 * (x - cross(delta, r_com)))
    var u = rot(iquat, v)
    var w_mid_body = rot(iquat, w_mid)
    var nrm = sqrt((w_mid_body * w_mid_body).reduce_add())
    var axis = w_mid_body / nrm if nrm >= 1e-15 else basis(0)
    var half = 0.5 * h * nrm
    var s = sin(half)
    var c = cos(half)
    var au = cross(axis, u)
    var dr = 2 * s * c * au + 2 * s * s * cross(axis, au)
    var xquat_new = mul(xquat, Q4(c, axis[0] * s, axis[1] * s, axis[2] * s))
    var dl = rot(xquat, dr) + rot(xquat_new, du)
    out[0] = dl[0]; out[1] = dl[1]; out[2] = dl[2]
    return out


def mid_eligible(mi: TI, res: TO, f: Int, b: Int) -> Bool:
    """midpoint_eligible (engine_forward.c:1519), the runtime part: no live contact involves a geom of the body (the static part, a
    standalone free body, is what model.py mid_block lists). Equalities and tendon limits are refused at export."""
    comptime if STAGE >= 6:
        for t in range(Int(get(res, f, OPA))):
            var p = Int(get(res, f, OPA + 1 + t))
            if Int(mi[GBODY + Int(mi[CPH + 11 * p])]) == b or Int(mi[GBODY + Int(mi[CPH + 11 * p + 1])]) == b:
                return False
    return True


@always_inline
def vmid(qvel: TV, res: TO, f: Int, d: Int, h: Float32) -> Float32:
    """The velocity mj_advance integrates positions with: qvel + h qacc, or the midpoint's for a midpoint dof (OQMV)."""
    comptime if NMID > 0:
        return get(res, f, OQMV + d)
    else:
        return qvel[d * F + f] + h * get(res, f, OQACC + d)


def euler(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int):
    """MuJoCo mj_Euler (or, INTEGRATOR 3, mj_implicitSkip implicitfast) with implicit dof damping (contact-free: qfrc = qfrc_passive - qfrc_bias): qH = M + h diag(damping),
    factor, solve qH qacc = qfrc (mj_solveLD), qvel += h qacc, mj_integratePos. qvel is read, not written back."""
    var h = mf[TSTEP]
    comptime if IMPL:
        implicit_factor(mi, mf, qvel, res, f)
    else:
        for c in range(NC):
            res[(OQH + c) * F + f] = get(res, f, OM + c)
        for v in range(NV):
            res[(OQH + Int(mi[MADR + v]) + Int(mi[MNNZ + v]) - 1) * F + f] += h * mf[DAMP + v]
        comptime if INTEGRATOR == 3:  # implicitfast: the tendon damping term of qDeriv gathered onto M's lower pattern (mapD2M), as mj_implicitSkip
            for t in range(NTEN):
                var damp = mf[TNF + NWRAP + NTEN + t]
                if damp == 0:
                    continue
                for k in range(NV):
                    var jk = ten_j(mi, mf, res, f, t, k)
                    if jk == 0:
                        continue
                    for pp in range(NV):
                        var e = Int(mi[MPAIR + k * NV + pp])
                        if e >= 0:
                            res[(OQH + e) * F + f] += h * damp * jk * ten_j(mi, mf, res, f, t, pp)
        comptime if INTEGRATOR == 3 and NFLUID:  # mjd_passive_vel fluid terms, B symmetrized (mj_implicitSkip implicitfast)
            fluid_deriv(mi, mf, res, f, h, True)
        comptime if INTEGRATOR == 3 and STAGE >= 7:
            # mj_implicitSkip (engine_forward.c:1864) implicitfast: qH = M - h qDeriv. The dof damping above is -qDeriv of mjd_passive_vel;
            # mjd_actuator_vel adds moment' bias_vel moment per joint actuator (diagonal: gear^2 biasprm[2]), skipped where forcerange clamps
            for u in range(NU):
                var a = CACT + 5 * u
                if mi[a] != 0 and mi[a] != 3:
                    continue
                var af = CACF + 9 * u
                var force = get(res, f, OAF + u)
                if mi[a + 3] != 0 and (force <= mf[af + 6] or force >= mf[af + 7]):
                    continue
                var dof = Int(mi[a + 1])
                if mi[a] == 3:  # fixed tendon: moment = gear ten_J, so qDeriv += gear^2 biasprm[2] J' J onto the MPAIR pattern (mjd_actuator_vel)
                    comptime if NTEN > 0:
                        var g2b = mf[af + 8] * mf[af + 8] * mf[af + 3]
                        if g2b != 0:
                            for k in range(NV):
                                var jk = ten_j(mi, mf, res, f, dof, k)
                                if jk == 0:
                                    continue
                                for pp in range(NV):
                                    var e = Int(mi[MPAIR + k * NV + pp])
                                    if e >= 0:
                                        res[(OQH + e) * F + f] -= h * g2b * jk * ten_j(mi, mf, res, f, dof, pp)
                    continue
                res[(OQH + Int(mi[MADR + dof]) + Int(mi[MNNZ + dof]) - 1) * F + f] -= h * mf[af + 8] * mf[af + 8] * mf[af + 3]
        factorI(mi, res, f, OQH, OQHI)
    for v in range(NV):
        res[(OQACC + v) * F + f] = qsmooth(res, f, v)
        comptime if STAGE >= 6:
            res[(OQACC + v) * F + f] += get(res, f, OQFC + v)
        comptime if STAGE >= 7:
            res[(OQACC + v) * F + f] += get(res, f, OQACT + v)
    comptime if IMPL:
        implicit_solve(mi, res, f, OQACC)
    else:
        solveLD(mi, res, f, OQACC, OQH, OQHI)
    comptime if NMID > 0:  # mj_implicitSkip: qvel_mid = qvel + h qacc, midpoint dofs overwritten by (new + old) / 2, then qacc of those dofs = (new - old) / h
        for v in range(NV):
            res[(OQMV + v) * F + f] = qvel[v * F + f] + h * get(res, f, OQACC + v)
        for k in range(NMID):
            var b = Int(mi[CMID + 3 * k])
            var adr = Int(mi[CMID + 3 * k + 1])
            if not mid_eligible(mi, res, f, b):
                continue
            var old = S6(qvel[adr * F + f], qvel[(adr + 1) * F + f], qvel[(adr + 2) * F + f], qvel[(adr + 3) * F + f], qvel[(adr + 4) * F + f], qvel[(adr + 5) * F + f], 0, 0)
            var qfrc = S6(0)
            for c in range(6):
                var t = get(res, f, OPAS + adr + c)
                comptime if STAGE >= 6:
                    t += get(res, f, OQFC + adr + c)
                comptime if STAGE >= 7:
                    t += get(res, f, OQACT + adr + c)
                qfrc[c] = t
            var dq = midpoint(mi, mf, res, f, k, old, qfrc, h)
            for c in range(3 if mi[CMID + 3 * k + 2] != 0 else 0, 6):
                res[(OQMV + adr + c) * F + f] = old[c] + 0.5 * dq[c]
                res[(OQACC + adr + c) * F + f] = dq[c] / h
                comptime if STAGE >= 6:  # MuJoCo overwrites d->qacc there too (the warm start took the solve's qacc before)
                    res[(OCQACC + adr + c) * F + f] = dq[c] / h
    for a in range(NQ):
        res[(OQPOS + a) * F + f] = qpos[a * F + f]
    for j in range(NJ):  # mj_integratePos with the new qvel
        var a = Int(mi[QADR + j])
        var da = Int(mi[DOFADR + j])
        var t = Int(mi[JTYPE + j])
        if t == 0:
            for c in range(3):
                res[(OQPOS + a + c) * F + f] += h * vmid(qvel, res, f, da + c, h)
            a += 3
            da += 3
        if t <= 1:  # mju_quatIntegrate: q = unit(q) * quat(axis, h |w|), w in the body frame
            var w = V3(0)
            for c in range(3):
                w[c] = vmid(qvel, res, f, da + c, h)
            var n = sqrt((w * w).reduce_add())
            var axis = w / n if n >= 1e-15 else basis(0)
            var half = 0.5 * h * n
            var s = sin(half)
            var q = unit(Q4(get(res, f, OQPOS + a), get(res, f, OQPOS + a + 1), get(res, f, OQPOS + a + 2), get(res, f, OQPOS + a + 3)))
            put(res, f, OQPOS + a, mul(q, Q4(cos(half), axis[0] * s, axis[1] * s, axis[2] * s)), 4)
        else:
            res[(OQPOS + a) * F + f] += h * vmid(qvel, res, f, da, h)


def rkintegrate(mi: TI, res: TO, f: Int, h: Float32):
    """MuJoCo mj_integratePos (engine_support.c) of X0 = ORKQ along the velocity in ORKW over h, into OQPOS: free and ball joints by
    mju_quatIntegrate (unit(q) * quat(axis, h |w|), w in the body frame), slide and hinge by q += h v."""
    for a in range(NQ):
        res[(OQPOS + a) * F + f] = get(res, f, ORKQ + a)
    for j in range(NJ):
        var a = Int(mi[QADR + j])
        var da = Int(mi[DOFADR + j])
        var t = Int(mi[JTYPE + j])
        if t == 0:
            for c in range(3):
                res[(OQPOS + a + c) * F + f] += h * get(res, f, ORKW + da + c)
            a += 3
            da += 3
        if t <= 1:
            var w = V3(get(res, f, ORKW + da), get(res, f, ORKW + da + 1), get(res, f, ORKW + da + 2), 0)
            var n = sqrt((w * w).reduce_add())
            var axis = w / n if n >= 1e-15 else basis(0)
            var half = 0.5 * h * n
            var s = sin(half)
            var q = unit(Q4(get(res, f, OQPOS + a), get(res, f, OQPOS + a + 1), get(res, f, OQPOS + a + 2), get(res, f, OQPOS + a + 3)))
            put(res, f, OQPOS + a, mul(q, Q4(cos(half), axis[0] * s, axis[1] * s, axis[2] * s)), 4)
        else:
            res[(OQPOS + a) * F + f] += h * get(res, f, ORKW + da)


def rkstage(mi: TI, mf: TF, qpos: TQ, qvel: TV, res: TO, f: Int, s: Int):
    """MuJoCo mj_RungeKutta N=4 (engine_forward.c:1290), one stage per call: the forward pass just ran at X_s (held in the
    qpos/qvel tensors), so F_s = its constrained qacc. Stages 0..2 write X_{s+1} = X0 '+' (A v_s, h) into the tensors
    (A = 1/2, 1/2, 1: qvel_{s+1} = v0 + h A F_s, qpos_{s+1} = integratePos(q0, A v_s, h)); stage 3 forms the B = 1/6, 1/3, 1/3, 1/6
    sums, puts them in OQACC (dX acc: qvel_next = v0 + h OQACC) and OQPOS (integratePos(q0, dX vel, h)) and restores X0 in
    the tensors, so smooth()'s write-back and K=0 re-launches see the original state."""
    var h = mf[TSTEP]
    sync()  # every lane is past its reads of qpos/qvel from the forward pass before they are overwritten
    comptime if STAGE >= 6:
        for v in range(NV):
            res[(ORKF + s * NV + v) * F + f] = get(res, f, OCQACC + v)
    else:  # contact-free: qacc_smooth = M^-1 (qfrc_passive - qfrc_bias)
        for v in range(NV):
            res[(OQACC + v) * F + f] = qsmooth(res, f, v)
        solveLD(mi, res, f, OQACC, OLD, OLDI)
        for v in range(NV):
            res[(ORKF + s * NV + v) * F + f] = get(res, f, OQACC + v)
    for v in range(NV):
        res[(ORKV + s * NV + v) * F + f] = qvel[v * F + f]
    if s == 0:
        for a in range(NQ):
            res[(ORKQ + a) * F + f] = qpos[a * F + f]
    if s < 3:
        var c: Float32 = 0.5 if s < 2 else 1.0
        for v in range(NV):
            res[(ORKW + v) * F + f] = c * get(res, f, ORKV + s * NV + v)
        rkintegrate(mi, res, f, h)
        for v in range(NV):
            qvel[v * F + f] = get(res, f, ORKV + v) + h * c * get(res, f, ORKF + s * NV + v)
        for a in range(NQ):
            qpos[a * F + f] = get(res, f, OQPOS + a)
    else:
        absolute(mi, qpos, res, f)  # the rows hold the last forward pass (MuJoCo's d after mj_step): world positions of X_3, whose origin is in the tensors now
        for v in range(NV):
            res[(OQACC + v) * F + f] = (get(res, f, ORKF + v) + 2 * get(res, f, ORKF + NV + v) + 2 * get(res, f, ORKF + 2 * NV + v) + get(res, f, ORKF + 3 * NV + v)) / 6
            res[(ORKW + v) * F + f] = (get(res, f, ORKV + v) + 2 * get(res, f, ORKV + NV + v) + 2 * get(res, f, ORKV + 2 * NV + v) + get(res, f, ORKV + 3 * NV + v)) / 6
        rkintegrate(mi, res, f, h)
        for v in range(NV):
            qvel[v * F + f] = get(res, f, ORKV + v)
        for a in range(NQ):
            qpos[a * F + f] = get(res, f, ORKQ + a)
        comptime if K > 0 and STAGE >= 6:  # mj_advance: qacc_warmstart = d->qacc, the last forward pass's
            for v in range(NV):
                res[(OWARM + v) * F + f] = get(res, f, OCQACC + v)
    sync()


def origin(mi: TI, qpos: TQ, f: Int, b: Int) -> V3:
    """Floating origin of body b's tree: the position of its root's free joint (each animal its own), else 0 (the
    world and trees fixed to it). The step computes every position relative to it, so differences of nearby points
    stay exact in float32 however far each animal is from the world origin or from the other animals (one shared
    origin put a spider 8 mm from the fly at a ~100x float32 input floor)."""
    var r = Int(mi[ROOT + b])
    var ja = Int(mi[JNTADR + r])
    if r > 0 and Int(mi[JNTNUM + r]) >= 1 and mi[JTYPE + ja] == 0:
        var a = Int(mi[QADR + ja])
        return V3(qpos[a * F + f], qpos[(a + 1) * F + f], qpos[(a + 2) * F + f], 0)
    return V3(0)


def absolute(mi: TI, qpos: TQ, res: TO, f: Int):
    """Back to world positions, once per step, in the rows a later reader takes as world: xpos, xipos, geom_xpos,
    xanchor, subtree_com (STAGE >= 2), each by its own tree's origin (world geoms are already world)."""
    for i in range(NB - 1):
        var o = origin(mi, qpos, f, i + 1)
        put(res, f, 3 * i, V3(get(res, f, 3 * i), get(res, f, 3 * i + 1), get(res, f, 3 * i + 2), 0) + o, 3)
        put(res, f, OXI + 3 * i, V3(get(res, f, OXI + 3 * i), get(res, f, OXI + 3 * i + 1), get(res, f, OXI + 3 * i + 2), 0) + o, 3)
    for g in range(NG):
        var o = origin(mi, qpos, f, Int(mi[GBODY + g]))
        put(res, f, OG + 3 * g, V3(get(res, f, OG + 3 * g), get(res, f, OG + 3 * g + 1), get(res, f, OG + 3 * g + 2), 0) + o, 3)
    for i in range(1, NB):
        var o = origin(mi, qpos, f, i)
        for j in range(Int(mi[JNTADR + i]), Int(mi[JNTADR + i]) + Int(mi[JNTNUM + i])):
            put(res, f, OA + 3 * j, V3(get(res, f, OA + 3 * j), get(res, f, OA + 3 * j + 1), get(res, f, OA + 3 * j + 2), 0) + o, 3)
    comptime if STAGE >= 2:
        for i in range(NB):
            var o = origin(mi, qpos, f, i)
            put(res, f, OSC + 3 * i, V3(get(res, f, OSC + 3 * i), get(res, f, OSC + 3 * i + 1), get(res, f, OSC + 3 * i + 2), 0) + o, 3)


def smooth(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO):
    var f = block_idx.x if WAVE else block_idx.x * block_dim.x + thread_idx.x
    if f >= F:
        return
    smooth_at(mi, mf, qpos, qvel, ctrl, res, f)


def smooth_at(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO, f: Int):
    """K steps of animal f: the kernel's body, also the CPU build's per-animal call."""
    for _ in range(max(K, 1)):
        comptime if SLEEP:  # asleep (mj_sleep, engine_sleep.c:563): nothing runs; mj_wake (:240) wakes it when the host wrote a velocity
            if get(res, f, OSLP) >= SLEPT:
                var moved = False
                for v in range(NV):
                    moved = moved or qvel[v * F + f] != 0
                if not moved:
                    continue
                res[OSLP * F + f] = 0
        comptime if INTEGRATOR == 1:  # RK4: four forward passes, X_1..3 held in qpos/qvel, X0 back after the last
            for rks in range(4):
                step(mi, mf, qpos, qvel, ctrl, res, f, rks)
        else:
            step(mi, mf, qpos, qvel, ctrl, res, f)
            absolute(mi, qpos, res, f)
        comptime if SLEEP:  # mj_sleep in mj_advance (engine_forward.c:1145), on the step's starting qvel, before the advance: the tree can
            # sleep when dof_length |qvel| < tolerance on every dof (treeCanSleep, isSmaller; tolerance 0: every qvel exactly 0); the
            # mjMINAWAKE-th quiet check in a row puts it to sleep (one animal = one tree = its island): qvel and qacc zero, no advance
            var quiet = True
            for v in range(NV):
                quiet = quiet and (mf[SLPF + 1 + v] * abs(qvel[v * F + f]) < mf[SLPF] if mf[SLPF] > 0 else qvel[v * F + f] == 0)
            var c = get(res, f, OSLP) + 1 if quiet else Float32(0)
            if c >= 10:
                c = SLEPT
                for v in range(NV):
                    qvel[v * F + f] = 0  # asleep
                    comptime if K > 0:
                        res[(OWARM + v) * F + f] = 0
            res[OSLP * F + f] = c
            if c >= SLEPT:
                continue
        comptime if K > 0:  # mj_Euler's write-back: qvel += h qacc, qpos = integrated qpos
            for v in range(NV):
                qvel[v * F + f] += mf[TSTEP] * get(res, f, OQACC + v)
            for a in range(NQ):
                qpos[a * F + f] = get(res, f, OQPOS + a)


def step(mi: TI, mf: TF, qpos: TQ, qvel: TV, ctrl: TU, res: TO, f: Int, rks: Int = 0):

    for i in range(1, NB):
        var ja = Int(mi[JNTADR + i])
        var jn = Int(mi[JNTNUM + i])
        var xpos: V3
        var xquat: Q4
        if jn == 1 and mi[JTYPE + ja] == 0:  # free joint: pose straight from qpos
            var a = Int(mi[QADR + ja])
            xpos = V3(0)  # its own tree's origin
            xquat = unit(Q4(qpos[(a + 3) * F + f], qpos[(a + 4) * F + f], qpos[(a + 5) * F + f], qpos[(a + 6) * F + f]))
            put(res, f, OA + 3 * ja, xpos, 3)
            put(res, f, OX + 3 * ja, m3(mf, JAXIS + 3 * ja), 3)
        else:
            var pid = Int(mi[PARENT + i])
            var p = body(res, f, pid)
            xpos = rot(p[1], m3(mf, BPOS + 3 * i)) + p[0] if pid else m3(mf, BPOS + 3 * i)  # a fixed tree's origin is 0
            xquat = mul(p[1], m4(mf, BQUAT + 4 * i)) if pid else m4(mf, BQUAT + 4 * i)
            comptime if NMOCAP > 0:  # mj_kinematics (engine_core_smooth.c:86): a mocap body takes its pose from mocap_pos/quat, a world child
                var mid = Int(mi[CMOC + i])
                if mid >= 0:
                    var mrow = NU + 7 * mid
                    xpos = V3(ctrl[mrow * F + f], ctrl[(mrow + 1) * F + f], ctrl[(mrow + 2) * F + f], 0)
                    xquat = Q4(ctrl[(mrow + 3) * F + f], ctrl[(mrow + 4) * F + f], ctrl[(mrow + 5) * F + f], ctrl[(mrow + 6) * F + f])
            for j in range(ja, ja + jn):
                var a = Int(mi[QADR + j])
                var t = mi[JTYPE + j]
                var xaxis = rot(xquat, m3(mf, JAXIS + 3 * j))
                var xanchor = rot(xquat, m3(mf, JPOS + 3 * j)) + xpos
                if t == 2:  # slide
                    xpos += xaxis * (qpos[a * F + f] - mf[QPOS0 + a])
                else:  # ball (1) or hinge (3)
                    var loc: Q4
                    if t == 1:
                        loc = unit(Q4(qpos[a * F + f], qpos[(a + 1) * F + f], qpos[(a + 2) * F + f], qpos[(a + 3) * F + f]))
                    else:
                        var h = 0.5 * (qpos[a * F + f] - mf[QPOS0 + a])
                        var s = sin(h)
                        loc = Q4(cos(h), mf[JAXIS + 3 * j] * s, mf[JAXIS + 3 * j + 1] * s, mf[JAXIS + 3 * j + 2] * s)
                    xquat = mul(xquat, loc)
                    xpos = xanchor - rot(xquat, m3(mf, JPOS + 3 * j))
                put(res, f, OA + 3 * j, xanchor, 3)
                put(res, f, OX + 3 * j, xaxis, 3)
        xquat = unit(xquat)
        put(res, f, 3 * (i - 1), xpos, 3)
        put(res, f, OXQ + 4 * (i - 1), xquat, 4)
        put(res, f, OXI + 3 * (i - 1), rot(xquat, m3(mf, IPOS + 3 * i)) + xpos, 3)
    for g in range(NG):
        var b = body(res, f, Int(mi[GBODY + g]))
        put(res, f, OG + 3 * g, rot(b[1], m3(mf, GPOS + 3 * g)) + b[0], 3)
    comptime if STOP == 1:
        return
    comptime if STAGE >= 2:
        # mj_comPos: subtree_com = sum(mass x xipos) over the subtree / subtree mass
        for i in range(NB):
            var xi = V3(0) if i == 0 else V3(get(res, f, OXI + 3 * (i - 1)), get(res, f, OXI + 3 * (i - 1) + 1), get(res, f, OXI + 3 * (i - 1) + 2), 0)
            put(res, f, OSC + 3 * i, xi * mf[MASS + i], 3)
        for i in range(NB - 1, 0, -1):
            var p = Int(mi[PARENT + i])
            var o = origin(mi, qpos, f, i) * mf[SUBMASS + i] if p == 0 else V3(0)  # a tree's sum into the world: world frame
            for c in range(3):
                res[(OSC + 3 * p + c) * F + f] += get(res, f, OSC + 3 * i + c) + o[c]
        for i in range(NB):
            var sm = mf[SUBMASS + i]
            for c in range(3):
                if sm < 1e-15:
                    res[(OSC + 3 * i + c) * F + f] = 0 if i == 0 else get(res, f, OXI + 3 * (i - 1) + c)
                else:
                    res[(OSC + 3 * i + c) * F + f] = get(res, f, OSC + 3 * i + c) / sm
        # cinert: body inertia about subtree_com of its root, in the world frame (mju_inertCom)
        for i in range(1, NB):
            var r = 3 * Int(mi[ROOT + i])
            var dif = V3(get(res, f, OXI + 3 * (i - 1)) - get(res, f, OSC + r), get(res, f, OXI + 3 * (i - 1) + 1) - get(res, f, OSC + r + 1),
                         get(res, f, OXI + 3 * (i - 1) + 2) - get(res, f, OSC + r + 2), 0)
            var b = body(res, f, i)
            var iq = mul(b[1], m4(mf, IQUAT + 4 * i))
            var c0 = col(iq, 0); var c1 = col(iq, 1); var c2 = col(iq, 2)
            var I = m3(mf, INERTIA + 3 * i)
            var mass = mf[MASS + i]
            # mat * diag(I) * mat': entry (a, b) = sum_k mat[a][k] I_k mat[b][k], column k of mat = c_k
            var o = OCI + 10 * (i - 1)
            for a in range(3):
                for bb in range(a, 3):
                    var v = c0[a] * I[0] * c0[bb] + c1[a] * I[1] * c1[bb] + c2[a] * I[2] * c2[bb]
                    var slot = a if a == bb else (3 if a == 0 and bb == 1 else (4 if a == 0 else 5))
                    res[(o + slot) * F + f] = v
            res[(o + 0) * F + f] += mass * (dif[1] * dif[1] + dif[2] * dif[2])
            res[(o + 1) * F + f] += mass * (dif[0] * dif[0] + dif[2] * dif[2])
            res[(o + 2) * F + f] += mass * (dif[0] * dif[0] + dif[1] * dif[1])
            res[(o + 3) * F + f] -= mass * dif[0] * dif[1]
            res[(o + 4) * F + f] -= mass * dif[0] * dif[2]
            res[(o + 5) * F + f] -= mass * dif[1] * dif[2]
            put(res, f, o + 6, dif * mass, 3)
            res[(o + 9) * F + f] = mass
        # cdof: motion dofs in the frame centred at subtree_com (mju_dofCom)
        for i in range(1, NB):
            var r = 3 * Int(mi[ROOT + i])
            var com = V3(get(res, f, OSC + r), get(res, f, OSC + r + 1), get(res, f, OSC + r + 2), 0)
            var bq = body(res, f, i)[1]
            for j in range(Int(mi[JNTADR + i]), Int(mi[JNTADR + i]) + Int(mi[JNTNUM + i])):
                var da = OCD + 6 * Int(mi[DOFADR + j])
                var t = mi[JTYPE + j]
                var anchor = V3(get(res, f, OA + 3 * j), get(res, f, OA + 3 * j + 1), get(res, f, OA + 3 * j + 2), 0)
                var axis = V3(get(res, f, OX + 3 * j), get(res, f, OX + 3 * j + 1), get(res, f, OX + 3 * j + 2), 0)
                var off = com - anchor
                if t == 0:  # free: 3 translations along world x, y, z, then 3 rotations as a ball
                    for k in range(3):
                        put(res, f, da + 6 * k, V3(0), 3)
                        put(res, f, da + 6 * k + 3, basis(k), 3)
                    da += 18
                if t <= 1:  # free or ball: rotation about each body axis
                    for k in range(3):
                        var ax = col(bq, k)
                        put(res, f, da + 6 * k, ax, 3)
                        put(res, f, da + 6 * k + 3, cross(ax, off), 3)
                elif t == 2:  # slide
                    put(res, f, da, V3(0), 3)
                    put(res, f, da + 3, axis, 3)
                else:  # hinge
                    put(res, f, da, axis, 3)
                    put(res, f, da + 3, cross(axis, off), 3)
        # mj_crb: crb = cinert, then accumulate each body into its parent, backward
        for c in range(10 * (NB - 1)):
            res[(OCRB + c) * F + f] = get(res, f, OCI + c)
        for i in range(NB - 1, 0, -1):
            var p = Int(mi[PARENT + i])
            if p > 0:
                for c in range(10):
                    res[(OCRB + 10 * (p - 1) + c) * F + f] += get(res, f, OCRB + 10 * (i - 1) + c)
        # M(i, j) for every dof i and its ancestors j, MuJoCo's sparse row layout
        for i in range(NV):
            var adr = OM + Int(mi[MADR + i]) + Int(mi[MNNZ + i]) - 1
            var c = OCRB + 10 * (Int(mi[DBODY + i]) - 1)
            var v = OCD + 6 * i
            var I = SIMD[DType.float32, 16](0)
            for k in range(10):
                I[k] = get(res, f, c + k)
            var d0 = get(res, f, v); var d1 = get(res, f, v + 1); var d2 = get(res, f, v + 2)
            var d3 = get(res, f, v + 3); var d4 = get(res, f, v + 4); var d5 = get(res, f, v + 5)
            var buf = SIMD[DType.float32, 8](  # mju_mulInertVec
                I[0] * d0 + I[3] * d1 + I[4] * d2 - I[8] * d4 + I[7] * d5,
                I[3] * d0 + I[1] * d1 + I[5] * d2 + I[8] * d3 - I[6] * d5,
                I[4] * d0 + I[5] * d1 + I[2] * d2 - I[7] * d3 + I[6] * d4,
                I[8] * d1 - I[7] * d2 + I[9] * d3,
                I[6] * d2 - I[8] * d0 + I[9] * d4,
                I[7] * d0 - I[6] * d1 + I[9] * d5, 0, 0)
            var j = i
            var first = True
            for _ in range(Int(mi[MNNZ + i])):  # a simple dof's row (dof_simplenum, a childless free body with ipos 0 and iquat identity) is its diagonal alone
                var w = OCD + 6 * j
                var dot: Float32 = 0
                for k in range(6):
                    dot += get(res, f, w + k) * buf[k]
                res[adr * F + f] = (mf[ARMATURE + i] if first else Float32(0)) + dot
                first = False
                adr -= 1
                j = Int(mi[DPARENT + j])
    comptime if STOP == 2:
        return
    comptime if STAGE >= 3:
        for c in range(NC):
            res[(OLD + c) * F + f] = get(res, f, OM + c)
        factorI(mi, res, f, OLD, OLDI)
    comptime if STOP == 3:
        return
    comptime if STAGE >= 4:
        velocity(mi, mf, qpos, qvel, res, f)
        comptime if NXF > 0:
            xfrc(mi, ctrl, res, f)
    comptime if STOP == 4:
        return
    comptime if STAGE >= 6:
        contacts(mi, mf, qpos, qvel, res, f)
        comptime if UNION:
            for w in range(OUPM, NROW):
                res[w * F + f] = 0
        var na = 0
        for sl in range(NSLOT):
            if get(res, f, OVALID + sl) != 0:
                res[(OACT + 1 + na) * F + f] = Float32(sl)
                na += 1
                comptime if UNION:
                    res[(OUSM + sl // 24) * F + f] += Float32(1 << (sl % 24))
        res[OACT * F + f] = Float32(na)
        var npa = 0  # pairs with a valid slot, ascending: every solver loop below visits these
        for p in range(NCP):
            var live = False
            for sl in range(nslot(mi, res, f, p)):
                live = live or get(res, f, OVALID + sbase(mi, res, f, p) + sl) != 0
            if live:
                res[(OPA + 1 + npa) * F + f] = Float32(p)
                npa += 1
                comptime if UNION:
                    res[(OUPM + p // 24) * F + f] += Float32(1 << (p % 24))
        res[OPA * F + f] = Float32(npa)
        comptime if NEQ > 0:
            equality(mi, mf, qpos, qvel, ctrl, res, f)
        comptime if STAGE >= 7:
            comptime if GAME:  # not started: every row zero (a dead animal has dead = 1, so it is never re-initialised)
                if get(res, f, OGAME) == 0 and get(res, f, OGAME + 1) == 0 and get(res, f, OGAME + 2) == 0 and get(res, f, OGAME + 3) == 0:
                    for r in range(3):
                        res[(OGAME + r) * F + f] = 1
            actuation(mi, mf, qpos, qvel, ctrl, res, f)
            comptime if GAME:  # physical work of this step: sum |qfrc_actuator x qvel| x dt (qvel before the step's update)
                var work: Float32 = 0
                comptime if NMUSCLE > 0:
                    for u in range(NU):
                        var a = CACT + 5 * u
                        if mi[a] != 4:
                            continue
                        var m = Int(mi[a + 4])
                        var p = CMUSF + 11 * m
                        var lv = tendon(mi, mf, qpos, qvel, f, Int(mi[a + 1]))
                        comptime if SPATIAL:
                            lv = (get(res, f, OTLEN + Int(mi[a + 1])), get(res, f, OTVEL + Int(mi[a + 1])))
                        var stretch = lv[0] - mf[p + 6]
                        stretch = stretch if stretch > 0.0 else 0.0
                        var active = get(res, f, OAF + u) - mf[p + 5] * stretch * stretch
                        work += abs(active * lv[1]) * mf[TSTEP]
                else:
                    for v in range(NV):
                        work += abs(get(res, f, OQACT + v) * qvel[v * F + f]) * mf[TSTEP]
                var next = game.advance_state(game.GameState(get(res, f, OGAME), get(res, f, OGAME + 1), get(res, f, OGAME + 2), get(res, f, OGAME + 3)),
                                              work, False, False, False, mf[GAMEF], mf[GAMEF + 1], 0, mf[GAMEF + 2], 0, 0)
                res[OGAME * F + f] = next.energy
                res[(OGAME + 1) * F + f] = next.health
                res[(OGAME + 2) * F + f] = next.stamina
                res[(OGAME + 3) * F + f] = next.dead
        comptime if STOP == 5:
            return
        comptime if SOLVER == 0:
            pgs(mi, mf, res, f)
        else:
            newton(mi, mf, res, f)
        comptime if STOP == 8 or STOP == 9:
            return
        noslip(mi, mf, res, f)
        comptime if K > 0 and INTEGRATOR != 1:  # RK4 saves it after its last forward pass (rkstage)
            for v in range(NV):
                res[(OWARM + v) * F + f] = get(res, f, OCQACC + v)
        comptime if STOP == 6:
            return
    comptime if STAGE >= 5:
        comptime if INTEGRATOR == 1:
            rkstage(mi, mf, qpos, qvel, res, f, rks)
        else:
            euler(mi, mf, qpos, qvel, res, f)

def load(path: String, size: Int) raises -> List[UInt8]:
    var raw = open(path, "r").read_bytes()
    if len(raw) != size * 4:
        raise Error(path + ": expected " + String(size) + " words, got " + String(len(raw) // 4))
    return raw^


def host[T: DType, L: TensorLayout](mut a: List[UInt8], layout: L) -> TileTensor[T, L, MutAnyOrigin]:
    return TileTensor[T, L, MutAnyOrigin](ptr=a.unsafe_ptr().unsafe_bitcast[Scalar[T]]().unsafe_origin_cast[MutAnyOrigin](), layout=layout)


def main_cpu(dir: String) raises:
    """-DCPU=1: main's run on the host (same files in, same smooth.bin / state.bin out, same timing line)."""
    var mi = load(dir + "mi.bin", NI)
    var mf = load(dir + "mf.bin", NFLT)
    var qp = load(dir + "qpos.bin", NQ * F)
    var qv = load(dir + "qvel.bin", NV * F)
    var uv = load(dir + "ctrl.bin", NUC * F) if NUC > 0 else List[UInt8](length=4 * F, fill=0)
    var ob = List[UInt8](length=4 * NROW * F, fill=0)
    var ti = host[DType.int32](mi, LI); var tf = host[DType.float32](mf, LF); var tq = host[DType.float32](qp, LQ)
    var tv = host[DType.float32](qv, LV); var tu = host[DType.float32](uv, LU); var to = host[DType.float32](ob, LO)
    for f in range(F):
        smooth_at(ti, tf, tq, tv, tu, to, f)
    comptime if NMUSCLE > 0:
        with open(dir + "muscle-first.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=ob.unsafe_ptr(), length=NOUTX * F * 4))
    comptime if K > 0:
        with open(dir + "state.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=qp.unsafe_ptr(), length=NQ * F * 4))
            file.write_bytes(Span(unsafe_ptr=qv.unsafe_ptr(), length=NV * F * 4))
        comptime if 10 % K == 0:
            for _ in range(10 // K - 1):
                for f in range(F):
                    smooth_at(ti, tf, tq, tv, tu, to, f)
            with open(dir + "state10.bin", "w") as file:
                file.write_bytes(Span(unsafe_ptr=qp.unsafe_ptr(), length=NQ * F * 4))
                file.write_bytes(Span(unsafe_ptr=qv.unsafe_ptr(), length=NV * F * 4))
    var t0 = perf_counter_ns()
    for _ in range(100):
        for f in range(F):
            smooth_at(ti, tf, tq, tv, tu, to, f)
    var us = Float64(perf_counter_ns() - t0) / 100 / 1000
    with open(dir + "smooth.bin", "w") as file:
        file.write_bytes(Span(unsafe_ptr=ob.unsafe_ptr(), length=NOUTX * F * 4))
    comptime if DUMP and STAGE >= 6:  # tools/union_check.py: every row, and where the lists and masks sit
        with open(dir + "dump.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=ob.unsafe_ptr(), length=NROW * F * 4))
        with open(dir + "union.txt", "w") as file:
            file.write(String(NROW, " ", F, " ", OACT, " ", OPA, " ", OUPM, " ", OUSM, " ", NSLOT, " ", NCP, " ", Int(UNION), "\n"))
    print(F, "animals, stage", STAGE, ", K", max(K, 1), ":", us, "us per launch,", Float64(F * max(K, 1)) / us * 1e6, "animal-steps/s (CPU)")


def main() raises:
    var dir = String(argv()[1])
    comptime if get_defined_int["LAYOUT", 0]() == 1:  # byte audit: float32 rows per animal of each region of the o buffer, then the inputs
        print("rows per animal NROW", NROW, "=", NROW * 4, "B; NI", NI, "NFL", NFL, "(shared); NQ", NQ, "NV", NV, "NUC", NUC)
        print("region outputs", NOUTX); print("region body_scratch", OVALID - NOUTX); print("region contact_rows", OJ - OVALID)
        print("region jacobian", OQSM - OJ); print("region solver_vec", OHN - OQSM); print("region hessian", ODIAG - OHN)
        print("region noslip_act", OPP + 2 - ODIAG if STAGE >= 6 else 0); print("region tail", NROW - NROW1)
        return
    comptime if CPU:
        main_cpu(dir)
        return
    # no MAX pool: every buffer is made once, so VRAM is exactly the buffers plus the HIP runtime; a caller wins
    _ = external_call["setenv", Int32]("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT".unsafe_ptr(), "0".unsafe_ptr(), Int32(0))
    var ctx = DeviceContext(0)
    var mi_b = ctx.enqueue_create_buffer[DType.int32](NI)
    var mf_b = ctx.enqueue_create_buffer[DType.float32](NFLT)
    var q_b = ctx.enqueue_create_buffer[DType.float32](NQ * F)
    var v_b = ctx.enqueue_create_buffer[DType.float32](NV * F)
    var u_b = ctx.enqueue_create_buffer[DType.float32](max(NUC, 1) * F)
    var o_b = ctx.enqueue_create_buffer[DType.float32](NROW * F)
    o_b.enqueue_fill(0)  # warm start reads OWARM before the first step writes it
    var mi = load(dir + "mi.bin", NI)
    var mf = load(dir + "mf.bin", NFLT)
    var qp = load(dir + "qpos.bin", NQ * F)
    var qv = load(dir + "qvel.bin", NV * F)
    var uv = load(dir + "ctrl.bin", NUC * F) if NUC > 0 else List[UInt8](length=4 * F, fill=0)
    ctx.enqueue_copy(mi_b, mi.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(mf_b, mf.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(q_b, qp.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(v_b, qv.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.enqueue_copy(u_b, uv.unsafe_ptr().unsafe_bitcast[Float32]())
    var kernel = ctx.compile_function[smooth]()
    ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=GRID, block_dim=BLOCK)
    ctx.synchronize()
    comptime if NMUSCLE > 0:
        var first = List[Float32](length=NOUTX * F, fill=0)
        ctx.enqueue_copy(first.unsafe_ptr(), o_b)
        ctx.synchronize()
        with open(dir + "muscle-first.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=first.unsafe_ptr().unsafe_bitcast[UInt8](), length=NOUTX * F * 4))
    comptime if K > 0:
        var state = List[Float32](length=(NQ + NV) * F, fill=0)
        ctx.enqueue_copy(state.unsafe_ptr(), q_b)
        ctx.enqueue_copy(state.unsafe_ptr().unsafe_offset(NQ * F), v_b)
        ctx.synchronize()
        with open(dir + "state.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=state.unsafe_ptr().unsafe_bitcast[UInt8](), length=(NQ + NV) * F * 4))
        comptime if 10 % K == 0:  # state after 10 steps (10 / K launches): K=1 and K=10 must give the same bytes
            for _ in range(10 // K - 1):
                ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=GRID, block_dim=BLOCK)
            ctx.enqueue_copy(state.unsafe_ptr(), q_b)
            ctx.enqueue_copy(state.unsafe_ptr().unsafe_offset(NQ * F), v_b)
            ctx.synchronize()
            with open(dir + "state10.bin", "w") as file:
                file.write_bytes(Span(unsafe_ptr=state.unsafe_ptr().unsafe_bitcast[UInt8](), length=(NQ + NV) * F * 4))
    var t0 = perf_counter_ns()
    for _ in range(100):
        ctx.enqueue_function(kernel, TileTensor(mi_b, LI), TileTensor(mf_b, LF), TileTensor(q_b, LQ), TileTensor(v_b, LV), TileTensor(u_b, LU), TileTensor(o_b, LO), grid_dim=GRID, block_dim=BLOCK)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 100 / 1000
    var host = List[Float32](length=NROW * F, fill=0)
    ctx.enqueue_copy(host.unsafe_ptr(), o_b)
    ctx.synchronize()
    with open(dir + "smooth.bin", "w") as file:
        file.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=NOUTX * F * 4))
    comptime if DUMP and STAGE >= 6:
        with open(dir + "dump.bin", "w") as file:
            file.write_bytes(Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=NROW * F * 4))
        with open(dir + "dump.txt", "w") as file:
            file.write(String(NROW, " ", F, " ", OVALID, " ", OD, " ", OAREF, " ", OJ, " ", NSLOT, " ", CPH, " ", NCP, " ", CKIND if NCV > 0 else -1, " ", OSD, "\n"))
    print(F, "animals, stage", STAGE, ", K", max(K, 1), ":", us, "us per launch,", Float64(F * max(K, 1)) / us * 1e6, "animal-steps/s")
    comptime if STAGE >= 6:
        var it: Float64 = 0; var ev: Float64 = 0; var hb: Float64 = 0; var slots: Float64 = 0; var itmax: Float32 = 0
        for a in range(F):
            it += Float64(host[ODIAG * F + a]); ev += Float64(host[(ODIAG + 1) * F + a]); hb += Float64(host[(ODIAG + 2) * F + a]); slots += Float64(host[ONCON * F + a]); itmax = max(itmax, host[ODIAG * F + a])
        print("Newton iterations per animal", it / Float64(F), "max", itmax, "; line-search evaluations", ev / Float64(F))
        print("Hessian rebuilds per animal", hb / Float64(F), "; active slots", slots / Float64(F))
