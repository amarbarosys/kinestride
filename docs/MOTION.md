# Motion: any mocap skeleton in, one body, tracked physics out

Any BVH skeleton (CMU, Mixamo-style, any units) is retargeted onto one 21-body, 70 kg character, cleaned of isolated
spikes, its contact runs planted, and Kinestride tracks it with joint servos and a root assist, 64 animals at a time,
matching MuJoCo's own run of the same loop to 1.2e-7 m mean body position over 800 frames (`check.sh track`). Ideas observed in Godot 4.7.2
and rebuilt clean-room: `provenance/motion-godot.md`. Plan: `plans/MOTION_DATASET.md`.

```
bvh mjcf OUT.xml                 the one character   check.sh bvh    (CPU)
bvh clip CLIP.bvh OUT            OUT.qref, OUT.mclip check.sh bvh    (CPU)
track DIR STEM                   STEM.ks.mclip       check.sh track  (GPU, under gpu-wait)
motion_metrics REF SIM [OUT]     five numbers        check.sh motion (CPU)
```

Test clips are not in git: `tools/assets.sh` copies CMU 02_05, 10_01, 55_22, 60_01 and the CMU index into
`.work/assets/cmu/` (CMU: "may be copied, modified, or redistributed without permission").

## One body for every clip, because the skeleton is retargeted, not copied

```
BVH -> parse (MuJoCo frame, local quaternions) -> repair spikes -> retarget to the profile -> ground
    -> contacts -> plant limbs -> hinge angles (frozen orders) -> .qref + reference .mclip
```

- **Profile** (comptime table in `bvh.mojo`): Hips, Spine, Chest, Neck, Head, and per side Shoulder, UpperArm, LowerArm,
  Hand, UpperLeg, LowerLeg, Foot, Toes. Per body: source-name candidates (`mixamorig:` stripped, first found wins),
  parent, offset, bone vector, capsule radius, hinge axis order (frozen from the four CMU clips: worst middle angle 70
  deg, so no gimbal lock).
- **Retarget**: each body takes its source joint's world rotation, corrected at rest so the canonical bone lies along
  the source bone, and re-aimed along the current source chain where unmapped joints sit inside it (CMU's Neck,
  Mixamo's Spine1). The root is scaled by hip height, then the clip is grounded: the lower toe's 10th-percentile height
  equals the 3.5 cm foot radius. Fingers, LHipJoint, LowerBack and twist bones vanish.
- **Spike repair**: a source joint more than CAP/hz (30 rad/s) from both neighbours while they agree within it becomes
  their midpoint, judged on the unrepaired clip. 0 to 5 per clip; every other frame is unchanged (error 0). A per-frame
  velocity cap would be wrong here: a CMU kick's toes reach 59 rad/s and a salsa foot turn 247 rad/s for real.
- **Planting**: contact runs (joined across gaps of up to 2 BLEND frames) hold the limb end at the run's mean position by
  one analytic two-bone solve per limb (law of cosines, bending about the limb's current bend normal); outside runs the
  edge correction fades over BLEND = 6 frames; the end body keeps its world orientation. Legs plant on foot contacts;
  arms on hand contacts (hand under HANDZ = 6.5 cm, none in these clips; `HANDZ=2` exercises the path in the gate).

| knob (env) | default | meaning |
|---|---|---|
| `MASS`, `OMEGA`, `ASSIST`, `SUB`, `HZ` | 70, 10, 0.005, 4, 120 | `bvh mjcf`: kg, servo Hz, weld time constant (0 = none), steps per frame, clip rate |
| `SCALE` | 0.056444 | m per BVH unit (CMU; 0.01 for centimetres) |
| `FIRST`, `FRAMES` | 1, all | clip window (CMU conversions open on a rest pose) |
| `REPAIR`, `PLANT`, `CAP`, `BLEND`, `HANDZ` | 1, 1, 30, 6, 0.065 | `bvh clip` cleaning and planting |

Velocity gains live on the joints, not the servos: the assist weld needs the Euler integrator, which integrates joint
damping implicitly and a servo's kv explicitly. Bones are convex meshes (plane-mesh contact is ported, plane-capsule
is not).

## Checks (`check.sh bvh`, per clip)

An independent numpy twin (matrices, its own CMU alias table, the canonical rest read back from the MJCF) reproduces
the retarget within 3.1e-7 on frames no planting window touches; Mojo's FK matches MuJoCo's FK of the .qref within
3e-7; a Mixamo-style rewrite of the clip (cm, split spine, random twist leaves, reordered) retargets to the same .qref
within 4.8e-7; spike counts equal the twin's; planting holds the end to 0.0002 mm/frame where reachable, never pops,
keeps the bend side, and leaves every other frame exact. 14 planted bugs, each caught.

## Measured

| | ours | Godot 4.7.2 (same machine) |
|---|---|---|
| retarget | 2.8M to 3.65M frames/s (21 bodies, parallel over frames) | RetargetModifier3D 23.8k to 25.1k skeletons/s |
| limb planting, whole pass (FK + contacts + solves) | 5.9M to 6.2M limb solves/s | TwoBoneIK3D 243k to 267k solves/s |
| code | `bvh.mojo` 527 code lines (parser, retarget, repair, planting, MJCF and clip writers) | 5,998 code lines in its 30 modifier files |

Retarget is 112x to 146x Godot; planting 24x, short of the 100x target (load average 8 to 11 on the box during the
runs).

Tracking on 10_01 (800 frames, 64 animals, `check.sh track`):

| loop | pos_err m | rot_err rad | npss | foot_skate m/s | contact_err |
|---|---|---|---|---|---|
| Kinestride vs reference | 0.01758 | 0.08096 | 0.05811 | 0.2562 | 0.7812 |
| MuJoCo vs reference | 0.01758 | 0.08096 | 0.05811 | 0.2562 | 0.7812 |
| Kinestride vs MuJoCo | 1.2e-7 | 4.1e-7 | 2.0e-7 | | 0 |

- The assist still does much of the balancing (v0.8.0: 0.090 m at `ASSIST=0.02`); the servos alone do not yet balance
  these clips. foot_skate here is the simulated foot's: the reference's ankle no longer moves inside contact runs.
- contact_err compares the reference's height + speed rule with the solver's contacts (two definitions), not barred.
- 28 ms per frame (4 steps, 64 animals; v0.8.0's 31-body character: 61 ms), card shared with other jobs.
