# Provenance: motion v2 (canonical body, spike repair, limb planting, influence)

Where we saw the ideas: Godot 4.7.2-stable (official linux build, SHA-512 matched to the release's SHA512-SUMS.txt),
its class reference only: SkeletonProfileHumanoid, BoneMap, RetargetModifier3D, LimitAngularVelocityModifier3D,
TwoBoneIK3D (and FABRIK3D, CCDIK3D, JacobianIK3D, SplineIK3D, ChainIK3D, IterateIK3D, JointLimitationCone3D),
SkeletonModifier3D.influence, PhysicalBoneSimulator3D. Godot's source was not read: 30 files of its modifiers
(8,119 lines) sit in `.work/godot/ref-src/` for line counts and `clean-room-check` only.

## The ideas, in our words, with their primary sources

- **One standard body.** Any skeleton's motion is expressed on one fixed set of bodies by matching names to roles and
  carrying each role's world rotation relative to its rest pose. Primary: rest-relative retargeting as in Gleicher,
  "Retargetting Motion to New Characters", SIGGRAPH 1998 (the general problem); the rotation algebra is textbook.
  Ours: 20 physics bodies, not 56 animation bones; the whole clip offline; one MJCF for every clip.
- **Spike repair.** A sample that disagrees with both neighbours while the neighbours agree is an outlier, replaced from
  its neighbours. Primary: Hampel, "The influence curve and its role in robust estimation", JASA 1974 (outlier
  identification against a local reference). Ours: non-causal on quaternions (slerp of the neighbours), so it adds no
  lag; a causal per-frame velocity clamp lags every fast motion.
- **Limb planting.** A two-segment limb reaching a target has a closed form: the law of cosines fixes the middle joint's
  angle, a pole fixes the bend plane. Primary: textbook trigonometry; Tolani, Goswami, Badler, "Real-time inverse
  kinematics techniques for anthropomorphic limbs", Graphical Models 2000. Ours: one solver for four limbs (every human
  limb is two segments), targets from contact runs, blended at the run edges.
- **Influence.** A controller's output is mixed with the uncontrolled motion by a weight. Ours: per animal, per frame,
  as a servo target blend ctrl = q + w (q_ref - q) (equal to scaling the servo gain by w) and an equality on/off flag
  with MuJoCo's eq_active meaning, so a fall after a push is unassisted physics, checked against MuJoCo.

## Godot measured (tools/godot_bench.gd, headless, this machine, other jobs on the box)

Godot runs skeleton modifiers once per engine frame; a headless frame has a ~6.9 ms floor (300 frames of 1,000 legs took
2.07 s, of 10,000 legs 7.08 s), so the bench uses enough skeletons that work dominates and checks the output: the
ankle must land on the target inside `modification_processed` (worst miss 3.1e-7 m). TwoBoneIK3D writes nothing
without a pole node.

| modifier | rate (3 runs) | setup |
|---|---|---|
| TwoBoneIK3D | 243k, 250k, 267k solves/s | 10,000 legs x 300 frames |
| RetargetModifier3D | 23.8k, 25.1k, 24.1k skeleton retargets/s | 1,000 humanoid pairs (56 bones, global pose) x 300 frames |

## Targets, frozen before building (plan `cozy-mapping-milner`)

- Limb solves >= 100x Godot's TwoBoneIK3D rate: >= 25M solves/s.
- Frame retargets (20 bodies) >= 100x Godot's per-skeleton retarget rate: >= 2.4M frames/s.
- Code: `mojo-loc` of ours <= 1/10 of Godot's 8,119 lines for the same four ideas.
- Behaviour: retarget round trip across two topologies within 1e-5; spike repair adds zero lag; reference foot skate in
  contact runs < 1 mm/frame; Kinestride matches MuJoCo with the influence ramps on.
