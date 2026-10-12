# Motion: mocap clips in, tracked physics out

A CMU mocap clip becomes a 70 kg MJCF character and a reference clip; Kinestride tracks the reference with joint servos
and a root assist, 64 animals at a time, and its trajectory matches MuJoCo's own run of the same loop to 0.67 µm mean
body position over 800 frames (`check.sh track`). Plan and design notes: `plans/MOTION_DATASET.md`.

```
bvh mjcf CLIP.bvh OUT.xml        character            check.sh bvh    (CPU)
bvh clip CLIP.bvh OUT            OUT.qref, OUT.mclip  check.sh bvh    (CPU)
track DIR STEM                   STEM.ks.mclip        check.sh track  (GPU, under gpu-wait)
motion_metrics REF SIM [OUT]     five numbers         check.sh motion (CPU)
```

Test clips are not in git: `tools/assets.sh` copies CMU 02_05, 10_01, 55_22, 60_01 and the CMU index into
`.work/assets/cmu/` (CMU: "may be copied, modified, or redistributed without permission").

## The character tracks the clip because the clip is re-expressed for hinges

Every BVH joint is a body with three hinges. Hinge angles are intrinsic Tait-Bryan angles, so the axis order matters:
the BVH's own Z Y X order puts CMU shoulders and thumbs through gimbal lock, where z and x swing by pi between frames.
`bvh` picks, per joint, the order whose middle angle stays farthest from 90 degrees over the clip, re-extracts every
frame in it, and keeps each frame on the branch nearest the previous one. Forward kinematics is unchanged by any of
this, which is what `check.sh bvh` holds: Mojo FK, an independent numpy FK and MuJoCo FK of the `.qref` agree within
3e-7 on all four clips. Frame 0 of a CMU conversion is a rest pose and is dropped (`FIRST=1`).

| knob (env, `bvh mjcf`) | default | meaning |
|---|---|---|
| `MASS` | 70 | total kg; density is set from the hull volumes |
| `OMEGA` | 10 | servo frequency (Hz): kp = I w^2 on the servo, damping 2 I w on the joint, I per hinge axis |
| `ASSIST` | 0.005 | weld time constant (s) between the root and a mocap body on the reference root; 0 = none |
| `SUB` | 4 | physics steps per clip frame (timestep 1 / (SUB x clip hz)) |
| `SCALE` | 0.056444 | m per BVH unit (CMU) |
| `FIRST`, `FRAMES` | 1, all | clip window |

Velocity gains live on the joints, not the servos: the assist weld needs the Euler integrator, and Euler integrates
joint damping implicitly but a servo's kv explicitly (kv on a hanging leg's vertical axis blew up in 4 steps).

Bones are convex meshes (an octagon ring at each end and a pole beyond): plane-mesh contact is ported, plane-capsule
is not. A foot bone's radius is its joint's 10th-percentile height in the clip, so the foot reaches the floor.

## Measured on 10_01 (kick, 800 frames, 64 animals)

| loop | pos_err m | rot_err rad | npss | foot_skate m/s | contact_err |
|---|---|---|---|---|---|
| Kinestride vs reference | 0.02147 | 0.08156 | 0.05868 | 0.1673 | 0.5091 |
| MuJoCo vs reference | 0.02147 | 0.08155 | 0.05868 | 0.1673 | 0.5091 |
| Kinestride vs MuJoCo | 6.7e-7 | 2.6e-6 | 2.6e-6 | | 0 |

- The assist does much of the root's work: at `ASSIST=0.02` the same loop tracks at 0.090 m / 0.303 rad. These clips
  are not yet balanced by the servos alone.
- contact_err vs the reference compares two definitions (the reference's height + speed guess, the sim's active solver
  contacts), so it is reported, not barred; Kinestride and MuJoCo agree on every contact.
- 61 ms per frame (4 steps, 64 animals) on a card shared with other queued jobs: one thread per animal, latency-bound
  at this batch size.
