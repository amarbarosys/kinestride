# Engine API

A program drives Kinestride through three files and one binary, nothing else. Stage order, scratch rows and solver
internals change between releases; the files, rows and fields below change only with a release note.

## A model becomes two tables and its own binary

```
python -m kinestride.model export MJCF OUT        # or model.export(mjcf, out, regex, act, convex)
mojo build $(cat OUT/sizes.txt) -DFLIES=N -DSTAGE=7 [-DK=k] [-DNXF=k] kinestride/smooth.mojo -o step
```

`OUT/mi.bin` (int32) and `OUT/mf.bin` (float32) hold the model; `OUT/sizes.txt` holds the `-D` sizes, so a fly, a
spider or a game character each compiles a kernel shaped for it. Export refuses what the engine cannot simulate
(`FAIL export: ...`) instead of dropping it.

## Each animal owns columns in four arrays

Every per-animal array is `[row][animal]`, animals fastest, float32.

| file | rows | meaning |
|---|---|---|
| `qpos.bin` | NQ | MuJoCo qpos; a K-step launch writes it back in place |
| `qvel.bin` | NV | MuJoCo qvel; a K-step launch writes it back in place |
| `ctrl.bin`, first | NU | MuJoCo ctrl (actuators) |
| `ctrl.bin`, then 7 per mocap body | 7 NMOCAP | mocap_pos 3, mocap_quat 4 |
| `ctrl.bin`, then 7 per push slot | 7 NXF | MuJoCo xfrc_applied, sparse: body id (0 = empty), force 3, torque 3 at the body's centre of mass, world frame |

Two push slots on one body add. `-DNXF=0`, the default, compiles the push path out.

## One launch steps every animal

`step DIR/` (or the kernel `smooth`, or the host `smooth_at`, from Mojo with `-I kinestride`) advances every animal one
step, or K steps with `-DK=k`. `DIR/smooth.bin` holds the stage's MuJoCo fields per animal in world positions; their
names and order are `model.STAGES[stage]`. Each stage is checked against MuJoCo 3.9.0 by `kinestride/check.sh STAGE`.

## A live loop is a Viewer plus the app's own control

`viewer.Viewer(ctx, dir, cam)` loads the model, state and mesh tables of DIR onto one DeviceContext. Per frame the app
writes `u_b` (its controller), calls `step(ctx, k)` and `frame(ctx, kc, kr, ks, ksc, kcs)` with kernels it compiled once
(kernel handles cannot live in a struct), and reads `out_h` after copying `img_b`. `view.mojo` is that loop with ctrl held.

Not in this surface yet: contacts as a list, a window for game use.
