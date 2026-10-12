"""Gates for track.mojo: the same tracking loop in MuJoCo and in Kinestride, scored by motion_metrics.

Loop, per animal and reference frame f -> f + 1: hinge servo target = q + w (ref - q), assist mocap = the reference root,
assist weld on until the animal's push frame (MuJoCo eq_active), push wrench on its body for its push frames, w ramping
from 1 to its fall influence after the push; SUB physics steps; frames 1..T-1 recorded. Start: the clip's frame 0,
velocities by mj_differentiatePos to frame 1. Animal a plays clip a mod (number of clips); shorter clips hold their last
frame. Sim contacts are the solver's (a foot body is in contact when any contact of its geoms is active).

track (one clip, every animal the same, no push), measured on 10_01 in MuJoCo first (0.021 m, 0.081 rad, npss 0.059;
bars about 2x): every animal byte-equal; each loop vs the reference pos_err < 0.04 m, rot_err < 0.15 rad, npss < 0.15;
Kinestride vs MuJoCo pos_err < 0.01 m, rot_err < 0.05 rad, contact_err < 0.02.
dataset (4 clips, half the animals pushed at 1 s on the chest, 300-500 N for 0.1 s, fall influence 0.3 over 0.25 s):
every animal vs MuJoCo before the push and 10 frames after it within the track bars (later a fall is chaotic: reported);
pushed animals' hips drop more than 0.5 m, unpushed animals' stay within 0.1 m of the reference; labels TSV written.

usage: python -m kinestride.track_check prepare DIR FLIES SCHEDULE STEM...   export DIR/character.xml, ref.bin, plan.bin, start, maps
       python -m kinestride.track_check mujoco DIR [ANIMALS]                MuJoCo's loop -> DIR/mj.bin (first ANIMALS)
       python -m kinestride.track_check track DIR METRICS                    the track bars over DIR/out.bin and DIR/mj.bin
       python -m kinestride.track_check dataset DIR METRICS CMU_INDEX        the dataset bars, DIR/labels.tsv
(SCHEDULE: none | push)
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

PUSH_BODY = "Chest"


def read_qref(stem):
    raw = Path(f"{stem}.qref").read_bytes()
    t, nq = struct.unpack_from("<2i", raw)
    hz = struct.unpack_from("<f", raw, 8)[0]
    return hz, np.frombuffer(raw, "<f4", offset=12).reshape(t, nq).astype(np.float64)


def read_mclip(path):
    raw = Path(path).read_bytes()
    t, nb, nf = struct.unpack_from("<3i", raw)
    hz = struct.unpack_from("<f", raw, 12)[0]
    feet = np.frombuffer(raw, "<i4", nf, 16)
    v = np.frombuffer(raw, "<f4", offset=16 + 4 * nf)
    return hz, feet, v[:t * nb * 3].reshape(t, nb, 3), v[t * nb * 3:t * nb * 7].reshape(t, nb, 4), v[t * nb * 7:].reshape(t, nf)


def write_mclip(path, hz, feet, pos, quat, contact):
    t, nb, _ = pos.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<3if", t, nb, len(feet), hz))
        f.write(np.asarray(feet, "<i4").tobytes())
        for a in (pos, quat, contact):
            f.write(np.asarray(a, "<f4").tobytes())


def write_out(path, hz, feet, pos, quat, con):
    """track.mojo's out.bin layout from [animal][t][...] arrays."""
    na, t, nb, _ = pos.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<4if", t, nb, len(feet), na, hz))
        f.write(np.asarray(feet, "<i4").tobytes())
        for k in range(t):
            for a in (pos[:, k].reshape(na, -1).T, quat[:, k].reshape(na, -1).T, con[:, k].T):
                f.write(np.ascontiguousarray(a, "<f4").tobytes())


def read_out(path):
    """out.bin -> hz, feet, pos [animal][t][body][3], quat [..][4], contact [animal][t][feet]."""
    raw = Path(path).read_bytes()
    t, nb, nf, na = struct.unpack_from("<4i", raw)
    hz = struct.unpack_from("<f", raw, 16)[0]
    feet = np.frombuffer(raw, "<i4", nf, 20)
    v = np.frombuffer(raw, "<f4", offset=20 + 4 * nf).reshape(t, 7 * nb + nf, na)
    pos = v[:, :3 * nb].reshape(t, nb, 3, na).transpose(3, 0, 1, 2)
    quat = v[:, 3 * nb:7 * nb].reshape(t, nb, 4, na).transpose(3, 0, 1, 2)
    return hz, feet, pos, quat, v[:, 7 * nb:].transpose(2, 0, 1)


def clips(out):
    return Path(out, "clips.txt").read_text().split()


def prepare(out, flies, schedule, stems):
    from kinestride.model import export
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    export(str(out / "character.xml"), str(out), "^g", True, False)
    m = mujoco.MjModel.from_xml_path(str(out / "character.xml"))
    refs = [read_qref(s) for s in stems]
    hz = refs[0][0]
    t = min(int(os.environ.get("FRAMES", 10 ** 9)), max(len(r) for _, r in refs))
    ref = np.zeros((t, m.nq, flies))
    q0, v0 = np.zeros((m.nq, flies)), np.zeros((m.nv, flies))
    for a in range(flies):
        r = refs[a % len(stems)][1][:t]
        ref[:, :, a] = np.concatenate([r, np.repeat(r[-1:], t - len(r), 0)])  # a shorter clip holds its last frame
        q0[:, a] = r[0]
        va = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, va, 1 / hz, r[0], r[1])  # free joint angular part in the body frame
        v0[:, a] = va
    with open(out / "ref.bin", "wb") as f:
        f.write(struct.pack("<if", t, hz))
        f.write(ref.astype("<f4").tobytes())
    plan = np.zeros((flies, 11))
    plan[:, 0] = -1
    if schedule == "push":
        rng = np.random.default_rng(11)
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PUSH_BODY)
        for a in range(0, flies, 2):
            ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(300, 500)
            plan[a] = [round(hz), body, round(0.1 * hz), mag * np.cos(ang), mag * np.sin(ang), 0, 0, 0, 0, 0.3, round(0.25 * hz)]
    plan.astype("<f4").tofile(out / "plan.bin")
    for name, a in (("qpos.bin", q0), ("qvel.bin", v0)):
        a.astype("<f4").tofile(out / name)  # [row][animal]
    (out / "pairbody.bin").write_bytes(np.asarray([m.geom_bodyid[m.pair_geom2[p]] - 1 for p in range(m.npair)], "<i4").tobytes())
    _, feet, *_ = read_mclip(f"{stems[0]}.mclip")
    (out / "feet.bin").write_bytes(np.asarray([len(feet), *feet], "<i4").tobytes())
    (out / "clips.txt").write_text("\n".join(str(s) for s in stems) + "\n")


def mujoco_track(out, animals=None):
    out = Path(out)
    m = mujoco.MjModel.from_xml_path(str(out / "character.xml"))
    d = mujoco.MjData(m)
    raw = (out / "ref.bin").read_bytes()
    t, hz = struct.unpack_from("<if", raw)
    flies = np.fromfile(out / "plan.bin", "<f4").size // 11
    ref = np.frombuffer(raw, "<f4", offset=8).reshape(t, m.nq, flies).astype(np.float64)
    plan = np.fromfile(out / "plan.bin", "<f4").reshape(flies, 11).astype(np.float64)
    q0 = np.fromfile(out / "qpos.bin", "<f4").reshape(m.nq, flies).astype(np.float64)
    v0 = np.fromfile(out / "qvel.bin", "<f4").reshape(m.nv, flies).astype(np.float64)
    feet = np.fromfile(out / "feet.bin", "<i4")[1:]
    sub = round(1 / (hz * m.opt.timestep))
    nb = m.nbody - 1 - m.nmocap  # character bodies 1..nb; the assist mocap body is last
    na = animals or flies
    pos, quat, con = np.zeros((na, t - 1, nb, 3)), np.zeros((na, t - 1, nb, 4)), np.zeros((na, t - 1, len(feet)))
    for a in range(na):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = q0[:, a], v0[:, a]
        p = plan[a]
        for f in range(t - 1):
            since = f - int(p[0]) if p[0] >= 0 else -1
            w = 1 - (1 - p[9]) * min(1.0, (since + 1) / max(1.0, p[10])) if since >= 0 else 1.0
            d.ctrl[:] = ref[f + 1, 7:, a] if w == 1 else d.qpos[7:] + w * (ref[f + 1, 7:, a] - d.qpos[7:])
            d.mocap_pos[0], d.mocap_quat[0] = ref[f + 1, :3, a], ref[f + 1, 3:7, a]
            d.xfrc_applied[:] = 0
            if 0 <= since < p[2]:
                d.xfrc_applied[int(p[1])] = p[3:9]
            d.eq_active[0] = 0 if since >= 0 else 1
            for _ in range(sub):
                mujoco.mj_step(m, d)
            pos[a, f], quat[a, f] = d.xpos[1:nb + 1], d.xquat[1:nb + 1]
            bodies = {m.geom_bodyid[g] for c in d.contact[:d.ncon] for g in c.geom}
            con[a, f] = [1.0 if b + 1 in bodies else 0.0 for b in feet]  # feet are clip body indices; MuJoCo ids are + 1
    write_out(out / "mj.bin", hz, feet, pos, quat, con)


def reference(out, a, t):
    """Animal a's reference .mclip over frames 1..t (held at its last frame), for motion_metrics."""
    stems = clips(out)
    hz, feet, pos, quat, con = read_mclip(f"{stems[a % len(stems)]}.mclip")
    pick = np.minimum(np.arange(1, t + 1), len(pos) - 1)
    return hz, feet, pos[pick], quat[pick], con[pick]


def metrics(binary, out, ref, sim):
    """motion_metrics over two (hz, feet, pos, quat, contact) clips."""
    write_mclip(f"{out}/m_ref.mclip", *ref)
    write_mclip(f"{out}/m_sim.mclip", *sim)
    words = subprocess.run([binary, f"{out}/m_ref.mclip", f"{out}/m_sim.mclip"], check=True, capture_output=True, text=True).stdout.split()
    return dict(zip(words[0::2], map(float, words[1::2])))


def head(c, n):
    """The first n frames of a (hz, feet, pos, quat, contact) clip."""
    return c[0], c[1], c[2][:n], c[3][:n], c[4][:n]


def bars(name, got, bar, fails):
    bad = [k for k, b in bar.items() if not got[k] < b]
    print(f"{'FAIL' if bad else 'ok  '} {name}: " + "  ".join(f"{k} {v:.4g}" + (f" (< {bar[k]})" if k in bar else "") for k, v in got.items()))
    fails += [f"{name} {k}" for k in bad]


REF_BARS = {"pos_err": 0.04, "rot_err": 0.15, "npss": 0.15}
SIM_BARS = {"pos_err": 0.01, "rot_err": 0.05, "contact_err": 0.02}


def track_gate(out, binary):
    hz, feet, pos, quat, con = read_out(f"{out}/out.bin")
    _, _, mpos, mquat, mcon = read_out(f"{out}/mj.bin")
    fails = []
    same = all(np.array_equal(x[0], x[a]) for x in (pos, quat, con) for a in range(1, len(pos)))
    print(f"{'ok  ' if same else 'FAIL'} {len(pos)} animals with the same input byte-equal")
    if not same:
        fails.append("animals differ")
    ks, mj = (hz, feet, pos[0], quat[0], con[0]), (hz, feet, mpos[0], mquat[0], mcon[0])
    if os.environ.get("BARS") != "mujoco":  # mutant runs: the driver against MuJoCo only (short clips inflate npss)
        ref = reference(out, 0, pos.shape[1])
        bars("ks vs reference", metrics(binary, out, ref, ks), REF_BARS, fails)
        bars("MuJoCo vs reference", metrics(binary, out, ref, mj), REF_BARS, fails)
    bars("ks vs MuJoCo", metrics(binary, out, mj, ks), SIM_BARS, fails)
    if fails:
        print(f"FAIL track: {', '.join(fails)}")
        sys.exit(1)
    print("PASS track: animals equal, both loops under the bars, Kinestride within the MuJoCo bars")


def dataset_gate(out, binary, index):
    hz, feet, pos, quat, con = read_out(f"{out}/out.bin")
    _, _, mpos, mquat, mcon = read_out(f"{out}/mj.bin")
    plan = np.fromfile(f"{out}/plan.bin", "<f4").reshape(-1, 11)
    stems = clips(out)
    desc = dict(line.split("\t", 1) for line in Path(index).read_text(errors="replace").split("\n") if "\t" in line)
    fails, rows = [], ["animal\tclip\tdescription\tpush_frame\tpush_N\tassist_off_frame\tfell\thips_drop_m\tpos_err_m\trot_err_rad"]
    t = pos.shape[1]
    for a in range(len(mpos)):
        push = int(plan[a, 0])
        ks, mj, ref = (hz, feet, pos[a], quat[a], con[a]), (hz, feet, mpos[a], mquat[a], mcon[a]), reference(out, a, t)
        cut = t if push < 0 else min(t, push + 10)
        bars(f"animal {a} ks vs MuJoCo, frames 1..{cut}", metrics(binary, out, head(mj, cut), head(ks, cut)), SIM_BARS, fails)
        hips = ref[2][:, 0, 2] - pos[a][:, 0, 2]
        drop = float(hips.max()) if push >= 0 else float(np.abs(hips).max())
        fell, stays = push >= 0 and drop > 0.5, push < 0 and drop < 0.1
        if not (fell or stays):
            fails.append(f"animal {a} {'did not fall' if push >= 0 else 'left the reference'} ({drop:.3f} m)")
        pre = metrics(binary, out, head(ref, push if push > 0 else t), head(ks, push if push > 0 else t))
        clip = Path(stems[a % len(stems)]).name
        rows.append(f"{a}\t{clip}\t{desc.get(clip, '').strip()}\t{push}\t{np.linalg.norm(plan[a, 3:6]):.0f}\t{push}\t{int(fell)}\t{drop:.3f}\t{pre['pos_err']:.4f}\t{pre['rot_err']:.4f}")
    print(f"{'ok  ' if not any('fall' in f or 'reference' in f for f in fails) else 'FAIL'} falls: pushed animals' hips drop > 0.5 m, unpushed stay within 0.1 m")
    Path(out, "labels.tsv").write_text("\n".join(rows) + "\n")
    print(f"labels: {out}/labels.tsv ({len(rows) - 1} animals)")
    if fails:
        print(f"FAIL dataset: {', '.join(fails)}")
        sys.exit(1)
    print("PASS dataset: every animal matches MuJoCo through its push, pushed animals fall, unpushed track")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "prepare":
        prepare(args[0], int(args[1]), args[2], args[3:])
    elif cmd == "mujoco":
        mujoco_track(args[0], int(args[1]) if len(args) > 1 else None)
    elif cmd == "track":
        track_gate(args[0], args[1])
    elif cmd == "dataset":
        dataset_gate(args[0], args[1], args[2])
