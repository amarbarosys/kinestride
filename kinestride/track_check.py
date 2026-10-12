"""Gate for track.mojo (unit B): the same tracking loop in MuJoCo and in Kinestride on a CMU clip, scored by
motion_metrics against the reference.

Loop, per reference frame f -> f + 1: hinge servo targets = qref[f + 1] hinges, assist mocap = qref[f + 1] root,
SUB physics steps; frames 1..T-1 recorded. Start: qref[0], velocities by mj_differentiatePos to qref[1]. Sim contacts
come from the solver (a foot body is in contact when any contact of its geoms is active), the reference's from bvh clip
(height + speed), so contact_err compares two definitions and is reported, not barred.

Bars (measured on 10_01 in MuJoCo first: pos 0.021 m, rot 0.081 rad, npss 0.059; bars about 2x): each loop vs the
reference pos_err < 0.04 m, rot_err < 0.15 rad, npss < 0.15; Kinestride vs MuJoCo pos_err < 0.01 m, rot_err < 0.05 rad,
contact_err < 0.02 (both are solver contacts).

usage: python -m kinestride.track_check prepare STEM DIR FLIES    export + start state + pairbody.bin + STEM.ref1.mclip
       python -m kinestride.track_check mujoco STEM               MuJoCo's loop -> STEM.mj.mclip
       python -m kinestride.track_check gate STEM METRICS_BINARY  the bars above over STEM.ks.mclip and STEM.mj.mclip
                                                                  (BARS=mujoco: Kinestride vs MuJoCo only)
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np


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


def start(m, qref, hz):
    v = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, v, 1 / hz, qref[0], qref[1])  # free joint angular part in the body frame
    return qref[0], v


def prepare(stem, out, flies):
    from kinestride.model import export
    export(f"{stem}.xml", out, "^g", True, False)
    m = mujoco.MjModel.from_xml_path(f"{stem}.xml")
    hz, qref = read_qref(stem)
    q, v = start(m, qref, hz)
    out = Path(out)
    for name, a in (("qpos.bin", q), ("qvel.bin", v)):
        (out / name).write_bytes(np.repeat(np.asarray(a, "<f4")[:, None], flies, 1).tobytes())  # [row][animal]
    (out / "pairbody.bin").write_bytes(np.asarray([m.geom_bodyid[m.pair_geom2[p]] - 1 for p in range(m.npair)], "<i4").tobytes())
    hz, feet, pos, quat, contact = read_mclip(f"{stem}.mclip")
    write_mclip(f"{stem}.ref1.mclip", hz, feet, pos[1:], quat[1:], contact[1:])


def mujoco_track(stem):
    hz, qref = read_qref(stem)
    _, feet, _, _, _ = read_mclip(f"{stem}.mclip")
    m = mujoco.MjModel.from_xml_path(f"{stem}.xml")
    d = mujoco.MjData(m)
    sub = round(1 / (hz * m.opt.timestep))
    nb = m.nbody - 1 - m.nmocap  # character bodies 1..nb; the assist mocap body is last
    d.qpos[:], d.qvel[:] = start(m, qref, hz)
    t = len(qref) - 1
    pos, quat, contact = np.zeros((t, nb, 3)), np.zeros((t, nb, 4)), np.zeros((t, len(feet)))
    for f in range(t):
        d.ctrl[:] = qref[f + 1, 7:]
        d.mocap_pos[0], d.mocap_quat[0] = qref[f + 1, :3], qref[f + 1, 3:7]
        for _ in range(sub):
            mujoco.mj_step(m, d)
        pos[f], quat[f] = d.xpos[1:nb + 1], d.xquat[1:nb + 1]
        bodies = {m.geom_bodyid[g] for c in d.contact[:d.ncon] for g in c.geom}
        contact[f] = [1.0 if b + 1 in bodies else 0.0 for b in feet]  # feet are clip body indices; MuJoCo ids are + 1
    write_mclip(f"{stem}.mj.mclip", hz, feet, pos, quat, contact)


def metrics(binary, ref, sim):
    out = subprocess.run([binary, ref, sim], check=True, capture_output=True, text=True).stdout.split()
    return dict(zip(out[0::2], map(float, out[1::2])))


def gate(stem, binary):
    bars = {"ks vs reference": ("ref1", "ks", {"pos_err": 0.04, "rot_err": 0.15, "npss": 0.15}),
            "MuJoCo vs reference": ("ref1", "mj", {"pos_err": 0.04, "rot_err": 0.15, "npss": 0.15}),
            "ks vs MuJoCo": ("mj", "ks", {"pos_err": 0.01, "rot_err": 0.05, "contact_err": 0.02})}
    if os.environ.get("BARS") == "mujoco":  # mutant runs: the driver against MuJoCo only (short clips inflate npss vs the reference)
        bars = {"ks vs MuJoCo": bars["ks vs MuJoCo"]}
    fails = []
    for name, (ref, sim, bar) in bars.items():
        got = metrics(binary, f"{stem}.{ref}.mclip", f"{stem}.{sim}.mclip")
        bad = [k for k, b in bar.items() if not got[k] < b]
        print(f"{'FAIL' if bad else 'ok  '} {name}: " + "  ".join(f"{k} {v:.4g}" + (f" (< {bar[k]})" if k in bar else "") for k, v in got.items()))
        fails += [f"{name} {k}" for k in bad]
    if fails:
        print(f"FAIL track {Path(stem).name}: {', '.join(fails)}")
        sys.exit(1)
    print(f"PASS track {Path(stem).name}: both loops under the bars, Kinestride within the MuJoCo bars")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"prepare": lambda: prepare(args[0], args[1], int(args[2])), "mujoco": lambda: mujoco_track(args[0]),
     "gate": lambda: gate(args[0], args[1])}[cmd]()
