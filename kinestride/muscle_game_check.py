"""Compare GAME=1 muscle work drains against a per-step analytic oracle."""
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from kinestride.model import export
from kinestride.muscle_check import Params, activation, fatigue, force, velocity_factor
from kinestride.muscle_trace_check import CHECKPOINTS, excitation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "kinestride/models/muscle_export.xml"
GAME_RULES = (0.0, 0.1, 0.0)  # basal drain, work drain, health regen


def prepare(work):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    model = export(str(FIXTURE), work, regex="^$", act=True, limits=True)
    mf_path = work / "mf.bin"
    mf_path.write_bytes(mf_path.read_bytes() + struct.pack("<3f", *GAME_RULES))
    np.zeros(model.nq, np.float32).tofile(work / "qpos.bin")
    np.full(model.nv, -0.5, np.float32).tofile(work / "qvel.bin")
    np.zeros(model.nu, np.float32).tofile(work / "ctrl.bin")
    return model


def check(binary, work):
    binary, work = Path(binary).resolve(), Path(work).resolve()
    model = prepare(work)
    trace_path = work / "trace.bin"
    trace_path.unlink(missing_ok=True)
    subprocess.run([str(binary), str(work) + "/", "240", "game"], check=True, cwd=ROOT)
    actual = np.fromfile(trace_path, np.float32)
    assert actual.size == len(CHECKPOINTS) * 10, f"game trace has {actual.size} floats"
    actual = actual.reshape(-1, 10)

    p = Params(fmax=10, lopt=0, width=.25, vmax=1, eccentric_cap=1.5,
               kpassive=0, lslack=0, tau_rise=.04, tau_fall=.07,
               tau_fatigue=.08, tau_recovery=.25)
    dt = model.opt.timestep
    a, f = 0.0, 0.0
    energy = stamina = 1.0
    expected = {}
    for step in range(1, 241):
        u = excitation(step)
        flfv = velocity_factor(-0.5, p)
        a1 = activation(a, u, dt, p)
        output = a * (1.0 - f) * flfv
        f1 = fatigue(f, output, dt, p, resting=(u == 0.0))
        active_force = force(a1, f, 0.0, -0.5, p)
        work = abs(active_force * -0.5) * dt
        energy = min(1.0, max(0.0, energy - work * GAME_RULES[1]))
        stamina = min(1.0, max(0.0, stamina - work * GAME_RULES[1]))
        if step in CHECKPOINTS:
            expected[step] = (u, active_force, active_force, a1, f1,
                              energy, 1.0, stamina, 0.0)
        a, f = a1, f1

    assert expected[120][5] < 1.0, "muscle work did not drain energy"
    assert expected[120][8] < 1.0, "muscle work did not drain stamina"
    assert expected[240][5] <= expected[120][5], "rest interval unexpectedly restored energy"
    for row, step in zip(actual, CHECKPOINTS):
        np.testing.assert_allclose(row[1:], np.asarray(expected[step]), rtol=5e-5, atol=5e-5,
                                   err_msg=f"game step {step}: [u, actuator, qfrc, a, f, energy, health, stamina, dead]")
    print("PASS muscle game oracle: work drains energy/stamina at six checkpoints; rest has no work refund")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "prepare":
        prepare(sys.argv[2])
    else:
        check(sys.argv[1], sys.argv[2])
