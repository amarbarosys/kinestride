"""Compare stage-7 muscle rows with the independent closed-form oracle."""
from pathlib import Path
import subprocess
import sys

import numpy as np

from kinestride.muscle_check import Params, activation, fatigue, force
from kinestride.model import export


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "kinestride/models/muscle_export.xml"


def check(binary, work):
    binary, work = Path(binary).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "muscle-first.bin").unlink(missing_ok=True)
    m = export(str(FIXTURE), work, regex="^$", act=True, limits=True)
    qpos = np.zeros(m.nq, np.float32)
    qvel = np.full(m.nv, -0.5, np.float32)
    ctrl = np.ones(m.nu, np.float32)
    qpos.tofile(work / "qpos.bin")
    qvel.tofile(work / "qvel.bin")
    ctrl.tofile(work / "ctrl.bin")
    subprocess.run([str(binary), str(work) + "/"], check=True, cwd=ROOT)
    rows = np.fromfile(work / "muscle-first.bin", np.float32)
    assert len(rows) >= 4, f"expected force, qfrc, activation, fatigue rows; got {len(rows)}"
    got_force, got_qfrc, got_activation, got_fatigue = rows[-4:]
    p = Params(fmax=10, lopt=0, width=.25, vmax=1, eccentric_cap=1.5,
               kpassive=0, lslack=0, tau_rise=.04, tau_fall=.07,
               tau_fatigue=8, tau_recovery=20)
    a1 = activation(0, 1, m.opt.timestep, p)
    f0 = fatigue(0, 0, m.opt.timestep, p)
    expected_force = force(a1, 0, 0, -.5, p)
    checks = (("actuator force", got_force, expected_force),
              ("tendon qfrc", got_qfrc, expected_force),
              ("activation", got_activation, a1),
              ("fatigue", got_fatigue, f0))
    for name, got, want in checks:
        assert abs(float(got) - float(want)) <= 2e-6 * max(1.0, abs(float(want))), (name, got, want)
    print("PASS muscle engine vs oracle: force, tendon qfrc, activation, fatigue")


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2])
