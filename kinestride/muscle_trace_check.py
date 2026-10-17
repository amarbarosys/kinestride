"""Compare a 240-step stage-7 engine trace with the closed-form oracle."""
from pathlib import Path
import subprocess
import sys

import numpy as np

from kinestride.model import export
from kinestride.muscle_check import Params, activation, fatigue, force, velocity_factor


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "kinestride/models/muscle_export.xml"
CHECKPOINTS = (1, 40, 120, 160, 200, 240)


def excitation(step):
    if step <= 40:
        return step / 40.0
    if step <= 120:
        return 1.0
    return 0.0


def check(binary, work):
    binary, work = Path(binary).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    model = export(str(FIXTURE), work, regex="^$", act=True, limits=True)
    np.zeros(model.nq, np.float32).tofile(work / "qpos.bin")
    np.full(model.nv, -0.5, np.float32).tofile(work / "qvel.bin")
    np.zeros(model.nu, np.float32).tofile(work / "ctrl.bin")
    trace_path = work / "trace.bin"
    trace_path.unlink(missing_ok=True)
    subprocess.run([str(binary), str(work) + "/", "240", "trace"], check=True, cwd=ROOT)
    actual = np.fromfile(trace_path, np.float32)
    assert actual.size == len(CHECKPOINTS) * 6, f"trace has {actual.size} floats"
    actual = actual.reshape(-1, 6)

    p = Params(fmax=10, lopt=0, width=.25, vmax=1, eccentric_cap=1.5,
               kpassive=0, lslack=0, tau_rise=.04, tau_fall=.07,
               tau_fatigue=.08, tau_recovery=.25)
    a, f = 0.0, 0.0
    expected = {}
    for step in range(1, 241):
        u = excitation(step)
        p0 = a * (1.0 - f) * velocity_factor(-0.5, p)  # length factor is one
        a1 = activation(a, u, model.opt.timestep, p)
        f1 = fatigue(f, p0, model.opt.timestep, p)
        tendon_force = force(a1, f, 0.0, -0.5, p)
        if step in CHECKPOINTS:
            expected[step] = (u, tendon_force, tendon_force, a1, f1)
        a, f = a1, f1

    assert expected[120][4] > expected[40][4], "fatigue did not accumulate during excitation"
    assert 0 < expected[240][4] < expected[120][4], "fatigue did not recover during rest"
    for row, step in zip(actual, CHECKPOINTS):
        got = row[1:]
        want = np.asarray(expected[step])
        np.testing.assert_allclose(got, want, rtol=3e-5, atol=3e-5,
                                   err_msg=f"step {step}: [excitation, tendon force, qfrc, activation, fatigue]")
    print("PASS muscle trace vs oracle: 240 steps, 6 checkpoints, activation, tendon force, fatigue accumulation/recovery")
    check_near_zero_recovery(binary, work, model, p)


def check_near_zero_recovery(binary, work, model, p):
    a0, f0, speed = 1.0 / 7.5, 0.8, -0.5
    target = a0 * (1.0 - f0) * velocity_factor(speed, p)
    assert 0 < target < f0 and abs(target - 0.01) < 1e-12
    for u, mode in ((0.0, "recovery0"), (0.01, "recovery001")):
        subprocess.run([str(binary), str(work) + "/", "1", mode], check=True, cwd=ROOT)
        actual = np.fromfile(work / "trace.bin", np.float32)
        assert actual.size == 6, f"{mode} trace has {actual.size} floats"
        a1 = activation(a0, u, model.opt.timestep, p)
        f1 = fatigue(f0, target, model.opt.timestep, p)
        f_wrong = target + (f0 - target) * np.exp(-model.opt.timestep / p.tau_fatigue)
        assert abs(f1 - f_wrong) > 1e-5, "fixture cannot distinguish recovery from fatigue tau"
        expected = np.asarray((u, force(a1, f0, 0.0, speed, p),
                               force(a1, f0, 0.0, speed, p), a1, f1))
        np.testing.assert_allclose(actual[1:], expected, rtol=3e-5, atol=3e-5,
                                   err_msg=f"{mode}: low target must use tau_recovery")
    print("PASS low-excitation recovery: u=0 and 0.01 both use tau_recovery from f=0.8")


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2])
