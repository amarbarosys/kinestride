"""CPU exporter checks for annotated fixed-tendon muscles."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from kinestride import model


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "kinestride/models/muscle_export.xml"


def check(out):
    out = Path(out)
    m = mujoco.MjModel.from_xml_path(str(FIXTURE))
    ints, floats, count = model.actuators(m)
    assert count == 1 and ints.reshape(-1, 5)[0].tolist() == [4, 0, 1, 0, 0]
    params = np.array([10, 0, .25, 1, 1.5, 0, 0, .04, .07, 8, 20])
    assert np.allclose(floats[-11:], params)
    exported = model.export(m, out / "good", act=True)
    assert exported.nu == 1 and "NMUSCLE=1" in (out / "good/sizes.txt").read_text()

    legacy_tree = ET.parse(FIXTURE)
    legacy_tree.getroot().remove(legacy_tree.find("custom"))
    legacy = mujoco.MjModel.from_xml_string(ET.tostring(legacy_tree.getroot(), encoding="unicode"))
    li, lf, ln = model.actuators(legacy)
    assert ln == 0 and li.reshape(-1, 5)[0, 0] == mujoco.mjtTrn.mjTRN_TENDON and len(lf) == legacy.nu * 9
    model.export(legacy, out / "legacy", act=True)
    assert "NMUSCLE" not in (out / "legacy/sizes.txt").read_text()

    try:
        model.export(m, out / "no_act", act=False)
        raise AssertionError("muscle annotation was silently dropped without stage-7 export")
    except SystemExit as exc:
        assert "require the stage-7 actuator block" in str(exc.code)

    tree = ET.parse(FIXTURE)
    tree.find("custom/numeric").set("data", "10 1 .25 1 .5 0 0 .04 .07 8 20")
    bad = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    try:
        model.actuators(bad)
        raise AssertionError("invalid eccentric cap was accepted")
    except SystemExit as exc:
        assert "invalid parameters" in str(exc.code)
    print("PASS muscle exporter: encoding, parameters, legacy layout, stage refusal, invalid parameter refusal")


if __name__ == "__main__":
    import sys
    check(sys.argv[1])
