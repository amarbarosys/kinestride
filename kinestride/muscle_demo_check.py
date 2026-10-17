"""Prepare the explicit-pair whole-spider model for the sine runner."""
from pathlib import Path
import sys

import numpy as np

from kinestride.model import export, with_auto_pairs


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paired = out / "muscle_spider_pairs.xml"
    source = with_auto_pairs("kinestride/models/muscle_spider.xml", str(paired), cone="pyramidal", explicit=True)
    m = export(source, out, regex="^$", act=True, limits=True)
    np.asarray(m.qpos0, np.float32).tofile(out / "qpos.bin")
    np.zeros(m.nv, np.float32).tofile(out / "qvel.bin")
    np.zeros(m.nu, np.float32).tofile(out / "ctrl.bin")
    assert m.nbody == 27 and m.nu == 1 and "NMUSCLE=1" in (out / "sizes.txt").read_text()
    print(f"PASS muscle spider fixture: {m.nbody} bodies, {m.nu} sine-driven muscle")


if __name__ == "__main__":
    prepare(sys.argv[1])
