"""Flybody flight model (flight_imitation, FruitFly with wings and the abdomen tendons) as a MuJoCo MJCF plus its meshes.
Run with fly-connectome's flight venv: FC/.work/neural-flies/flight-venv/bin/python tools/flight_export.py OUT_DIR [FC]
(writes OUT_DIR/flight.xml; FC = the fly-connectome checkout holding the wing pattern, default ~/Projects/science/fly-connectome)"""
import os
import sys

import numpy as np
from dm_control import mjcf
from flybody.fly_envs import flight_imitation

ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/Projects/science/fly-connectome')
env = flight_imitation(wpg_pattern_path=f'{ROOT}/.work/neural-flies/flight-data/wing_pattern_fmech.npy',
                       terminal_com_dist=float('inf'), random_state=np.random.RandomState(1))
mjcf.export_with_assets(env.task.root_entity.mjcf_model, sys.argv[1], out_file_name='flight.xml')
