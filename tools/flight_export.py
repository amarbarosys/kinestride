"""Flybody flight model (flight_imitation, FruitFly with wings and the abdomen tendons) as a MuJoCo MJCF plus its meshes.
Run with the flight venv: .work/neural-flies/flight-venv/bin/python models/flight_export.py OUT_DIR  (writes OUT_DIR/flight.xml)"""
import sys

import numpy as np
from dm_control import mjcf
from flybody.fly_envs import flight_imitation

ROOT = '/home/<user>/Projects/science/fly-connectome'
env = flight_imitation(wpg_pattern_path=f'{ROOT}/.work/neural-flies/flight-data/wing_pattern_fmech.npy',
                       terminal_com_dist=float('inf'), random_state=np.random.RandomState(1))
mjcf.export_with_assets(env.task.root_entity.mjcf_model, sys.argv[1], out_file_name='flight.xml')
