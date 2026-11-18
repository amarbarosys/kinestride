"""Flybody flight model (flight_imitation, FruitFly with wings and the abdomen tendons) as a MuJoCo MJCF plus its meshes.
Run with fly-connectome's flight venv: FC/.work/neural-flies/flight-venv/bin/python tools/flight_export.py OUT_DIR
(writes OUT_DIR/flight.xml; wing pattern from $WPG, default .work/flybody-data/wing_pattern_fmech.npy fetched by tools/fetch_flybody.sh)"""
import os
import sys

import numpy as np
from dm_control import mjcf
from flybody.fly_envs import flight_imitation

WPG = os.environ.get("WPG", os.path.join(os.path.dirname(__file__), "..", ".work/flybody-data/wing_pattern_fmech.npy"))
env = flight_imitation(wpg_pattern_path=WPG,
                       terminal_com_dist=float('inf'), random_state=np.random.RandomState(1))
mjcf.export_with_assets(env.task.root_entity.mjcf_model, sys.argv[1], out_file_name='flight.xml')
