import os
import sys
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from mestDS.classes.ClimateHealth import MultipleClimateHealth

ch_sim = MultipleClimateHealth("scripts/simulation5.yaml")
ch_sim.simulate()
