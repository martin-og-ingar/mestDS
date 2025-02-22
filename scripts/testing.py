import os
import sys
import numpy as np


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from mestDS.classes.SimulationDemo import SimulationsDemo
from mestDS.classes.Simulation import Simulations

ch_sim = SimulationsDemo("scripts/simulation6.yaml")
ch_sim.simulate()
# ch_sim.convert_to_csvs("testing_minimalist_multiregion/")
# ch_sim.eval_chap_model("models/minimalist_multiregion")

# ch_sim.convert_to_csvs("testing_minimalist_example_lag/")
# ch_sim.eval_chap_model("models/minimalist_example_lag")
