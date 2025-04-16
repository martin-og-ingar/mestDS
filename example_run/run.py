# Example file to run the simulation defined under /example_run/config_one.py or
# /example_run/config_two.py


import os
import sys
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from mestDS.SimulationDemo import SimulationsDemo

ch_sim = SimulationsDemo("example_run/config_one.yaml")
ch_sim.simulate()

# Uncomment the lines below to perform model evaluation of your data.
# Remember to load the external model first

# ch_sim.convert_to_csvs("<directory_where_you_want_to_store_dataset>")
# ch_sim.eval_chap_model("<path-to-external-model>")
