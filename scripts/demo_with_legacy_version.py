import os
import sys
import numpy as np

root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Adjust as needed
sys.path.append(root_dir)

from mestDS.classes import MultipleSimulations

m_sim = MultipleSimulations(yaml_path="scripts/simulation2.yaml")
m_sim.simulate()
m_sim.eval_chap_model("simulations", "models/minimalist_multiregion")
