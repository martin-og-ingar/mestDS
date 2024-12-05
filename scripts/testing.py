import os
import sys
import numpy as np

root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Adjust as needed
sys.path.append(root_dir)

from mestDS import Simulation
from mestDS.classes import MultipleSimulations, RainSeason


m_sim = MultipleSimulations(yaml_path="scripts/simulation2.yaml")
m_sim.simulations[0].graph()
