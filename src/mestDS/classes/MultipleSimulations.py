from typing import List

import yaml
from . import Simulation


class MultipleSimulations:
    simulations: List[Simulation]

    def __init__(self, yaml_path):
        self.simulations = parse_yaml(yaml_path)


def parse_yaml(yaml_path):
    parameters = load_yaml(yaml_path)
    simulations = []
    sim_base = Simulation()
    base = parameters.get("simulation", {}).get("base", {})
    for key, value in base.items():
        sim_base.__setattr__(key, value)

    sims = parameters.get("simulation", {}).get("sims", {})
    if sims:
        for i, sim in enumerate(sims):
            simulation = sim_base
            print(f"Sim {i} with values: ")
            for key, value in sim.items():
                simulation.__setattr__(key, value)
                print(f"{key}: {value}")
                simulations.append(simulation)
            simulation.simulate()
            simulation.simulated_data_to_csv(f"simulation_{i}.csv")

    return simulations


def load_yaml(yaml_path):
    parameters = None
    with open(yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    if parameters is None:
        raise ValueError

    return parameters
