from typing import List

import yaml

from mestDS.classes import RainSeason, Region
from . import Simulation


class MultipleSimulations:
    simulations: List[Simulation]

    def __init__(self, yaml_path):
        self.simulations = parse_yaml(yaml_path)

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()

    def graph(self, folder_name=None):
        for i, simulation in enumerate(self.simulations):
            title = f"Simulation (r:{simulation.beta_rainfall}, ls: {simulation.beta_lag_sickness}, t: {simulation.beta_temp}, n: {simulation.beta_neighbour_influence})"
            if folder_name:
                file_name = f"{folder_name}/{i}.png"
                simulation.graph(file_name=file_name, title=title)
            else:
                simulation.graph(title=title)


def parse_yaml(yaml_path):
    parameters = load_yaml(yaml_path)
    simulations = []
    sim_base = Simulation()
    regions = parameters.get("simulation", {}).get("regions", {})
    region_list = []
    for reg in regions:
        region = Region()
        for key, value in reg.items():
            if key == "rain_season":
                rain_season = []
                for season in value:
                    rain_season.append(RainSeason(season[0], season[1]))
                region.__setattr__(key, rain_season)
            else:

                region.__setattr__(key, value)
        region_list.append(region)
    sim_base.regions = region_list

    base = parameters.get("simulation", {}).get("base", {})
    for key, value in base.items():
        sim_base.__setattr__(key, value)

    sims = parameters.get("simulation", {}).get("sims", {})
    if sims:
        for i, sim in enumerate(sims):
            simulation = Simulation()
            simulation.__dict__.update(sim_base.__dict__)
            for key, value in sim.items():
                simulation.__setattr__(key, value)
            print(simulation.__dict__)
            simulations.append(simulation)

    return simulations


def load_yaml(yaml_path):
    parameters = None
    with open(yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    if parameters is None:
        raise ValueError

    return parameters
