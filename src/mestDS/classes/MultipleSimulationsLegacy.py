import os
import subprocess
from typing import List

import yaml

from mestDS.classes import RainSeason, Region

# from mestDS.utils.main import plot_data_with_sample_0
from .SimulationLegacy import SimulationLegacy


class MultipleSimulationsLegacy:
    simulations: List[SimulationLegacy]

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

    def to_csv(self, folder_name):
        self.datasets_path = folder_name
        os.makedirs(folder_name, exist_ok=True)
        for i, simulation in enumerate(self.simulations):
            dir_path = f"{folder_name}/{i}"
            simulation.simulated_data_to_csv(dir_path, "dataset.csv")

    def csv_train_test_split(self):
        for simulation in self.simulations:
            simulation.split_csv_to_train_and_test()

    def eval_chap_model(self, folder_name, model_name):
        self.to_csv(folder_name)
        self.csv_train_test_split()

        for i, simulation in enumerate(self.simulations):
            train_command = [
                "python",
                f"{model_name}/train.py",
                f"{simulation.dir_path}/dataset_train.csv",
                f"{simulation.dir_path}/model.bin",
            ]
            subprocess.run(train_command, check=True)

            test_command = [
                "python",
                f"{model_name}/predict.py",
                f"{simulation.dir_path}/model.bin",
                "",
                f"{simulation.dir_path}/dataset_x_test.csv",
                f"{simulation.dir_path}/predictions.csv",
            ]

            subprocess.run(test_command, check=True)

            plot_subtitle = f"Rain beta: {simulation.beta_rainfall}, Temp beta: {simulation.beta_temp}, Neighbour beta: {simulation.beta_neighbour_influence}, Sickness lag beta: {simulation.beta_lag_sickness}"
            # plot_data_with_sample_0(
            #     f"{simulation.dir_path}/dataset_y_test.csv",
            #     f"{simulation.dir_path}/predictions.csv",
            #     simulation.dir_path,
            #     True,
            #     plot_subtitle,
            # )


def parse_yaml(yaml_path):
    parameters = load_yaml(yaml_path)
    simulations = []
    sim_base = SimulationLegacy()
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
            simulation = SimulationLegacy()
            simulation.__dict__.update(sim_base.__dict__)
            for key, value in sim.items():
                simulation.__setattr__(key, value)
            simulations.append(simulation)

    return simulations


def load_yaml(yaml_path):
    parameters = None
    with open(yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    if parameters is None:
        raise ValueError

    return parameters
