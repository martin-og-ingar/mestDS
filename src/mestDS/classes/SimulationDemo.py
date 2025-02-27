import copy
import csv
import datetime
import os
import sys
import subprocess
from typing import Dict, Literal

import yaml
import numpy as np

from mestDS.classes import RainSeason
from mestDS.default_variables import DATEFORMAT, TIMEDELTA
from mestDS.utils.main import plot_data_with_sample_0, train_test_split_csv
from scripts import simulation

from .Feature import Feature
from .Region import Region
from mestDS.classes.ClimateHealthData import Obs
import matplotlib.pyplot as plt
from mestDS.FunctionPool import *


class List:
    name: str
    data: list[float]


class SimulationDemo:
    time_granularity: Literal["D", "W", "M"]
    simulation_length: int
    simulation_start_date: datetime.date
    temperatures: list[float]
    regions: list[Region]
    data: Dict[str, Dict[str, list[float]]]
    features: list[Feature]
    baseline_func: str
    lists: list[List]
    current_i: int
    current_region: str
    simulation_name: str
    history: list[float]
    real_data: dict[str, list[float]]

    def simulate(self):
        self.simulation_start_date = datetime.date(2024, 1, 1)
        self.initialize_data()
        self.real_data = {}

        delta = TIMEDELTA[self.time_granularity]
        for i in range(1, self.simulation_length):
            current_date = self.simulation_start_date + (i * delta)
            current_date = datetime.datetime.strftime(current_date, DATEFORMAT)
            self.current_i = i
            for region in self.regions:
                self.data[region.name]["time_period"].append(current_date)
                self.current_region = region
                for feature in self.features:
                    for mod in feature.modification:

                        self.calculate_feature(feature.name, mod)
        self.plot_data()

    def initialize_data(self):
        self.data = {region.name: {} for region in self.regions}
        t = np.arange(self.simulation_length)

        base_func = FUNCTION_POOL.get(self.baseline_func, lambda t: np.ones_like(t) * 1)
        for region in self.regions:
            self.data[region.name]["time_period"] = [
                datetime.datetime.strftime(self.simulation_start_date, DATEFORMAT)
            ]

            for feature in self.features:
                self.data[region.name][feature.name] = base_func(t)

    def calculate_feature(self, feature_name, mod: Feature):

        func = mod.get("function")
        mod_func = FUNCTION_POOL.get(func)
        params = mod.get("params", {})
        params["t"] = self.simulation_length
        params["current_i"] = self.current_i
        if func == "autoregression":
            history = self.data[self.current_region.name][feature_name]
            params["history"] = history
        else:
            history = None
        if func == "realistic_data_generation":
            # if (
            #     isinstance(self.real_data, (list, np.ndarray))
            #     and len(self.real_data) == 0
            # ):
            if feature_name not in self.real_data:
                self.real_data[feature_name] = (
                    self.data[self.current_region.name].get(feature_name, []).copy()
                )
                self.data[self.current_region.name][feature_name] = mod_func(**params)
        else:
            if func == "climate_dependent_disease_cases":
                params["rainfall"] = self.data[self.current_region.name]["rainfall"]
                params["temperature"] = self.data[self.current_region.name][
                    "temperature"
                ]
            modified_data = mod_func(
                **params,
            )

            self.data[self.current_region.name][feature_name][
                self.current_i
            ] += modified_data

    def plot_data(self):
        regions = self.data.keys()
        variables = [
            feature.name
            for feature in self.features
            if feature.name != "lagged_sickness"
        ]
        for region in regions:
            if region == "Test":
                plt.figure(figsize=(10, 6))

                for var in variables:
                    plt.plot(self.data[region][var], label=f"{region} - {var}")

                plt.xlabel("Time")
                plt.ylabel("Values")
                plt.title(self.simulation_name)
                plt.legend()
                plt.tight_layout()
                plt.show()

    def convert_to_csv(self, file_path):
        csv_rows = []
        columns = list(self.data[self.current_region.name].keys())
        columns.append("location")
        csv_rows.append(columns)

        for region in self.regions:
            for i in range(len(self.data[region.name]["time_period"])):
                row = []
                row.append(self.data[region.name]["time_period"][i])
                for feature in self.features:
                    row.append(self.data[region.name][feature.name][i])
                row.append(region.name)
                csv_rows.append(row)
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_rows)


class SimulationsDemo:
    simulations: list[SimulationDemo]
    folder_path: str

    def __init__(self, dsl_path, folder_path=""):
        self.simulations = parse_yaml(dsl_path)
        self.folder_path = folder_path

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()

    def convert_to_csvs(self, folder_path):
        self.folder_path = folder_path
        for i, simulation in enumerate(self.simulations):
            os.makedirs(
                os.path.dirname(f"{folder_path}{simulation.simulation_name}/"),
                exist_ok=True,
            )
            file_path = f"{folder_path}{simulation.simulation_name}/dataset.csv"
            simulation.convert_to_csv(file_path)

    def csv_train_test_split(self):
        for i, simulation in enumerate(self.simulations):
            train_test_split_csv(
                f"{self.folder_path}{simulation.simulation_name or i}/dataset.csv",
                f"{self.folder_path}{simulation.simulation_name or i}/",
            )

    def eval_chap_model(self, model_name):
        """
        This function
        """
        self.csv_train_test_split()

        for simulation in self.simulations:
            train_command = [
                sys.executable,
                f"{model_name}/train.py",
                f"{self.folder_path}{simulation.simulation_name}/dataset_train.csv",
                f"{self.folder_path}{simulation.simulation_name}/model.bin",
            ]
            subprocess.run(train_command, check=True)

            test_command = [
                sys.executable,
                f"{model_name}/predict.py",
                f"{self.folder_path}{simulation.simulation_name}/model.bin",
                "",
                f"{self.folder_path}{simulation.simulation_name}/dataset_x_test.csv",
                f"{self.folder_path}{simulation.simulation_name}/predictions.csv",
            ]

            subprocess.run(test_command, check=True)
            plot_data_with_sample_0(
                f"{self.folder_path}{simulation.simulation_name}/dataset_y_test.csv",
                f"{self.folder_path}{simulation.simulation_name}/predictions.csv",
                f"{self.folder_path}{simulation.simulation_name}",
                True,
            )


def parse_yaml(yaml_path):
    parameters = load_yaml(yaml_path)
    sim_base = SimulationDemo()
    base = parameters.get("model", {})

    modifications = base.get("modifications", [])
    simulation_length = base.get("simulation_length", 0)
    base_func_name = base.get("baseline_func", "constant")

    sim_base.simulation_length = simulation_length
    sim_base.baseline_function = base_func_name
    sim_base.modifications = modifications

    for key, value in base.items():
        if key == "regions":
            regions = []
            for reg in value:
                region = Region()
                for key, value in reg.items():
                    if key == "rain_season":
                        rain_season = []
                        for season in value:
                            rain_season.append(RainSeason(season[0], season[1]))
                        region.__setattr__(key, rain_season)
                    else:
                        region.__setattr__(key, value)
                regions.append(region)
            sim_base.regions = regions

        if key == "features":
            features = []
            for feat in value:
                feature = Feature()
                for key, value in feat.items():
                    feature.__setattr__(key, value)
                features.append(feature)
            sim_base.features = features
        else:
            sim_base.__setattr__(key, value)
    simulations = parameters.get("simulations", {})
    sims = [sim_base]
    for simulation in simulations:
        sim = copy.deepcopy(sim_base)
        for key, value in simulation.items():
            if key == "features":
                for feat in value:
                    feat_name = feat.get("name")
                    if feat_name is None:
                        raise ValueError
                    base_feature = next(
                        (
                            feature
                            for feature in sim.features
                            if feature.name == feat_name
                        ),
                        None,
                    )

                    if base_feature:
                        base_feature_copy = copy.deepcopy(base_feature)
                        new_modification = feat.get("modification", [])

                        if new_modification:
                            base_feature_copy.modification = new_modification
                        else:
                            base_feature_copy.modification = base_feature.modification

                        sim.features = [
                            f if f.name != feat_name else base_feature_copy
                            for f in sim.features
                        ]
                    else:
                        sim.features.append(feat)
            else:
                sim.__setattr__(key, value)
        sims.append(sim)
    return sims


def load_yaml(yaml_path):
    parameters = None
    with open(yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    if parameters is None:
        raise ValueError

    return parameters
