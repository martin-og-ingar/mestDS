import copy
import csv
import datetime
import inspect
import os
import random
import subprocess
from typing import Dict, Literal

from sympy import sympify
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


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)


class List:
    name: str
    data: list[float]


class Simulation:
    time_granularity: Literal["D", "W", "M"]
    simulation_length: int
    simulation_start_date: datetime.date
    temperatures: list[float]
    regions: list[Region]
    data: Dict[str, Dict[str, list[float]]]
    features: list[Feature]
    lists: list[List]
    current_i: int
    current_region: str
    simulation_name: str

    def simulate(self):
        self.simulation_start_date = datetime.date(2024, 1, 1)
        self.initialize_data()
        # For each day/week/month, simulate the features and calculate the sickness for each region.
        delta = TIMEDELTA[self.time_granularity]
        for i in range(1, self.simulation_length):
            current_date = self.simulation_start_date + (i * delta)
            current_date = datetime.datetime.strftime(current_date, DATEFORMAT)
            self.current_i = i
            for region in self.regions:
                self.data[region.name]["time_period"].append(current_date)
                self.current_region = region
                for feature in self.features:
                    self.calculate_feature(feature)
        self.plot_data()

    def adjust_beta(self):
        beta_values = [feature.beta for feature in self.features]
        beta_values_adjusted = softmax(beta_values)
        for i, beta in enumerate(beta_values_adjusted):
            self.features[i].beta = beta

    def calculate_feature(self, feature: Feature):
        local_context = {}
        exec(
            feature.function,
            {"np": np, "i": self.current_i, "region": self.current_region},
            local_context,
        )
        func_name = list(local_context.keys())[0]
        func = local_context[func_name]
        signature = inspect.signature(func)
        parameters_required = signature.parameters

        args = [self.get_feature(param) for param in parameters_required]
        result = func(*args)

        self.data[self.current_region.name][feature.name].append(result)

    def get_feature(self, param):
        if param == "region":
            return self.current_region
        if param == "i":
            return self.current_i
        for feat in self.features:
            if feat.name == param:
                return self.data[self.current_region.name][param]
        return None

    def initialize_data(self):
        self.data = {region.name: {} for region in self.regions}
        for region in self.regions:
            self.data[region.name] = {}
            self.data[region.name]["time_period"] = [
                datetime.datetime.strftime(self.simulation_start_date, DATEFORMAT)
            ]

            for feature in self.features:
                self.data[region.name][feature.name] = [25]

    def plot_data(self):
        regions = self.data.keys()
        variables = [
            feature.name
            for feature in self.features
            if feature.name != "lagged_sickness"
        ]
        for region in regions:
            if region == "Troms":
                plt.figure(figsize=(10, 6))

                for var in variables:
                    plt.plot(self.data[region][var], label=f"{region} - {var}")

                plt.xlabel("Time")
                plt.ylabel("Values")
                plt.legend()
                plt.tight_layout()
                plt.show()

    def convert_to_csv(self, file_path):
        csv_rows = []
        columns = [feature.name for feature in self.features]
        columns.append("location")
        csv_rows.append(columns)

        for region in self.regions:
            for i in range(len(self.data[region.name][self.features[0].name])):
                row = []
                for feature in self.features:
                    row.append(self.data[region.name][feature.name][i])
                row.append(region.name)
                csv_rows.append(row)
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_rows)


class Simulations:
    simulations: list[Simulation]
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
                "python",
                f"{model_name}/train.py",
                f"{self.folder_path}{simulation.simulation_name}/dataset_train.csv",
                f"{self.folder_path}{simulation.simulation_name}/model.bin",
            ]
            subprocess.run(train_command, check=True)

            test_command = [
                "python",
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
    sim_base = Simulation()
    base = parameters.get("model", {})
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
                    print(type(sim.features[0]))
                    feat_name = feat.get("name")
                    if feat_name is None:
                        raise ValueError
                    index = next(
                        (
                            i
                            for i, feature in enumerate(sim.features)
                            if feat_name == feature.name
                        )
                    )
                    sim.features[index].function = feat.get("function", {})
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
