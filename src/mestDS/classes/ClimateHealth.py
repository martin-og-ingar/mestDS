import datetime
import inspect
import random
from typing import Dict, Literal

from sympy import sympify
import yaml
import numpy as np

from mestDS.classes import RainSeason

from .Feature import Feature
from .Region import Region
from mestDS.classes.ClimateHealthData import Obs
import matplotlib.pyplot as plt


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)


class List:
    name: str
    data: list[float]


class ClimateHealth:
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

    def simulate(self):
        self.initialize_data()
        # For each day/week/month, simulate the features and calculate the sickness for each region.
        for i in range(1, self.simulation_length):
            self.current_i = i
            for region in self.regions:
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
        parameters = {}
        for param in feature.parameters:
            parameters[param] = self.get_feature(param)
        if hasattr(feature, "calculation"):
            result = eval(
                feature.calculation,
                {"np": np, "i": self.current_i, "parameters": parameters},
            )
        if hasattr(feature, "function"):
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

            args = [parameters.get(param) for param in feature.parameters]
            for param in parameters_required:
                if param == "region":
                    args.append(self.current_region)
                if param == "i":
                    args.append(self.current_i)
            result = func(*args)

        self.data[self.current_region.name][feature.name].append(result)

    def get_feature(self, param):
        # could be simplified
        for feat in self.features:
            if feat.name == param:
                return self.data[self.current_region.name][param]
        for lst in self.lists:
            if lst.name == param:
                return lst.data
        return []

    def initialize_data(self):
        self.data = {region.name: {} for region in self.regions}
        for region in self.regions:
            self.data[region.name] = {}
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


class MultipleClimateHealth:
    simulations: list[ClimateHealth]

    def __init__(self, dsl_path):
        self.simulations = parse_yaml(dsl_path)

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()


def parse_yaml(yaml_path):
    parameters = load_yaml(yaml_path)
    sim_base = ClimateHealth()
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
    return [sim_base]


def load_yaml(yaml_path):
    parameters = None
    with open(yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    if parameters is None:
        raise ValueError

    return parameters
