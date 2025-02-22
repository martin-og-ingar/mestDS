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
    real_data: list[float]

    def simulate(self):
        self.simulation_start_date = datetime.date(2024, 1, 1)
        self.initialize_data()
        self.real_data = []

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
        if func == "autoregression":
            history = self.data[self.current_region.name][feature_name]
            params["history"] = history
        else:
            history = None
        params = mod.get("params", {})
        if func == "realistic_data_generation":
            if (
                isinstance(self.real_data, (list, np.ndarray))
                and len(self.real_data) == 0
            ):
                self.real_data = self.data[self.current_region.name][feature_name]

                self.data[self.current_region.name][feature_name] = mod_func(
                    **params, t=self.simulation_length, current_i=self.current_i
                )

        else:
            if func == "climate_dependent_disease_cases":
                params["rainfall"] = self.data[self.current_region.name]["rainfall"][
                    self.current_i
                ]
                params["temperature"] = self.data[self.current_region.name][
                    "temperature"
                ][self.current_i]
            modified_data = mod_func(
                **params,
                t=self.simulation_length,
                current_i=self.current_i,
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
                plt.legend()
                plt.tight_layout()
                plt.show()


class SimulationsDemo:
    simulations: list[SimulationDemo]
    folder_path: str

    def __init__(self, dsl_path, folder_path=""):
        self.simulations = parse_yaml(dsl_path)
        self.folder_path = folder_path

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()


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
