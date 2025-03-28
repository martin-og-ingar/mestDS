import copy
import csv
import datetime
from dateutil.relativedelta import relativedelta
import inspect
import os
import subprocess
from typing import Dict, Literal
import yaml
import numpy as np
import matplotlib.pyplot as plt


from ..default_variables import DATEFORMAT, TIMEDELTA
from ..utils import generate_report, generate_report_v2, train_test_split_csv
from .Feature import Feature
from .Region import Region
from .RainSeason import RainSeason


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)


class List:
    name: str
    data: list[float]


class Simulation:
    time_granularity: Literal["D", "W", "M"]
    simulation_length: int
    simulation_start_date: datetime.date
    regions: list[Region]
    data: Dict[str, Dict[str, list[float]]]
    features: list[Feature]
    current_i: int
    current_region: str
    simulation_name: str
    full_set_path: str
    train_set_x_path: str
    train_set_y_path: str
    test_set_x_path: str
    test_set_y_path: str

    def simulate(self):
        self.simulation_start_date = datetime.date(2024, 1, 1)
        self.initialize_data()
        # For each day/week/month, simulate the features and calculate the sickness for each region.
        delta = TIMEDELTA[self.time_granularity]
        for i in range(1, self.simulation_length):

            if self.time_granularity == "W":
                current_date = self.simulation_start_date + datetime.timedelta(weeks=i)
                iso_year, iso_week, _ = current_date.isocalendar()
                current_date_str = f"{iso_year}W{iso_week}"  # Correct week format
            elif self.time_granularity == "M":
                current_date = self.simulation_start_date + relativedelta(months=i)
                current_date_str = current_date.strftime(
                    DATEFORMAT[self.time_granularity]
                )
            else:
                current_date = self.simulation_start_date + (i * delta)
                current_date_str = current_date.strftime(
                    DATEFORMAT[self.time_granularity]
                )

            self.current_i = i
            for region in self.regions:
                self.data[region.name]["time_period"].append(current_date_str)
                self.current_region = region
                for feature in self.features:
                    self.calculate_feature(feature)

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
            if self.time_granularity == "W":
                current_date = self.simulation_start_date + datetime.timedelta(weeks=0)
                iso_year, iso_week, _ = current_date.isocalendar()
                current_date_str = f"{iso_year}W{iso_week}"
                self.data[region.name]["time_period"] = [current_date_str]
            else:
                self.data[region.name]["time_period"] = [
                    datetime.datetime.strftime(
                        self.simulation_start_date, DATEFORMAT[self.time_granularity]
                    )
                ]

            for feature in self.features:
                self.data[region.name][feature.name] = [0]

    def plot_data(self):
        regions = self.data.keys()
        variables = [
            feature.name
            for feature in self.features
            if feature.name != "lagged_sickness" and feature.name != "white_noise"
        ]
        for region in regions:
            plt.figure(figsize=(10, 6))

            for var in variables:
                plt.plot(self.data[region][var], label=f"{region} - {var}")

            plt.title(self.simulation_name)
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def convert_to_csv(self, file_path):

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        csv_rows = []
        columns = ["time_period"]
        columns += [feature.name for feature in self.features]
        columns.append("location")
        csv_rows.append(columns)

        for region in self.regions:
            for i in range(len(self.data[region.name][self.features[0].name])):
                row = []
                row.append(self.data[region.name]["time_period"][i])
                for feature in self.features:
                    row.append(self.data[region.name][feature.name][i])
                row.append(region.name)
                csv_rows.append(row)

        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_rows)
