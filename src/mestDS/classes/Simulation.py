import os
from typing import Dict, Literal
import datetime

from matplotlib import pyplot as plt

import numpy as np

from mestDS.utils.main import train_test_split_csv
from .ClimateHealthData import Obs
from .RainSeason import RainSeason
from .Region import Region
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from ..default_variables import (
    DEFAULT_RAIN_SEASON,
    DEFAULT_REGIONS,
    DEFAULT_TEMPERATURES,
)
import csv


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Simulation:
    """
    Class for initializing simulation parameters with default values.
    It also contains two function for simulating data and evaluating the simulated data
    on a selected model.

    : param time_granularity: The time granularity of the simulation. Default is "D" for daily.
    : param simulation_length: The length of the simulation. Default is 500.
    : param simulation_start_date: The start date of the simulation. Default is 2024-01-01.
    : param rain_season: The rain season of the simulation. Default is DEFAULT_RAIN_SEASON.
    : param temperatures: The temperatures of the simulation. Default is DEFAULT_TEMPERATURES.
    : param regions: The regions of the simulation. Default is DEFAULT_REGIONS.

    New sickness values per iteration in the simulation is calculated as follows:
        dot = dot product of (rain, temperature) and (rain weight, temperature weight)
        max_dot = highest possible dot product
        sickness = sickness + np.random.normal((dot / max_dot) - normal_dist_mean, normal_dist_stddev) * nomral_dist_scale)

    : param normal_dist_mean: The mean of the normal distribution. Default is 0.5.
    : param normal_dist_stddev: The standard deviation of the normal distribution. Default is 0.3.
    : param normal_dist_scale: The scale of the normal distribution. Default is 10.

    : param simulated_data: The simulated data. Default is None.

    """

    time_granularity: Literal["D", "W", "M"]
    simulation_length: int
    simulation_start_date: datetime.date
    rain_season_randomness: bool
    temperatures: list[float]
    regions: list[Region]
    simulated_data: Dict[str, list[Obs]]
    beta_rainfall: float
    beta_temp: float
    beta_lag_sickness: float
    beta_neighbour_influence: float
    neighbors: np.ndarray
    noise_std: float

    def __init__(
        self,
        time_granularity="D",
        simulation_length=500,
        simulation_start_date=datetime.date(2024, 1, 1),
        temperatures=DEFAULT_TEMPERATURES,
        regions=DEFAULT_REGIONS,
        beta_rainfall=0.5,
        beta_temp=0.5,
        beta_lag_sickness=0.5,
        beta_neighbour_influence=0.9,
        neighbors=np.array([[0, 1], [1, 0]]),
        noise_std=1,
    ):
        self.time_granularity = time_granularity
        self.simulation_length = simulation_length
        self.simulation_start_date = simulation_start_date
        self.temperatures = temperatures
        self.regions = regions
        self.beta_rainfall = beta_rainfall
        self.beta_temp = beta_temp
        self.beta_lag_sickness = beta_lag_sickness
        self.beta_neighbour_influence = beta_neighbour_influence

        self.neighbors = neighbors
        self.noise_std = noise_std
        self.simulated_data = None

    def simulate(self):
        from ..simulate import generate_data

        beta_values = [
            self.beta_rainfall,
            self.beta_temp,
            self.beta_lag_sickness,
            self.beta_neighbour_influence,
        ]
        beta_values = softmax(beta_values)
        self.beta_rainfall = beta_values[0]
        self.beta_temp = beta_values[1]
        self.beta_lag_sickness = beta_values[2]
        self.beta_neighbour_influence = beta_values[3]

        self.simulated_data = generate_data(self)

    def chap_evaluation_on_model(self, model, prediction_lenght=5):
        data = DataSet.from_period_observations(self.simulated_data)
        evaluate_model(
            model, data, report_filename="test.pdf", prediction_length=prediction_lenght
        )

    def graph(
        self,
        show_rain=False,
        show_temperature=False,
        show_sickness=True,
        file_name=None,
        title="",
    ):
        from mestDS.default_variables import DATEFORMAT, TIMEDELTA

        num_plots = sum([show_rain, show_temperature, show_sickness])
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        dates = [
            self.simulation_start_date + i * TIMEDELTA[self.time_granularity]
            for i in range(self.simulation_length)
        ]

        for region, observations in self.simulated_data.items():
            if show_rain:
                rainfall = [obs.rainfall for obs in observations]
                axes[0].plot(dates, rainfall, label=f"{region} - Rainfall")
                axes[0].set_ylabel("Rainfall (mm)")
                axes[0].legend()
                axes[0].set_title("Rainfall Over Time")

            if show_temperature:
                temperatures = [obs.mean_temperature for obs in observations]
                temp_axis = axes[1] if show_rain else axes[0]
                temp_axis.plot(dates, temperatures, label=f"{region} - Temperature")
                temp_axis.set_ylabel("Temperature (°C)")
                temp_axis.legend()
                temp_axis.set_title("Temperature Over Time")

            if show_sickness:
                cases = [obs.disease_cases for obs in observations]
                sick_axis = (
                    axes[2]
                    if show_rain and show_temperature
                    else axes[1] if show_rain or show_temperature else axes[0]
                )
                sick_axis.plot(dates, cases, label=f"{region} - Disease Cases")
                sick_axis.set_ylabel("Disease Cases")
                sick_axis.legend()
                sick_axis.set_title("Sickness Over Time")

        for ax in axes:
            ax.set_xlabel("Date")
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(DATEFORMAT))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
            ax.grid(True)
        plt.title(title)
        plt.tight_layout()
        if file_name:
            plt.savefig(file_name)
        else:
            plt.show()

        plt.close(fig)

    def simulated_data_to_csv(self, dir_path, file_name):
        self.dir_path = dir_path
        os.makedirs(dir_path, exist_ok=True)
        self.dataset_file_path = os.path.join(dir_path, file_name)

        header = [
            "time_period",
            "rainfall",
            "mean_temperature",
            "disease_cases",
            "location",
        ]
        rows = []
        for region in self.regions:
            for obs in self.simulated_data[region.name]:
                row = []
                row.append(obs.time_period)
                row.append(obs.rainfall)
                row.append(obs.mean_temperature)
                row.append(obs.disease_cases)
                row.append(region.name)
                rows.append(row)
        with open(self.dataset_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

    def split_csv_to_train_and_test(self, test_size=0.2):
        train_test_split_csv(self.dataset_file_path, self.dir_path, test_size)
