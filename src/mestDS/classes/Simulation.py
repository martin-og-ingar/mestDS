from typing import Dict, Literal
import datetime

from mestDS.visualize.main import graph
from .ClimateHealthData import Obs
from .RainSeason import RainSeason
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from ..default_variables import (
    DEFAULT_RAIN_SEASON,
    DEFAULT_REGIONS,
    DEFAULT_TEMPERATURES,
)
import csv


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
    rain_season: list[RainSeason]
    rain_season_randomness: bool
    temperatures: list[float]
    regions: list[str]
    simulated_data: Dict[str, list[Obs]]
    beta_rainfall: float
    beta_temp: float
    beta_lag_sickness: float
    noise_std: float

    # normal_dist_mean: float
    # normal_dist_stddev: float
    # nomral_dist_scale: float

    def __init__(
        self,
        time_granularity="D",
        simulation_length=500,
        simulation_start_date=datetime.date(2024, 1, 1),
        rain_season=None,
        temperatures=DEFAULT_TEMPERATURES,
        regions=DEFAULT_REGIONS,
        # normal_dist_mean=0.5,
        # normal_dist_stddev=0.3,
        # normal_dist_scale=10,
        beta_rainfall=0.2,
        beta_temp=0.9,
        beta_lag_sickness=0.8,
        noise_std=0.3,
    ):
        self.time_granularity = time_granularity
        self.simulation_length = simulation_length
        self.simulation_start_date = simulation_start_date
        self.rain_season = rain_season or DEFAULT_RAIN_SEASON
        self.temperatures = temperatures
        self.regions = regions

        self.beta_rainfall = beta_rainfall
        self.beta_temp = beta_temp
        self.beta_lag_sickness = beta_lag_sickness
        self.noise_std = noise_std
        self.simulated_data = None

        # self.normal_dist_mean = normal_dist_mean
        # self.normal_dist_stddev = normal_dist_stddev
        # self.nomral_dist_scale = normal_dist_scale

    def simulate(self):
        from ..simulate import generate_data

        self.simulated_data = generate_data(self)

    def chap_evaluation_on_model(self, model, prediction_lenght=5):
        data = DataSet.from_period_observations(self.simulated_data)
        evaluate_model(
            model, data, report_filename="test.pdf", prediction_length=prediction_lenght
        )

    def show_graph(self, show_rain=False, show_temperature=False, show_sickness=True):
        graph(self, show_rain, show_temperature, show_sickness)

    def simulated_data_to_csv(self, filepath):
        header = [
            "time_period",
            "rainfall",
            "mean_temperature",
            "disease_cases",
            "location",
        ]
        rows = []
        for region in self.regions:
            for obs in self.simulated_data[region]:
                row = []
                row.append(obs.time_period)
                row.append(obs.rainfall)
                row.append(obs.mean_temperature)
                row.append(obs.disease_cases)
                row.append(region)
                rows.append(row)
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
