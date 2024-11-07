from typing import Literal
import datetime
from .default_variables import (
    DEFAULT_RAIN_SEASON,
    DEFAULT_TEMPERATURES,
    DEFAULT_REGIONS,
)
from .ClimateHealthData_module import Obs
from ..simulate import generate_data


class RainSeason:
    start: int
    end: int

    def __init__(self, start, end):
        self.start = start
        self.end = end


class Simulation:
    time_granularity: Literal["D", "W", "M"]
    simulation_length: int
    simulation_start_date: datetime.date
    rain_season: list[RainSeason]
    rain_season_randomness: bool
    temperatures: list[float]
    regions: list[str]
    normal_dist_mean: float
    normal_dist_stddev: float
    nomral_dist_scale: float
    simulated_data: Obs

    def __init__(
        self,
        time_granularity="D",
        simulation_length=500,
        simulation_start_date=datetime.date(2024, 1, 1),
        rain_season=None,
        temperatures=DEFAULT_TEMPERATURES,
        regions=DEFAULT_REGIONS,
        normal_dist_mean=0.5,
        normal_dist_stddev=0.3,
        normal_dist_scale=10,
    ):
        self.time_granularity = time_granularity
        self.simulation_length = simulation_length
        self.simulation_start_date = simulation_start_date
        self.rain_season = rain_season or DEFAULT_RAIN_SEASON
        self.temperatures = temperatures
        self.regions = regions
        self.normal_dist_mean = normal_dist_mean
        self.normal_dist_stddev = normal_dist_stddev
        self.nomral_dist_scale = normal_dist_scale
        self.simulated_data = None

    def simulate(self):
        self.simulated_data = generate_data(self)

    def chap_evaluation_on_model(self, model):
        evaluate_model(model, self.simulated_data)
