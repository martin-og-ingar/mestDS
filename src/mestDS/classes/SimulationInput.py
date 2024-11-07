from typing import Dict, List, Literal
import datetime
from .ClimateHealthData_module import Obs
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

DEFAULT_TEMPERATURES = [
    23.72,
    24.26,
    24.25,
    23.71,
    23.18,
    22.67,
    22.31,
    22.68,
    22.86,
    23.16,
    23.21,
    23.03,
]

TIMEDELTA = {
    "D": datetime.timedelta(days=1),
    "W": datetime.timedelta(weeks=1),
    "M": datetime.timedelta(weeks=4),
}
DATEFORMAT = "%Y-%m-%d"


DEFAULT_REGIONS = ["Masindi", "Apac"]


class RainSeason:
    start: int
    end: int

    def __init__(self, start, end):
        self.start = start
        self.end = end


DEFAULT_RAIN_SEASON = [RainSeason(start=12, end=23), RainSeason(start=36, end=40)]


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
    normal_dist_mean: float
    normal_dist_stddev: float
    nomral_dist_scale: float
    simulated_data: Dict[str, list[Obs]]

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
        from ..simulate import generate_data

        self.simulated_data = generate_data(self)

    def chap_evaluation_on_model(self, model, prediction_lenght=5):
        data = DataSet.from_period_observations(self.simulated_data)
        evaluate_model(
            model, data, report_filename="test.pdf", prediction_lenght=prediction_lenght
        )
