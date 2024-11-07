from chap_core.data import DataSet, PeriodObservation, adaptors


class ClimatHealthData:
    precipitation: list[float]
    temperature: list[float]
    sickness: list[float]

    def __init__(
        self,
        precipitation: list[float],
        temperature: list[float],
        sickness: list[float],
    ):
        self.precipitation = precipitation
        self.temperature = temperature
        self.sickness = sickness


class Obs(PeriodObservation):
    population: int
    disease_cases: int
    rainfall: float
    mean_temperature: float


def to_dataset_format(dict):
    return DataSet.from_period_observations(dict)


def to_gluonTS_format(data_set):
    return adaptors.gluonts.from_dataset(data_set)
