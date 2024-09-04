from climate_health.data import DataSet, PeriodObservation, adaptors


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
    disease_cases: int
    rainfall: float
    temperature: float


def toDataSetFromat(dict):
    return DataSet.from_period_observations(dict)


def toGluonTsFormat(data_set):
    return adaptors.gluonts.from_dataset(data_set)
