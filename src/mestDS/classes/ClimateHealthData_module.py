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
