class ClimatHealthData:
    precipitation: list[float]
    temperature: list[float]
    sickness: list[float]
    date: str = "2024-01-01"  # Default

    def __init__(
        self,
        precipitation: list[float],
        temperature: list[float],
        sickness: list[float],
        date: str = list[str],
    ):
        self.precipitation = precipitation
        self.temperature = temperature
        self.sickness = sickness
        self.date = date
