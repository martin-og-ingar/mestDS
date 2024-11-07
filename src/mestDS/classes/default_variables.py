from datetime import timedelta

from mestDS.classes.SimulationInput import RainSeason

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

TIMEDELTA = {"D": timedelta(days=1), "W": timedelta(weeks=1), "M": timedelta(weeks=4)}
DATEFORMAT = "%Y-%m-%d"

DEFAULT_RAIN_SEASON = [RainSeason(start=12, end=23), RainSeason(start=36, end=40)]

DEFAULT_REGIONS = ["Masindi", "Apac"]
