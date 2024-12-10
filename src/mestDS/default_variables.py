from .classes.RainSeason import RainSeason
from .classes.Region import Region
import datetime

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
DEFAULT_RAIN_SEASON = [RainSeason(start=12, end=23), RainSeason(start=36, end=40)]
DEFAULT_REGIONS = [
    Region("Masadi", 1, DEFAULT_RAIN_SEASON, [2]),
    Region("Kasama", 2, DEFAULT_RAIN_SEASON, [1]),
]
