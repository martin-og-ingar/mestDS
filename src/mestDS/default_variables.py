from .classes.Region import Region
from datetime import timedelta
from dateutil.relativedelta import relativedelta


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
    "D": timedelta(days=1),
    "W": timedelta(weeks=1),
    "M": relativedelta(months=1),
}
DATEFORMAT = {"D": "%Y-%m-%d", "W": "%G-W%V", "M": "%Y-%m"}
DEFAULT_REGIONS = [
    Region(
        "Masadi",
        1,
    ),
]

DEFAULT_NUMBER_OF_FOLDS = 5
