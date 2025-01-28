# from mestDS.default_variables import DEFAULT_RAIN_SEASON
from .RainSeason import RainSeason


class Region:
    name: str
    region_id: int
    rain_season: list[RainSeason]
    neighbour: list[int]

    def __init__(self, name="", region_id=0, rain_season=[], neighbour=[]):
        self.name = name
        self.region_id = region_id
        self.rain_season = rain_season
        self.neighbour = neighbour
