# from mestDS.default_variables import DEFAULT_RAIN_SEASON
from .RainSeason import RainSeason


class Region:
    name: str
    region_id: int
    rain_season: list[RainSeason]
    neighbour: list[int]

    def __init__(self, name="Masadi", region_id=1, rain_season=[], neighbour=[2]):
        self.name = name
        self.region_id = region_id
        self.rain_season = rain_season
        self.neighbour = neighbour
