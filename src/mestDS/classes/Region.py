from typing import Any


class Region:
    name: str
    region_id: int
    seasons: dict[str, Any]
    neighbour: list[int]

    def __init__(self, name="", region_id=0, seasons=None, neighbour=None):
        self.name = name
        self.region_id = region_id
        self.seasons = seasons if seasons is not None else {}
        self.neighbour = neighbour if neighbour is not None else []
