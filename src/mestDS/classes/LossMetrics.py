from dataclasses import dataclass


@dataclass
class LossMetrics:
    location_name: str
    mse: float
    pocid: float
    thiels_u: float
