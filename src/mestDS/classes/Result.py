from dataclasses import dataclass

import matplotlib
import matplotlib.figure
from .LossMetrics import LossMetrics


@dataclass
class Result:
    name: str
    description: str
    plots: list[matplotlib.figure.Figure]
    location_metrics: list[LossMetrics]
