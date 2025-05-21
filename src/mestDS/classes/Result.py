from dataclasses import dataclass

import matplotlib
import matplotlib.figure
from .LossMetrics import LossMetrics


@dataclass
class Result:
    name: str
    plots: list[matplotlib.figure.Figure]
    metrics: list[LossMetrics]
