from dataclasses import dataclass


@dataclass
class LossMetrics:
    location: str
    metrics: dict[str, float]
