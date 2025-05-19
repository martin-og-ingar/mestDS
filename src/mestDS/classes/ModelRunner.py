from pathlib import Path
from chap_core.external.external_model import (
    get_model_from_directory_or_github_url,
)
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor
from chap_core.assessment.dataset_splitting import (
    train_test_generator,
)
from matplotlib import dates, pyplot as plt
import pandas as pd

from mestDS.classes.Result import Result
from mestDS.utils import convert_time_period, get_forecast_dicts, get_metrics, get_plots

# from ..utils import get_forecast_dicts


class ModelRunner:
    def __init__(self, model_path, prediction_length, n_test_sets, stride):
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.n_test_sets = n_test_sets
        self.stride = stride

    def run(self, simulation):
        self.model = get_model_from_directory_or_github_url(
            self.model_path, base_working_dir=Path("runs")
        )
        filename = f"{self.model._working_dir}/{simulation.simulation_name}.csv"

        simulation.convert_to_csv(filename)
        full_ds = DataSet.from_csv(filename, FullData)

        train, test_generator = train_test_generator(
            full_ds, self.prediction_length, self.n_test_sets, stride=self.stride
        )
        predictor = self.model.train(train)

        forecasts = []
        test_ds = []
        for historic_data, future_data, future_disease_cases in test_generator:
            forecast = predictor.predict(historic_data, future_data)
            forecasts.append(forecast)
            test_ds.append(future_disease_cases)
        forecast_dicts = get_forecast_dicts(forecasts)
        return Result(
            simulation.simulation_name,
            get_plots(full_ds, forecast_dicts),
            get_metrics(test_ds, forecast_dicts),
        )
