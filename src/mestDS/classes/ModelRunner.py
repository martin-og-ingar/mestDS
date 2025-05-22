import os
from pathlib import Path

from chap_core.external.external_model import (
    get_model_from_directory_or_github_url,
)
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.assessment.dataset_splitting import (
    train_test_generator,
)

from mestDS.classes.Result import Result
from mestDS.utils import get_forecast_dicts, get_metrics, get_plots


class ModelRunner:
    def __init__(self, model_path, prediction_length, n_test_sets, stride, metrics):
        self.model_path = model_path
        self.model = get_model_from_directory_or_github_url(
            model_path, base_working_dir=Path("runs"), run_dir_type="use_existing"
        )
        self.prediction_length = prediction_length
        self.n_test_sets = n_test_sets
        self.stride = stride
        self.metrics = metrics

    def run(self, simulation=None, filename=None):
        if simulation:
            filename = f"{self.model._working_dir}/{simulation.name}.csv"
            simulation.to_csv(filename)
            name = simulation.name
        else:
            name = os.path.splitext(os.path.basename(filename))[0]

        dataset = DataSet.from_csv(filename, FullData)

        train, test_generator = train_test_generator(
            dataset, self.prediction_length, self.n_test_sets, stride=self.stride
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
            name,
            get_plots(dataset, forecast_dicts),
            get_metrics(test_ds, forecast_dicts, self.metrics),
        )
