from pathlib import Path
from mestDS.utils import get_model_from_directory_or_github_url

from chap_core.external.external_model import (
    get_model_from_directory_or_github_url,
)
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor
from chap_core.assessment.prediction_evaluator import backtest, evaluate_model
from chap_core.assessment.dataset_splitting import (
    train_test_generator,
)


class BackTestEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, simulation):
        self.initialize_model()
        self.initialize_data(simulation, [])

        evaluate_model(
            self.model,
            self.full_ds,
            self.config.get("prediction_length"),
            4,
            "reports/testing.pdf",
        )
        # predictions_list = list(self.backtest_with_stride())
        # print(predictions_list)

    def initialize_model(self):
        model_path = self.config.get("model")
        model = get_model_from_directory_or_github_url(
            model_path, base_working_dir=Path("runs")
        )
        self.model = model
        self.working_dir = f"{self.model._working_dir}"
        self.is_external_chap_model = True

    def initialize_data(self, simulation, exclude_feature):
        time_granularity = self.config.get("time_granularity")
        if time_granularity is not None:
            simulation.time_granularity = time_granularity
        simulation.simulate()
        self.dataset_file = (
            f"{self.working_dir}/{simulation.simulation_name}_dataset.csv"
        )
        simulation.convert_to_csv(self.dataset_file)

        self.full_ds = DataSet.from_csv(self.dataset_file, FullData)

        self.predictions_file = f"{simulation.simulation_name}_predictions.csv"

    def backtest_with_stride(self):
        pred_len = self.config.get("prediction_length") or 12
        train, test_generator = train_test_generator(
            self.full_ds, pred_len, 10, stride=pred_len
        )
        predictor = self.model.train(train)
        for historic_data, future_data, _ in test_generator:
            yield predictor.predict(historic_data, future_data)
