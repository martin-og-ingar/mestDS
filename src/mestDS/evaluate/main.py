from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.predictor.model_registry import registry
from ..classes.ClimateHealthData import to_gluonTS_format


def evaluate_chap_model(datasets):
    model = registry.get_model("chap_ewars_monthly")
    for dataset in datasets:
        data = DataSet.from_period_observations(dataset)
        evaluate_model(model, data, report_filename="test.pdf")
