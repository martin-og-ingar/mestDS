from enum import Enum

from mestDS.classes.evaluation.BackTestEvaluator import BackTestEvaluator
from mestDS.classes.evaluation.CustomEvaluator import CustomEvaluator
from mestDS.classes.evaluation.HoldOutEvaluator import HoldOutEvaluator
from mestDS.classes.evaluation.CustomHoldOutEvaluator import CustomHoldOutEvaluator
from mestDS.classes.evaluation.TSCVEvaluator import TSCVEvaluator

from mestDS.default_variables import DEFAULT_NUMBER_OF_FOLDS


class EvaluationTechnique(Enum):
    backtest = "backtest"
    holdout = "holdout"


class EvaluatorGenerator:
    config: dict

    def __init__(self, config):
        self.config = config

    def create_evaluator(self):
        eval_technique = self.config.get("evaluation_technique")
        model = self.config.get("model")

        match eval_technique:
            case EvaluationTechnique.holdout.value:
                return HoldOutEvaluator(self.config)
            case EvaluationTechnique.backtest.value:
                return BackTestEvaluator(self.config)
            case _:
                raise Exception("A valid evaluation technique was not defined")
