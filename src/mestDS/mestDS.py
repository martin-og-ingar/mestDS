import copy
import os
import subprocess
import yaml

from mestDS.classes.Feature import Feature
from mestDS.classes.RainSeason import RainSeason
from mestDS.classes.Region import Region
from mestDS.classes.evaluation.EvaluatorGenerator import EvaluatorGenerator
from mestDS.classes.Simulation import Simulation

from mestDS.utils import generate_report, train_test_split_csv


class mestDS:

    simulations: list[Simulation]
    evaluators: list
    is_converted_to_csvs: bool
    folder_path: str

    def __init__(self, dsl_path):
        self.simulations, self.evaluators = parse_yaml(dsl_path)
        self.is_converted_to_csvs = False

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()

    def convert_to_csvs(self, folder_path):
        self.folder_path = folder_path
        for i, simulation in enumerate(self.simulations):
            os.makedirs(
                os.path.dirname(f"{folder_path}{simulation.simulation_name}/"),
                exist_ok=True,
            )
            file_path = f"{folder_path}{simulation.simulation_name}/dataset.csv"
            simulation.convert_to_csv(file_path)
        self.is_converted_to_csvs = True

    def csv_train_test_split(
        self, split_train=False, exclude_features=[], test_size=0.2
    ):
        for i, simulation in enumerate(self.simulations):
            train_test_split_csv(
                f"{self.folder_path}{simulation.simulation_name or i}/dataset.csv",
                f"{self.folder_path}{simulation.simulation_name or i}/",
                exclude_feature=exclude_features,
                split_train=split_train,
                test_size=test_size,
                time_granularity=simulation.time_granularity,
            )

    def eval_chap_model(self, model_name, exclude_features=[]):
        self.csv_train_test_split(exclude_features)

        for simulation in self.simulations:
            train_command = [
                "python",
                f"{model_name}/train.py",
                f"{self.folder_path}{simulation.simulation_name}/dataset_train.csv",
                f"{self.folder_path}{simulation.simulation_name}/model.bin",
            ]
            subprocess.run(train_command, check=True)

            test_command = [
                "python",
                f"{model_name}/predict.py",
                f"{self.folder_path}{simulation.simulation_name}/model.bin",
                f"{self.folder_path}{simulation.simulation_name}/dataset_train.csv",
                f"{self.folder_path}{simulation.simulation_name}/dataset_x_test.csv",
                f"{self.folder_path}{simulation.simulation_name}/predictions.csv",
            ]

            subprocess.run(test_command, check=True)

            generate_report(
                simulation,
                self.folder_path,
                model_name=model_name,
            )

    def plot_data(self):
        for sim in self.simulations:
            sim.plot_data()

    def evaluate(self):
        for evaluator in self.evaluators:
            for simulation in self.simulations:
                evaluator.evaluate(simulation)


def parse_yaml(yaml_path):
    parameters = load_yaml(yaml_path)
    sim_base = Simulation()
    base = parameters.get("model", {})
    for key, value in base.items():
        if key == "regions":
            regions = []
            for reg in value:
                region = Region()
                for key, value in reg.items():
                    if key == "rain_season":
                        rain_season = []
                        for season in value:
                            rain_season.append(RainSeason(season[0], season[1]))
                        region.__setattr__(key, rain_season)
                    else:
                        region.__setattr__(key, value)
                regions.append(region)
            sim_base.regions = regions
        if key == "features":
            features = []
            for feat in value:
                feature = Feature()
                for key, value in feat.items():
                    feature.__setattr__(key, value)
                features.append(feature)
            sim_base.features = features
        else:
            sim_base.__setattr__(key, value)

    simulations = parameters.get("simulations", {})
    sims = [sim_base]
    for simulation in simulations:
        sim = copy.deepcopy(sim_base)
        for key, value in simulation.items():
            if key == "features":
                for feat in value:
                    feat_name = feat.get("name")
                    if feat_name is None:
                        raise ValueError
                    index = next(
                        (
                            i
                            for i, feature in enumerate(sim.features)
                            if feat_name == feature.name
                        )
                    )
                    sim.features[index].function = feat.get("function", {})
            else:
                sim.__setattr__(key, value)
        sims.append(sim)

    evaluators = parameters.get("evaluators", {})
    evals = []
    for evaluator in evaluators:
        eval = EvaluatorGenerator(evaluator).create_evaluator()
        evals.append(eval)

    return sims, evals


def load_yaml(yaml_path):
    parameters = None
    with open(yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    if parameters is None:
        raise ValueError

    return parameters
