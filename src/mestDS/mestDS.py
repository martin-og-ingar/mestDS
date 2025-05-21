import copy
import os
import yaml

from mestDS.classes.Feature import Variable
from mestDS.classes.Region import Region
from mestDS.classes.evaluation.Evaluator import Evaluator
from mestDS.classes.Simulation import Simulation

from mestDS.utils import slugify


class mestDS:

    simulators: list[Simulation]
    evaluators: list
    is_converted_to_csvs: bool
    folder_path: str

    def __init__(self, dsl_path):
        self.simulators, self.evaluators = parse_yaml(dsl_path)

    def simulate(self):
        for simulator in self.simulators:
            simulator.simulate()

    def convert_to_csvs(self, folder_path):
        self.folder_path = folder_path
        for simulator in self.simulators:
            os.makedirs(
                os.path.dirname(f"{folder_path}{simulator.name}/"),
                exist_ok=True,
            )
            file_path = f"{folder_path}{simulator.name}/dataset.csv"
            simulator.convert_to_csv(file_path)

    def plot_data(self, folder=None, dont_show=[]):
        for simulator in self.simulators:
            if folder:
                simulator.plot_data(
                    dont_show, filename=f"{folder}{slugify(simulator.name)}.png"
                )

    def evaluate(self):
        for evaluator in self.evaluators:
            evaluator_copy = copy.deepcopy(evaluator)
            evaluator_copy.evaluate(self.simulators)


def parse_yaml(yaml_path):

    def set_x_variables(_x, sim):
        for x in _x:
            name = x.get("name")
            index = next(
                (i for i, variable in enumerate(sim.x) if name == variable.name),
                None,
            )
            if index is None:
                sim.x.append(Variable())
                index = len(sim.x) - 1
            for key, value in x.items():
                if key == "function":
                    sim.x[index].function = value
                elif key == "function_ref":
                    sim.x[index].function = public_functions.get(value)
                elif key == "params":
                    if (
                        not hasattr(sim.x[index], "params")
                        or sim.x[index].params is None
                    ):
                        sim.x[index].params = {}
                    for param_key, param_value in value.items():
                        sim.x[index].params[
                            param_key
                        ] = param_value  # Only updates provided keys
                else:
                    setattr(sim.x[index], key, value)
        return sim

    def set_y_variables(_y, sim):
        for y in _y:
            name = y.get("name")
            index = next(
                (i for i, variable in enumerate(sim.y) if name == variable.name),
                None,
            )
            if index is None:
                sim.y.append(Variable())
                index = len(sim.y) - 1
            for key, value in y.items():
                if key == "function":
                    sim.y[index].function = value
                elif key == "function_ref":
                    sim.y[index].function = public_functions.get(value)
                elif key == "params":
                    if (
                        not hasattr(sim.y[index], "params")
                        or sim.y[index].params is None
                    ):
                        sim.y[index].params = {}
                    for param_key, param_value in value.items():
                        sim.y[index].params[
                            param_key
                        ] = param_value  # Only updates provided keys
                else:
                    setattr(sim.y[index], key, value)
        return sim

    def set_regions(regions, sim):
        for region in regions:
            name = region.get("name")
            index = next(
                (i for i, r in enumerate(sim.regions) if name == r.name),
                None,
            )
            if index is None:
                sim.regions.append(Region())
                index = len(sim.regions) - 1  # safer than -1

            for key, value in region.items():
                if key == "seasons":
                    if (
                        not hasattr(sim.regions[index], "seasons")
                        or sim.regions[index].seasons is None
                    ):
                        sim.regions[index].seasons = {}
                    for season_entry in value:
                        season_name = season_entry.get("name")
                        season_data = season_entry.get("season")
                        if season_data is None:
                            season_ref = season_entry.get("season_ref")
                            season_data = public_lists.get(season_ref)
                        if season_name and season_data:
                            sim.regions[index].seasons[
                                season_name
                            ] = season_data  # update, not overwrite
                else:
                    setattr(sim.regions[index], key, value)

        return sim

    dsl = load_yaml(yaml_path)

    public = dsl.get("public")
    public_functions = public.get("functions")
    public_lists = public.get("lists")

    _simulators = dsl.get("simulators")
    simulators = []
    _evaluators = dsl.get("evaluators")
    evaluators = []

    for i, simulator in enumerate(_simulators):

        inherits = simulator.get("inherit")
        if inherits:
            sim_to_inherit = next(sim for sim in simulators if sim.id == inherits)
            sim = copy.deepcopy(sim_to_inherit)
        else:
            sim = Simulation()
            sim.public_lists = public_lists
        for key, value in simulator.items():
            if key == "x":
                sim = set_x_variables(value, sim)
            elif key == "y":
                sim = set_y_variables(value, sim)
            elif key == "regions":
                sim = set_regions(value, sim)
            elif key != "inherits":
                sim.__setattr__(key, value)
        simulators.append(sim)

    for evaluator in _evaluators:
        eval = Evaluator(evaluator)
        evaluators.append(eval)

    return simulators, evaluators


def load_yaml(yaml_path):
    parameters = None
    with open(yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    if parameters is None:
        raise ValueError

    return parameters
