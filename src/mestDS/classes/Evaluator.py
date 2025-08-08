import copy
from datetime import datetime
import os
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mestDS.classes.ModelRunner import ModelRunner
from mestDS.classes.Result import Result
from mestDS.classes.Simulation import Simulation
from mestDS.classes.PDF import PDF
from mestDS.utils import ensure_trailing_slash, set_runner


class Evaluator:
    results: list[Result]
    runner: ModelRunner

    def __init__(self, config):
        self.runner = set_runner(config)
        self.time_delta = config.get("time_delta")
        self.sim_length = config.get("simulation_length")
        self.results = []

    def evaluate(
        self,
        report_path,
        simulations: list[Simulation] = None,
        path: str = None,
    ):

        print(f"mestDS - Evaluator running on model {self.runner.model_path}")
        if simulations:
            for sim in simulations:
                _sim = copy.deepcopy(sim)
                if self.time_delta:
                    _sim.time_delta = self.time_delta
                if self.sim_length:
                    _sim.length = self.sim_length
                if self.sim_length or self.time_delta:
                    _sim.simulate()
                self.results.append(self.runner.run(simulation=_sim))
                if _sim.description is not None:
                    self.results[-1].description = _sim.description
        elif path:
            path_obj = Path(path)
            if path_obj.is_file() and path_obj.suffix == ".csv":
                self.results.append(self.runner.run(filename=path_obj))
            elif path_obj.is_dir():
                for filename in path_obj.glob("*.csv"):
                    self.results.append(self.runner.run(filename=filename))
            else:
                print(f"Path '{path}' is not a valid CSV file or directory.")
        self.generate_report(report_path)

    def generate_report(self, path):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = (
            f"{ensure_trailing_slash(path)}{timestamp}_{self.runner.model.name}.pdf"
        )
        pdf = PDF()
        pdf.add_page()
        pdf.add_header(f"Model Evaluation: {self.runner.model.name}")
        for result in self.results:
            pdf.add_subheader_table_and_plot(result)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pdf.output(filename)
