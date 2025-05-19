import copy
from datetime import datetime
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mestDS.classes.ModelRunner import ModelRunner
from mestDS.classes.Result import Result
from mestDS.classes.Simulation import Simulation
from mestDS.classes.PDF import PDF
from mestDS.utils import set_runner


class Evaluator:
    results: list[Result]
    runner: ModelRunner

    def __init__(self, config):
        self.runner = set_runner(config)
        self.time_granularity = config.get("time_granularity")
        self.results = []

    def evaluate(self, simulations: list[Simulation]):
        print(f"mestDS - Evaluator running on model {self.runner.model_path}")
        for sim in simulations:
            _sim = copy.deepcopy(sim)
            if self.time_granularity:
                _sim.time_granularity = self.time_granularity
            _sim.simulate()

            self.results.append(self.runner.run(_sim))

        self.generate_report()

    def generate_report(self):
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"reports/{timestamp}_{self.runner.model.name}.pdf"
        pdf = PDF(orientation="L")
        pdf.add_page()
        pdf.add_header(f"Model Evaluation: {self.runner.model.name}")
        # for result in self.results:
        #     pdf.add_subheader(result.simulation_name)
        #     pdf.add_table(result.metrics)
        #     for plot in result.plots:
        #         pdf.add_plot(plot)
        for result in self.results:
            for plot in result.plots:
                pdf.add_subheader_table_and_plot(
                    result.simulation_name, result.metrics, plot
                )
        pdf.output(filename)
        # with PdfPages(filename) as pdf:
        #     for result in self.results:
        #         fig = plt.figure(figsize=(8.5, 11))
        #         ax = fig.add_subplot(111)
        #         ax.axis("off")

        #         fig.text(
        #             0.5,
        #             0.95,
        #             result.simulation_name,
        #             ha="center",
        #             va="top",
        #             fontsize=14,
        #         )

        #         from io import BytesIO

        #         buf = BytesIO()
        #         result.plot.savefig(buf, format="png", bbox_inches="tight")
        #         buf.seek(0)
        #         img = plt.imread(buf)
        #         fig.figimage(img, xo=50, yo=100)

        #         pdf.savefig(fig)
        #         plt.close(fig)
