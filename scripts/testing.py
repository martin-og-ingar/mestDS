import os
import sys

root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Adjust as needed
sys.path.append(root_dir)


from mestDS import Simulation, plot_data_with_sample_0
from chap_core.predictor.model_registry import registry
from mestDS.classes import RainSeason
from mestDS.utils.main import train_test_split_csv
from models.minimalist_multiregion.train import train
from models.minimalist_multiregion.predict import predict
from mestDS.evaluate.main import test_model_on_simulated_data

sim = Simulation(
    simulation_length=365,
    time_granularity="D",
    rain_season=[RainSeason(start=12, end=20), RainSeason(start=36, end=40)],
)
sim.simulate()
sim.show_graph()
# sim.chap_evaluation_on_model(registry.get_model("chap_ewars_monthly"))
