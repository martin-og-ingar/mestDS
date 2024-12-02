import os
import sys
import numpy as np

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

neighbors = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]])
sim = Simulation(
    simulation_length=365,
    time_granularity="D",
    rain_season=[RainSeason(start=12, end=20), RainSeason(start=36, end=40)],
    regions=["A", "B", "C"],
    beta_rainfall=-1,
    beta_temp=0.9,
    beta_lag_sickness=0.9,
    beta_neighbour_influence=0.9,
    neighbors=neighbors,
)
sim.simulate()
sim.show_graph()

# sim.chap_evaluation_on_model(registry.get_model("chap_ewars_monthly"))
# filename = "with_sickness_lag"
# model = f"model_{filename}"
# sim = Simulation(
#     simulation_length=1000,
#     time_granularity="D",
#     rain_season=[RainSeason(start=30, end=90), RainSeason(start=180, end=250)],
#     beta_rainfall=0.1,
#     beta_temp=0.1,
#     beta_lag_sickness=0.9,
# )
# sim.simulate()
# sim.simulated_data_to_csv(filepath=f"datasets/{filename}.csv")

# train_test_split_csv(f"datasets/{filename}.csv", "datasets/splitted_datasets/")

# train(f"datasets/splitted_datasets/{filename}_train.csv", model)
# predict(
#     model,
#     "",
#     f"datasets/splitted_datasets/{filename}_x_test.csv",
#     f"datasets/predictions/{filename}_predictions.csv",
# )
# plot_data_with_sample_0(
#     f"datasets/splitted_datasets/{filename}_y_test.csv",
#     f"datasets/predictions/{filename}_predictions.csv",
#     f"figures/{filename}.png",
# )

# filename = "without_sickness_lag"
# model = f"model_{filename}"
# sim = Simulation(
#     simulation_length=1000,
#     time_granularity="D",
#     rain_season=[RainSeason(start=30, end=90), RainSeason(start=180, end=250)],
#     beta_rainfall=0.9,
#     beta_temp=0.3,
#     beta_lag_sickness=0.3,
# )
# sim.simulate()
# sim.simulated_data_to_csv(filepath=f"datasets/{filename}.csv")

# train_test_split_csv(f"datasets/{filename}.csv", "datasets/splitted_datasets/")

# train(f"datasets/splitted_datasets/{filename}_train.csv", model)
# predict(
#     model,
#     "",
#     f"datasets/splitted_datasets/{filename}_x_test.csv",
#     f"datasets/predictions/{filename}_predictions.csv",
# )
# plot_data_with_sample_0(
#     f"datasets/splitted_datasets/{filename}_y_test.csv",
#     f"datasets/predictions/{filename}_predictions.csv",
#     f"figures/{filename}.png",
# )
