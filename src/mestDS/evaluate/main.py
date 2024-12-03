from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.predictor.model_registry import registry
from mestDS.classes import RainSeason
from ..classes.ClimateHealthData import to_gluonTS_format
from mestDS.utils.main import train_test_split_csv
from models.minimalist_multiregion.train import train
from models.minimalist_multiregion.predict import predict


def evaluate_chap_model(datasets):
    model = registry.get_model("chap_ewars_monthly")
    for dataset in datasets:
        data = DataSet.from_period_observations(dataset)
        evaluate_model(model, data, report_filename="test.pdf")


def test_model_on_simulated_data(
    filename,
    model,
    beta_rain,
    beta_temp,
    beta_lag_sickness,
    sim_length,
    time_granularity,
):
    from mestDS import Simulation, plot_data_with_sample_0

    filename = filename
    model = f"model_{filename}"
    sim = Simulation(
        simulation_length=sim_length,
        time_granularity=time_granularity,
        rain_season=[RainSeason(start=30, end=90), RainSeason(start=180, end=250)],
        beta_rainfall=beta_rain,
        beta_temp=beta_temp,
        beta_lag_sickness=beta_lag_sickness,
    )

    sim.simulate()
    sim.simulated_data_to_csv(filepath=f"datasets/{filename}.csv")
    train_test_split_csv(f"datasets/{filename}.csv", "datasets/splitted_datasets/")

    train(f"datasets/splitted_datasets/{filename}_train.csv", model)
    predict(
        model,
        "",
        f"datasets/splitted_datasets/{filename}_x_test.csv",
        f"datasets/predictions/{filename}_predictions.csv",
    )
    plot_data_with_sample_0(
        f"datasets/splitted_datasets/{filename}_y_test.csv",
        f"datasets/predictions/{filename}_predictions.csv",
        f"figures/{filename}.png",
    )
