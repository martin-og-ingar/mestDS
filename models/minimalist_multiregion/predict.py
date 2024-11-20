import argparse

import joblib
import pandas as pd

from .utils import get_df_per_location


def predict(model_fn, historic, future_climatedata_fn, predictions_fn):
    models = joblib.load(model_fn)
    locations = get_df_per_location(future_climatedata_fn)
    first_location = True
    for location, df in locations.items():
        X = df[["rainfall", "mean_temperature"]]
        model = models[location]
        y_pred = model.predict(X)
        df["sample_0"] = y_pred
        if first_location:
            df.to_csv(predictions_fn, index=False, mode="w", header=True)
            first_location = False
        else:
            df.to_csv(predictions_fn, index=False, mode="a", header=False)

        print("predict - forecast values: ", y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using the trained model.")

    parser.add_argument("model_fn", type=str, help="Path to the trained model file.")
    parser.add_argument(
        "historic_data_fn",
        type=str,
        help="Path to the CSV file historic data (here ignored).",
    )
    parser.add_argument(
        "future_climatedata_fn",
        type=str,
        help="Path to the CSV file containing future climate data.",
    )
    parser.add_argument(
        "predictions_fn", type=str, help="Path to save the predictions CSV file."
    )

    args = parser.parse_args()
    predict(
        args.model_fn,
        args.historic_data_fn,
        args.future_climatedata_fn,
        args.predictions_fn,
    )
