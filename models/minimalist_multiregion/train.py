import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import get_df_per_location


def read_pandas_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df


# print(read_pandas_from_csv("trainData.csv"))


# SHOWN TO USER
def train(csv_fn, model_fn):
    features = ["rainfall", "mean_temperature"]
    models = {}
    locations = get_df_per_location(csv_fn)
    for location, df in locations.items():

        X = df[features]
        Y = df["disease_cases"]
        Y = Y.fillna(
            0
        )  # set NaNs to zero (not a good solution, just for the example to work)
        model = LinearRegression()
        model.fit(X, Y)
        models[location] = model
        print(
            f"Train - model coefficients for location {location}: ",
            list(zip(features, model.coef_)),
        )

    joblib.dump(models, model_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a minimalist forecasting model."
    )

    parser.add_argument(
        "csv_fn", type=str, help="Path to the CSV file containing input data."
    )
    parser.add_argument("model_fn", type=str, help="Path to save the trained model.")
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)
