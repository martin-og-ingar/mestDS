from collections import defaultdict
from datetime import datetime
import re
from matplotlib import dates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from mestDS.classes.ModelRunner import ModelRunner
from mestDS.classes.LossMetrics import LossMetrics
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor


def convert_time_period(period):
    try:
        year, week = int(str(period)[:4]), int(str(period)[5:])
        return datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
    except Exception:
        raise ValueError(f"Unrecognized date format: {period}")


def set_runner(config):
    from mestDS.classes.ModelRunner import ModelRunner

    model_path = config.get("model")
    pred_len = config.get("prediction_length") or 12
    n_test_sets = config.get("n_test_sets") or 1
    stride = config.get("stride") or 1
    return ModelRunner(model_path, pred_len, n_test_sets, stride)


def get_forecast_dicts(forecasts):
    forecast_dicts = []
    for forecast in forecasts:
        forecast_dict = defaultdict(list)
        for location, samples in forecast.items():
            forecast_dict[location].append(ForecastAdaptor.from_samples(samples))
        forecast_dicts.append(forecast_dict)
    return forecast_dicts


def get_plots(full_ds, forecast_dicts):
    for location in full_ds.keys():
        location_data = full_ds[location][-100:]

        try:
            time_periods = pd.to_datetime(location_data.time_period.tolist())
        except Exception:
            time_periods = pd.Series(
                [convert_time_period(p) for p in location_data.time_period]
            )

        fig, ax = plt.subplots(figsize=(11.7, 6.5))

        ax.plot(
            time_periods,
            location_data.disease_cases,
            label="Actual Disease Cases",
            color="green",
            linestyle="-",
        )
        ax.plot(
            time_periods,
            location_data.rainfall,
            label="Rainfall",
            color="blue",
            linestyle="-",
            alpha=0.2,
        )
        ax.plot(
            time_periods,
            location_data.mean_temperature,
            label="Mean temperature",
            color="red",
            linestyle="-",
            alpha=0.2,
        )

        ax.xaxis.set_major_locator(dates.MonthLocator(interval=5))
        plt.xticks(rotation=30, ha="right")

        ax.set_title(f"Disease Cases Forecast for {location}", fontsize=14)
        ax.set_xlabel("Time Period", fontsize=12)
        ax.set_ylabel("Number of Cases", fontsize=12)
        ax.legend()

        for fore_dict in forecast_dicts:
            fore_dict[location][0].plot(
                color="darkorange", name="Predicted Disease Cases", show_label=True
            )

        yield plt


def theils_u(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse_model = np.sqrt(np.mean((y_true - y_pred) ** 2))

    naive_forecast = y_true[:-1]
    actual_values = y_true[1:]

    rmse_naive = np.sqrt(np.mean((actual_values - naive_forecast) ** 2))

    if rmse_naive == 0:
        return np.nan

    return round(rmse_model / rmse_naive, 4)


def pocid(y_true, y_pred):
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    correct = np.sum(direction_true == direction_pred)
    return round((correct / len(direction_true)) * 100, 2)


def get_metrics(full_ds, forecast_dicts):
    for location in full_ds[0].keys():
        location_actual = [
            entry.disease_cases for ds in full_ds for entry in ds[location]
        ]

        location_predicted_mean = []
        for fd in forecast_dicts:
            location_predicted_mean.extend(fd[location][0].mean)

        predicted = np.array(location_predicted_mean)
        actual = np.array(location_actual)

        if predicted.shape != actual.shape:
            raise ValueError(
                f"Shape mismatch for location {location}: predicted {predicted.shape}, actual {actual.shape}"
            )

        mse = round(np.mean((predicted - actual) ** 2), 2)
        tu = theils_u(actual, predicted)
        pcd = pocid(actual, predicted)

        yield LossMetrics(location, mse, pcd, tu)


def slugify(name: str) -> str:
    name = name.lower()
    name = name.replace("=", "equals")
    name = re.sub(r"[^\w\s\-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name
