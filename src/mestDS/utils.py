from collections import defaultdict
from datetime import datetime
import logging
import os
from pathlib import Path
import re
import shutil
import uuid
from venv import logger
from matplotlib import dates
from matplotlib.lines import Line2D
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# from mestDS.classes.ModelRunner import ModelRunner
from mestDS.classes.LossMetrics import LossMetrics
import git
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor

# from chap_core.external.external_model import (
#     get_model_template_from_mlproject_file,
#     get_model_from_mlproject_file,
# )
# from chap_core.external.mlflow_wrappers import ExternalModel
from chap_core.exceptions import InvalidModelException


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
    metrics = config.get("metrics") or ["mse", "theils_u", "pocid"]
    plot_length = config.get("plot_length") or (pred_len * n_test_sets) * 4
    user_options = config.get("user_options") or None
    return ModelRunner(
        model_path, pred_len, n_test_sets, stride, metrics, plot_length, user_options
    )


def get_forecast_dicts(forecasts):
    forecast_dicts = []
    for forecast in forecasts:
        forecast_dict = defaultdict(list)
        for location, samples in forecast.items():
            forecast_dict[location].append(ForecastAdaptor.from_samples(samples))
        forecast_dicts.append(forecast_dict)
    return forecast_dicts


def get_plots(full_ds, forecast_dicts, plot_length):
    for location in full_ds.keys():
        print(full_ds)
        location_data = full_ds[location][-plot_length:]

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
            color="royalblue",
            linestyle="-",
            alpha=0.2,
        )
        ax.plot(
            time_periods,
            location_data.mean_temperature,
            label="Mean temperature",
            color="crimson",
            linestyle="-",
            alpha=0.2,
        )

        ax.xaxis.set_major_locator(dates.MonthLocator(interval=5))
        plt.xticks(rotation=30, ha="right")

        ax.set_title(f"Disease Cases Forecast for {location}", fontsize=14)
        ax.set_xlabel("Time Period", fontsize=12)
        ax.set_ylabel("Number of Cases", fontsize=12)

        for fore_dict in forecast_dicts:
            fore_dict[location][0].plot(color="purple", ax=ax)

        pred_legend = Line2D([0], [0], color="purple", label="Predicted Disease Cases")
        ax.legend(handles=[pred_legend] + ax.get_legend_handles_labels()[0])

        yield plt


def slugify(name: str) -> str:
    name = name.lower()
    name = name.replace("=", "equals")
    name = re.sub(r"[^\w\s\-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def ensure_trailing_slash(path: str) -> str:
    if not path.endswith("/"):
        path += "/"
    return path


def get_metrics(full_ds, forecast_dicts, metrics):
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

        metrics_dict = {
            metric: loss_metric(metric, actual, predicted) for metric in metrics
        }

        yield LossMetrics(location, metrics_dict)


def loss_metric(metric: str, y_true, y_pred, quantile=0.5, seasonality=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metric = metric.lower()

    # Helper functions
    def mae():
        return np.mean(np.abs(y_true - y_pred))

    def mse():
        return np.mean((y_true - y_pred) ** 2)

    def rmse():
        return np.sqrt(mse())

    def mape():
        return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-10, None))) * 100

    def smape():
        return 100 * np.mean(
            2
            * np.abs(y_pred - y_true)
            / np.clip(np.abs(y_true) + np.abs(y_pred), 1e-10, None)
        )

    def wape():
        return np.sum(np.abs(y_true - y_pred)) / np.clip(
            np.sum(np.abs(y_true)), 1e-10, None
        )

    def rmsle():
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

    def mase():
        naive_forecast = y_true[:-seasonality]
        actual = y_true[seasonality:]
        mae_naive = np.mean(np.abs(actual - naive_forecast))
        return mae() / np.clip(mae_naive, 1e-10, None)

    def rmsse():
        naive_forecast = y_true[:-seasonality]
        actual = y_true[seasonality:]
        mse_naive = np.mean((actual - naive_forecast) ** 2)
        return rmse() / np.clip(np.sqrt(mse_naive), 1e-10, None)

    def sql():
        return np.mean(
            2
            * np.maximum(
                quantile * (y_true - y_pred), (quantile - 1) * (y_true - y_pred)
            )
        )

    def wql():
        weights = np.abs(y_true)
        return np.sum(
            weights
            * np.maximum(
                quantile * (y_true - y_pred), (quantile - 1) * (y_true - y_pred)
            )
        ) / np.clip(np.sum(weights), 1e-10, None)

    def theils_u():
        _y_true = np.array(y_true)
        _y_pred = np.array(y_pred)

        rmse_model = np.sqrt(np.mean((_y_true - _y_pred) ** 2))

        naive_forecast = _y_true[:-1]
        actual_values = _y_true[1:]

        rmse_naive = np.sqrt(np.mean((actual_values - naive_forecast) ** 2))

        if rmse_naive == 0:
            return np.nan

        return round(rmse_model / rmse_naive, 4)

    def pocid():
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        correct = np.sum(direction_true == direction_pred)
        return round((correct / len(direction_true)) * 100, 2)

    metrics = {
        "mae": mae,
        "mape": mape,
        "mase": mase,
        "mse": mse,
        "rmse": rmse,
        "rmsle": rmsle,
        "rmsse": rmsse,
        "smape": smape,
        "wape": wape,
        "sql": sql,
        "wql": wql,
        "pocid": pocid,
        "theils_u": theils_u,
    }

    if metric not in metrics:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported metrics: {list(metrics.keys())}"
        )

    value = metrics[metric]()

    return round(value, 2)


from chap_core.models.utils import (
    _get_model_code_base,
    get_model_template_from_mlproject_file,
)
from chap_core.models.model_template import ModelTemplate


def get_model_template_from_directory_or_github_url(
    model_template_path,
    base_working_dir=Path("runs/"),
    ignore_env=False,
    run_dir_type="timestamp",
    user_options=None,
) -> ModelTemplate:
    """
    Note: Preferably use ModelTemplate.from_directory_or_github_url instead of
    using this function directly. This function may be depcrecated in the future.

    Gets the model template and initializes a working directory with the code for the model.
    model_path can be a local directory or github url

    Parameters
    ----------
    model_template_path : str
        Path to the model. Can be a local directory or a github url
    base_working_dir : Path, optional
        Base directory to store the working directory, by default Path("runs/")
    ignore_env : bool, optional
        If True, will ignore the environment specified in the MLproject file, by default False
    run_dir_type : Literal["timestamp", "latest", "use_existing"], optional
        Type of run directory to create, by default "timestamp", which creates a new directory based on current timestamp for the run.
        "latest" will create a new directory based on the model name, but will remove any existing directory with the same name.
        "use_existing" will use the existing directory specified by the model path if that exists. If that does not exist, "latest" will be used.
    """

    logger.info(
        f"Getting model template from {model_template_path}. Ignore env: {ignore_env}. Base working dir: {base_working_dir}. Run dir type: {run_dir_type}"
    )
    working_dir = _get_model_code_base(
        model_template_path, base_working_dir, run_dir_type
    )

    logger.info(
        f"Current directory is {os.getcwd()}, working dir is {working_dir.absolute()}"
    )
    assert os.path.isdir(working_dir), working_dir
    assert os.path.isdir(os.path.abspath(working_dir)), working_dir

    # assert that a config file exists
    ml_project_path = working_dir / "MLproject"
    if not (ml_project_path).exists():
        raise InvalidModelException("No MLproject file found in model directory")
    elif user_options is not None:
        with open(working_dir / "MLproject", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)
        for key, value in user_options.items():
            config["user_options"][key] = value
        print("After changing user_options")
        print(config)
        with open(working_dir / "MLproject", "w") as file:
            yaml.dump(config, file, sort_keys=False)

    template = get_model_template_from_mlproject_file(
        working_dir / "MLproject", ignore_env=ignore_env
    )
    return template


# def get_model_from_directory_or_github_url(
#     model_path,
#     base_working_dir=Path("runs/"),
#     ignore_env=False,
#     run_dir_type: Literal["timestamp", "latest", "use_existing"] = "timestamp",
# ):
#     """
#     Gets the model and initializes a working directory with the code for the model.
#     model_path can be a local directory or github url

#     Parameters
#     ----------
#     model_path : str
#         Path to the model. Can be a local directory or a github url
#     base_working_dir : Path, optional
#         Base directory to store the working directory, by default Path("runs/")
#     ignore_env : bool, optional
#         If True, will ignore the environment specified in the MLproject file, by default False
#     run_dir_type : Literal["timestamp", "latest", "use_existing"], optional
#         Type of run directory to create, by default "timestamp", which creates a new directory based on current timestamp for the run.
#         "latest" will create a new directory based on the model name, but will remove any existing directory with the same name.
#         "use_existing" will use the existing directory specified by the model path if that exists. If that does not exist, "latest" will be used.
#     """
#     is_github = False
#     commit = None
#     if isinstance(model_path, str) and model_path.startswith("https://github.com"):
#         dir_name = model_path.split("/")[-1].replace(".git", "")
#         model_name = dir_name
#         if "@" in model_path:
#             model_path, commit = model_path.split("@")
#         is_github = True
#     else:
#         model_name = Path(model_path).name

#     if run_dir_type == "use_existing" and not Path(model_path).exists():
#         run_dir_type = "latest"

#     if run_dir_type == "latest":
#         working_dir = base_working_dir / model_name / "latest"
#         # clear working dir
#         if working_dir.exists():
#             logger.info(f"Removing previous working dir {working_dir}")
#             shutil.rmtree(working_dir)
#     elif run_dir_type == "timestamp":
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         unique_identifier = timestamp + "_" + str(uuid.uuid4())[:8]
#         # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S%f")
#         working_dir = base_working_dir / model_name / unique_identifier
#         # check that working dir does not exist
#         assert (
#             not working_dir.exists()
#         ), f"Working dir {working_dir} already exists. This should not happen if make_run_dir is True"
#     elif run_dir_type == "use_existing":
#         working_dir = Path(model_path)
#     else:
#         raise ValueError(f"Invalid run_dir_type: {run_dir_type}")

#     logger.info(f"Writing results to {working_dir}")

#     if is_github:
#         working_dir.mkdir(parents=True)
#         repo = git.Repo.clone_from(model_path, working_dir)
#         if commit:
#             logger.info(f"Checking out commit {commit}")
#             repo.git.checkout(commit)

#     elif run_dir_type == "use_existing":
#         logging.info("Not copying any model files, using existing directory")
#     else:
#         # copy contents of model_path to working_dir
#         logger.info(f"Copying files from {model_path} to {working_dir}")
#         shutil.copytree(model_path, working_dir)

#     logging.error(f"Current directory is {os.getcwd()}")
#     logging.error(f"Working dir is {working_dir}")
#     assert os.path.isdir(working_dir), working_dir
#     assert os.path.isdir(os.path.abspath(working_dir)), working_dir
#     # assert that a config file exists
#     if (working_dir / "MLproject").exists():
#         # return get_model_template_from_mlproject_file(
#         #     working_dir / "MLproject", ignore_env=ignore_env
#         # ).get_model(model_configuration=None)
#         return get_model_from_mlproject_file(
#             working_dir / "MLproject", ignore_env=ignore_env
#         )
#     else:
#         raise InvalidModelException("No MLproject file found in model directory")


# def get_model_template_from_mlproject_file(
#     mlproject_file, ignore_env=False
# ) -> ExternalModel:
#     working_dir = Path(mlproject_file).parent

#     with open(mlproject_file, "r") as file:
#         config = yaml.load(file, Loader=yaml.FullLoader)
#     config = ModelTemplateConfig.model_validate(config)

#     model_template = ModelTemplate(config, working_dir, ignore_env)
#     return model_template
