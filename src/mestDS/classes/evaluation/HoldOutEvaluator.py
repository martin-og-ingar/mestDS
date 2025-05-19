from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path

from fpdf import FPDF
from matplotlib import dates, pyplot as plt
import numpy as np
import pandas as pd

from mestDS.classes.Simulation import Simulation
from mestDS.utils import train_test_split_csv, convert_time_period
from mestDS.classes.Result import Result
from mestDS.classes.LossMetrics import LossMetrics

from chap_core.external.external_model import (
    get_model_from_directory_or_github_url,
)
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor
from chap_core.assessment.dataset_splitting import (
    train_test_generator,
)

import pandas as pd


class HoldOutEvaluator:

    def __init__(self, config):
        self.config = config

    def evaluate(self, simulation, exclude_feature=[]):
        self.initialize_model()
        self.initialize_data(simulation, exclude_feature)
        self.run_external_chap_model()
        # self.generate_report(simulation)

    def initialize_model(self):
        self.model_path = self.config.get("model")
        model = get_model_from_directory_or_github_url(
            self.model_path, base_working_dir=Path("runs")
        )
        self.model = model
        self.working_dir = f"{self.model._working_dir}"
        self.is_external_chap_model = True

    def initialize_data(self, simulation, exclude_feature):
        time_granularity = self.config.get("time_granularity")
        if time_granularity is not None:
            simulation.time_granularity = time_granularity
        simulation.simulate()
        self.dataset_file = (
            f"{self.working_dir}/{simulation.simulation_name}_dataset.csv"
        )
        simulation.convert_to_csv(self.dataset_file)

        self.full_ds = DataSet.from_csv(self.dataset_file, FullData)

        (
            self.train_file,
            self.test_file,
        ) = train_test_split_csv(
            self.dataset_file,
            self.working_dir,
            exclude_feature,
            self.config.get("prediction_length") or 12,
            simulation.time_granularity,
        )

        self.predictions_file = f"{simulation.simulation_name}_predictions.csv"

    def run_external_chap_model(self):
        self.train_external_model()
        self.test_external_model()
        self.get_forecast_dict()
        self.plot_forecast()

    def train_external_model(self):
        self.train_data = DataSet.from_csv(self.train_file, FullData)
        self.model.train(self.train_data)

    def test_external_model(self):
        self.test_dataset = DataSet.from_csv(self.test_file, FullData)
        self.forecasts = self.model.predict(self.train_data, self.test_dataset)
        self.forecasts.to_csv(self.predictions_file)

    def get_forecast_dict(self):
        forecast_dict = defaultdict(list)

        for location, samples in self.forecasts.items():
            forecast_dict[location].append(ForecastAdaptor.from_samples(samples))

        self.forecast_dict = forecast_dict

    def plot_forecast(self):

        for location in self.full_ds.keys():
            location_data = self.full_ds[location]

            try:
                time_periods = pd.to_datetime(location_data.time_period.tolist())
            except Exception:
                time_periods = pd.Series(
                    [convert_time_period(p) for p in location_data.time_period]
                )

            fig, ax = plt.subplots(figsize=(12, 8))

            ax.plot(
                time_periods,
                location_data.disease_cases,
                label="Actual Disease Cases",
                color="black",
                linestyle="-",
            )

            ax.xaxis.set_major_locator(dates.MonthLocator(interval=5))
            plt.xticks(rotation=30, ha="right")

            ax.set_title(f"Disease Cases Forecast for {location}", fontsize=14)
            ax.set_xlabel("Time Period", fontsize=12)
            ax.set_ylabel("Number of Cases", fontsize=12)

            self.forecast_dict[location][0].plot()

            plt.savefig(f"{self.working_dir}/{location}_plot.png")
            plt.close()

    def calculate_metrics(self, simulation):
        loss_per_location = []

        for location in simulation.regions:
            loc_name = location.name
            mean_predicted_values = self.forecast_dict[loc_name][0].mean
            actual_values = self.test_dataset[loc_name].disease_cases

            predicted = np.array(mean_predicted_values)
            actual = np.array(actual_values)

            if predicted.shape != actual.shape:
                raise ValueError(
                    f"Shape mismatch for location {loc_name}: predicted {predicted.shape}, actual {actual.shape}"
                )

            mse = np.mean((predicted - actual) ** 2)
            tu = theils_u(actual, predicted)
            pcd = pocid(actual, predicted)

            loss_metrics = LossMetrics(loc_name, mse, pcd, tu)
            loss_per_location.append(loss_metrics)

        return loss_per_location

    def generate_result(self):
        loss_per_location = self.calculate_metrics()
        result = Result()

    def generate_report(
        self,
        simulation,
    ):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()
        pdf.set_font("Times", "B", 20)
        pdf.cell(
            200,
            10,
            txt="Time Series Forecasting: Model Evaluation Report",
            ln=True,
            align="C",
        )
        pdf.ln(10)

        pdf.set_font("Times", "I", 14)
        pdf.cell(200, 10, txt=f"Model: {self.model.name}", ln=True, align="C")
        pdf.cell(
            200, 10, txt=f"Simulation: {simulation.simulation_name}", ln=True, align="C"
        )
        pdf.ln(10)
        # Table of Contents
        pdf.set_font("Times", "B", 12)
        pdf.cell(200, 10, txt="Table of Contents", ln=True, align="L")
        pdf.set_font("Times", "", 12)
        pdf.cell(200, 10, txt="1. Features Used in Simulation", ln=True, align="L")
        pdf.cell(200, 10, txt="2. Regions in Simulation", ln=True, align="L")
        pdf.cell(200, 10, txt="3. Model Evaluation Metrics", ln=True, align="L")
        pdf.cell(200, 10, txt="4. Plots and Visualizations", ln=True, align="L")
        pdf.ln(15)

        pdf.set_font("Times", "", 12)
        pdf.multi_cell(
            0,
            10,
            txt="This report provides a comprehensive evaluation of a time series forecasting model. The analysis covers various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Residual plots for different locations across the simulation. The results aim to help understand the models performance and guide further improvements.",
        )
        pdf.ln(10)

        pdf.add_page()

        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "1. Features Used in Simulation", ln=True)
        pdf.ln(5)
        pdf.set_font("Times", "", 12)
        for feature in simulation.features:
            pdf.set_font("Times", "B", 12)
            pdf.cell(0, 8, f"Feature: {feature.name}", ln=True)
            pdf.set_font("Times", "", 12)
            pdf.multi_cell(0, 6, feature.function)
            pdf.ln(5)

        pdf.add_page()

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Regions in Simulation:", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", "", 10)
        for region in simulation.regions:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 8, f"Region: {region.name} (ID: {region.region_id})", ln=True)

            pdf.set_font("Arial", "", 10)
            pdf.cell(
                0, 8, f"Neighbours: {', '.join(map(str, region.neighbour))}", ln=True
            )

            if region.rain_season:
                pdf.cell(0, 8, "Rain Seasons:", ln=True)
                for season in region.rain_season:
                    pdf.multi_cell(
                        0, 6, f"  - start:{season.start}, end: {season.end}", ln=True
                    )
            pdf.ln(5)

        pdf.add_page()

        loss_per_location = self.calculate_metrics(simulation)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Score metrics per region:", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", "", 10)
        col_width = pdf.epw / 5
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(col_width, 8, "Location", border=1, fill=True)
        pdf.cell(col_width, 8, "MSE", border=1, fill=True)
        pdf.cell(col_width, 8, "TU", border=1, fill=True)
        pdf.cell(col_width, 8, "POCID", border=1, fill=True)
        pdf.ln(8)

        pdf.set_fill_color(255, 255, 255)

        for location, metrics in loss_per_location.items():
            pdf.cell(col_width, 8, location, border=1)
            pdf.cell(col_width, 8, f"{metrics['MSE']:.3f}", border=1)
            pdf.cell(col_width, 8, f"{metrics['TU']:.3f}", border=1)
            pdf.cell(col_width, 8, f"{metrics['POCID']:.3f}", border=1)
            pdf.ln(8)

        pdf.add_page()
        y_position = 10
        for region in simulation.regions:
            plot_file = f"{self.working_dir}/{region.name}_plot.png"
            pdf.image(plot_file, x=10, y=y_position, w=190)  # Adjust y dynamically
            y_position += 130

            if y_position > 250:
                pdf.add_page()
                y_position = 10

        now = datetime.now()
        time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        model_name = self.model.name.replace(" ", "_")

        if not os.path.exists("reports"):
            os.makedirs("reports")
        print(
            f"reports/{time_stamp}_{model_name}_holdout_{simulation.simulation_name}.pdf"
        )
        pdf.output(
            f"reports/{time_stamp}_{model_name}_holdout_{simulation.simulation_name}.pdf"
        )


def theils_u(y_true, y_pred):
    num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = np.sqrt(np.mean(y_true[1:] ** 2)) + np.sqrt(np.mean(y_pred[1:] ** 2))
    return num / denom


def pocid(y_true, y_pred):
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    correct = np.sum(direction_true == direction_pred)
    return (correct / len(direction_true)) * 100
