from collections import defaultdict
import os
from pathlib import Path
import subprocess

from fpdf import FPDF
from matplotlib import dates, pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from mestDS.utils import train_test_split_csv

from chap_core.external.external_model import (
    get_model_from_directory_or_github_url,
    ExternalModel,
)
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor


import pandas as pd


class HoldOutEvaluator:
    is_external_chap_model = False

    def __init__(self, config):

        model_path = config.get("model")

        if model_path.startswith("https://github.com"):
            model = get_model_from_directory_or_github_url(
                model_path, base_working_dir=Path("runs"), run_dir_type="use_existing"
            )
            if isinstance(model, ExternalModel):
                self.model = model
                self.working_dir = f"{self.model._working_dir}"
                self.is_external_chap_model = True
            else:
                self.working_dir = model  # working dir
        else:
            self.working_dir = model_path

        if self.is_external_chap_model is False:
            self.train_command = config.get("train_command")
            self.test_command = config.get("test_command")

        self.test_size_percentage = config.get("test_size_percentage")
        self.test_size_time_periods = config.get("test_size_time_periods")

    def evaluate(self, simulation, exclude_feature=[]):

        self.initialize_data(simulation, exclude_feature)

        if self.is_external_chap_model:
            self.run_external_chap_model()
        else:
            self.run_model()

    def initialize_data(self, simulation, exclude_feature):
        simulation.simulate()
        self.dataset_file = (
            f"{self.working_dir}/{simulation.simulation_name}_dataset.csv"
        )
        simulation.convert_to_csv(self.dataset_file)
        (
            self.train_file,
            self.test_file,
        ) = train_test_split_csv(
            self.dataset_file,
            self.working_dir,
            exclude_feature,
            self.test_size_percentage,
            simulation.time_granularity,
        )

        self.predictions_file = f"{simulation.simulation_name}_predictions.csv"

    # def run_internal_chap_model(self, file_extension):

    #     language = "python" if file_extension == "py" else "Rscript"

    #     if file_extension == "R":
    #         # Step 1: Install missing dependencies
    #         install_dependencies = [
    #             "Rscript",
    #             "-e",
    #             'dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE);'
    #             'packages <- c("fmesher", "lifecycle", "rlang", "withr", "MatrixModels", "tsModel", "dlnm");'
    #             'new_packages <- packages[!(packages %in% installed.packages()[,"Package"])];'
    #             'if(length(new_packages)) install.packages(new_packages, repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))',
    #         ]
    #         subprocess.run(install_dependencies, check=True)

    #         install_inla = [
    #             "Rscript",
    #             "-e",
    #             'if (!"INLA" %in% installed.packages()[, "Package"]) { install.packages("INLA", repos="https://inla.r-inla-download.org/R/stable", lib=Sys.getenv("R_LIBS_USER")) }',
    #         ]
    #         subprocess.run(install_inla, check=True)

    #     train_command = [
    #         language,
    #         f"{self.model}/train.{file_extension}",
    #         self.train_file,
    #         "runs/model.bin",
    #     ]
    #     subprocess.run(train_command, check=True)

    #     test_command = [
    #         language,
    #         f"{self.model}/predict.{file_extension}",
    #         "runs/model.bin",
    #         self.train_file,
    #         self.test_file,
    #         self.predictions_file,
    #         "None",
    #         "samples",
    #     ]

    #     subprocess.run(test_command, check=True)

    def run_model():
        return

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
        # Load the dataset
        full_ds = DataSet.from_csv(self.dataset_file, FullData)

        for location in full_ds.keys():
            location_data = full_ds[location]
            time_periods = pd.to_datetime(location_data.time_period.tolist())

            fig, ax = plt.subplots(figsize=(12, 6))  # Wider figure

            # Plot actual disease cases
            ax.plot(
                time_periods,
                location_data.disease_cases,
                label="Actual Disease Cases",
                color="black",
                linestyle="-",
            )

            # Reduce number of x-axis labels (every 5th month)
            ax.xaxis.set_major_locator(dates.MonthLocator(interval=5))
            plt.xticks(rotation=30, ha="right")  # Rotate labels

            # Add title and labels
            ax.set_title(f"Disease Cases Forecast for {location}", fontsize=14)
            ax.set_xlabel("Time Period", fontsize=12)
            ax.set_ylabel("Number of Cases", fontsize=12)

            # Plot forecast
            self.forecast_dict[location][0].plot()

            # Show the plot
            plt.show()

    def generate_report(
        self,
        simulation,
    ):

        features_to_plot = ["mean_temperature", "rainfall"]
        dataset_df = pd.read_csv(self.dataset_file)
        test_y_df = pd.read_csv(self.test_y_file)
        pred_df = pd.read_csv(self.predictions_file)

        merged_df = pd.merge(
            test_y_df, pred_df, on=["time_period", "location"], how="left"
        )

        # Prepare to store evaluation metrics for all locations
        all_metrics = []

        # Plot the data and evaluate the model for each location
        locations = merged_df["location"].unique()

        for location in locations:
            location_data = merged_df[merged_df["location"] == location]
            location_data_dataset = dataset_df[dataset_df["location"] == location]

            # Actual and Predicted Values
            actual = location_data["disease_cases"].values
            predicted = location_data["sample_0"].values

            # Calculate Evaluation Metrics
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(actual, predicted)

            # Store metrics for the report
            all_metrics.append([location, mae, mse, rmse, mape])

            # Plotting
            plt.figure(figsize=(12, 10))

            # Subplot 1: Time Series of Actual vs Predicted, and training data
            plt.subplot(2, 1, 1)
            plt.plot(
                location_data_dataset["time_period"],
                location_data_dataset["disease_cases"],
                label=f"{location} - Disease cases",
                color="blue",
            )
            plt.plot(
                location_data["time_period"],
                location_data["sample_0"],
                label=f"{location} - Predicted Cases",
                linestyle=":",
                color="red",
            )
            for feature in features_to_plot:
                plt.plot(
                    location_data_dataset["time_period"],
                    location_data_dataset[feature],
                    label=f"{location} - {feature.replace('_', ' ').title()}",
                    linestyle="--",
                    linewidth=1.5,
                )
            plt.plot()
            plt.title(f"{location} - Disease Cases vs Predicted Cases")
            plt.xlabel("Time Period")
            plt.ylabel("Cases")
            plt.legend()
            plt.xticks(location_data_dataset["time_period"][::5], rotation=45)
            plt.grid(True)

            # Subplot 2: Residuals (Error over Time)
            residuals = actual - predicted
            plt.subplot(2, 1, 2)
            plt.plot(
                location_data["time_period"],
                residuals,
                label="Residuals",
                color="purple",
            )
            plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
            plt.title(f"{location} - Residuals (Actual - Predicted)")
            plt.xlabel("Time Period")
            plt.ylabel("Residuals")
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)

            plt.tight_layout()
            plot_file = f"mestds/{simulation.simulation_name}_{location}_plot.png"
            plt.savefig(plot_file)
            plt.close()

        # Create a PDF report
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

        # Subtitle and simulation/model details
        pdf.set_font("Times", "I", 14)
        pdf.cell(200, 10, txt=f"Model: {self.model}", ln=True, align="C")
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

        # Add some introductory text
        pdf.set_font("Times", "", 12)
        pdf.multi_cell(
            0,
            10,
            txt="This report provides a comprehensive evaluation of a time series forecasting model. The analysis covers various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Residual plots for different locations across the simulation. The results aim to help understand the models performance and guide further improvements.",
        )
        pdf.ln(10)

        # New page after Table of Contents
        pdf.add_page()

        # Features Section
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

        """
        Print all the regions in the simulation
        """
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

            # Add Rain Season details if available
            if region.rain_season:
                pdf.cell(0, 8, "Rain Seasons:", ln=True)
                for season in region.rain_season:
                    pdf.multi_cell(
                        0, 6, f"  - start:{season.start}, end: {season.end}", ln=True
                    )  # Indent for readability

            pdf.ln(5)  # Space between regions

        pdf.add_page()

        """
        Print all evaluation metrics for each location
        """
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Model Evaluation Metrics:", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", "", 10)
        col_width = pdf.epw / 5  # Distribute evenly over 5 columns
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(col_width, 8, "Location", border=1, fill=True)
        pdf.cell(col_width, 8, "MAE", border=1, fill=True)
        pdf.cell(col_width, 8, "MSE", border=1, fill=True)
        pdf.cell(col_width, 8, "RMSE", border=1, fill=True)
        pdf.cell(col_width, 8, "MAPE", border=1, fill=True)
        pdf.ln(8)

        for metrics in all_metrics:
            pdf.cell(col_width, 8, str(metrics[0]), border=1)
            pdf.cell(col_width, 8, f"{metrics[1]:.3f}", border=1)
            pdf.cell(col_width, 8, f"{metrics[2]:.3f}", border=1)
            pdf.cell(col_width, 8, f"{metrics[3]:.3f}", border=1)
            pdf.cell(col_width, 8, f"{metrics[4]:.3%}", border=1)
            pdf.ln(8)

        """
        Print plots for each simulations.
        """
        for location in locations:
            plot_file = f"mestds/{simulation.simulation_name}_{location}_plot.png"
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(
                0, 10, f"{location} - Time Series and Residuals", ln=True, align="C"
            )
            pdf.ln(10)
            pdf.image(plot_file, x=10, y=30, w=190)  # Adjust positioning as needed

        # Save the PDF

        model_name = os.path.basename(self.model)
        pdf.output(f"mestds/{simulation.simulation_name}_{model_name}_holdout.pdf")
