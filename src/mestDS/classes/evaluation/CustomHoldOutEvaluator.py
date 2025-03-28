import copy
import os
import subprocess
from fpdf import FPDF
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from mestDS.classes.Simulation import Simulation


class CustomHoldOutEvaluator:
    model: str

    def __init__(self, model):
        self.model = model

    def evaluate(self, simulation):
        # create train set
        train_sim = copy.deepcopy(simulation)
        train_sim.simulate()
        self.train_file = f"mestds/{train_sim.simulation_name}_train.csv"
        train_sim.convert_to_csv(self.train_file)

        # create test set
        test_sim = copy.deepcopy(simulation)
        test_sim.simulate()
        self.test_file = f"mestds/{train_sim.simulation_name}_test.csv"
        test_sim.convert_to_csv(self.test_file)

        train_command = [
            "python",
            f"{self.model}/train.py",
            self.train_file,
            "mestds/model.bin",
        ]
        subprocess.run(train_command, check=True)

        self.predictions_file = f"mestds/predictions.csv"

        test_command = [
            "python",
            f"{self.model}/predict.py",
            "mestds/model.bin",
            self.train_file,
            self.test_file,
            self.predictions_file,
        ]

        subprocess.run(test_command, check=True)

        self.generate_report(simulation)

    def generate_report(
        self,
        simulation,
    ):

        features_to_plot = ["mean_temperature", "rainfall"]
        train_df = pd.read_csv(self.train_file)
        pred_df = pd.read_csv(self.predictions_file)

        # Prepare to store evaluation metrics for all locations
        all_metrics = []

        # Plot the data and evaluate the model for each location
        locations = pred_df["location"].unique()

        for location in locations:
            location_data = pred_df[pred_df["location"] == location]
            location_data_train = train_df[train_df["location"] == location]

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
                location_data_train["time_period"],
                location_data_train["disease_cases"],
                label=f"{location} - Disease cases (train)",
                color="green",
            )
            plt.plot(
                location_data["time_period"],
                location_data["disease_cases"],
                label=f"{location} - Disease Cases",
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
                    location_data["time_period"],
                    location_data[feature],
                    label=f"{location} - {feature.replace('_', ' ').title()}",
                    linestyle="--",
                    linewidth=1.5,
                )
            plt.plot()
            plt.title(f"{location} - Disease Cases vs Predicted Cases")
            plt.xlabel("Time Period")
            plt.ylabel("Cases")
            plt.legend()
            plt.xticks(location_data["time_period"][::5], rotation=45)
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
        pdf.output(
            f"mestds/{simulation.simulation_name}_{model_name}_custom_holdout.pdf"
        )
