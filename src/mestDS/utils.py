import csv
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
import shutil
from typing import Literal
import uuid
import git
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from fpdf import FPDF

from mestDS.default_variables import DATEFORMAT
from chap_core.external.external_model import get_model_from_mlproject_file


def train_test_split_csv(
    file, directory_to_store, exclude_feature, test_size, time_granularity
):
    """
    Splits a CSV file into train, x_test, and y_test with an 80/20 split for each location,
    and includes location and time_period in the y_test file.
    The data is first sorted by time_period for correct chronological split.
    Args:
    - file (str): The path to the CSV file to be split.
    - directory_to_store (str): The directory where the split files should be saved.
    - test_size (float): The proportion of the data for each location to be used for testing (20% by default).
    """

    # Read the entire file into a list of rows
    with open(file, mode="r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    exclude_indices = [header.index(feature) for feature in exclude_feature]
    header = [col for col in header if col not in exclude_feature]
    # Group the rows by location
    location_data = defaultdict(list)
    for row in rows:
        row = [value for i, value in enumerate(row) if i not in exclude_indices]
        location_data[row[header.index("location")]].append(row)

    location_data = location_data

    # Initialize lists for training and testing data
    train_rows = []
    test_rows = []

    # Split each location's data
    for location, data in location_data.items():
        # Check if time_granularity is "W"
        # if time_granularity == "W":
        #     # Sort the data by 'time_period' to ensure chronological order
        #     data.sort(
        #         key=lambda row: datetime.strptime(
        #             row[header.index("time_period")],  # Week format like '2024W1'
        #             "%GW%V",  # Custom format for weeks (without adding a day)
        #         )
        #     )
        # else:
        #     # For other granularities (e.g., "D", "M"), apply a different sort logic if needed
        #     data.sort(
        #         key=lambda row: datetime.strptime(
        #             row[header.index("time_period")],
        #             DATEFORMAT[
        #                 time_granularity
        #             ],  # Use appropriate format for "D" or "M"
        #         )
        #     )

        # Split the data into train and test based on test_size
        test_size_count = int(len(data) * test_size)

        # Take the first 'test_size_count' for the test set and the rest for the train set
        train_rows.extend(data[:-test_size_count])  # 80% for training
        test_rows.extend(data[-test_size_count:])

    # Write the training data to a new file
    train_file = (
        f"{directory_to_store}/{file.split('/')[-1].replace('.csv', '_train.csv')}"
    )
    with open(train_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_rows)

    # Write the X_test data (excluding "disease_cases") to a new file
    test_file = (
        f"{directory_to_store}/{file.split('/')[-1].replace('.csv', '_test.csv')}"
    )
    with open(test_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test_rows)

    # Write the Y_test data (including "time_period", "location", and "disease_cases") to a new file
    # y_test_file = (
    #     f"{directory_to_store}/{file.split('/')[-1].replace('.csv', '_y_test.csv')}"
    # )
    # with open(y_test_file, mode="w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["time_period", "location", "disease_cases"])  # Updated header
    #     for row in test_rows:
    #         # Include time_period, location, and disease_cases in the y_test file
    #         writer.writerow(
    #             [
    #                 row[header.index("time_period")],
    #                 row[header.index("location")],
    #                 row[header.index("disease_cases")],
    #             ]
    #         )

    print(f"Data split completed by location. Files saved in {directory_to_store}")
    return train_file, test_file


def generate_report(
    simulation,
    folder_path,
    model_name,
):
    data_file = f"{folder_path}{simulation.simulation_name}/dataset.csv"
    y_test_file = f"{folder_path}{simulation.simulation_name}/dataset_y_test.csv"
    sample_0_file = f"{folder_path}{simulation.simulation_name}/predictions.csv"
    dir_path = f"{folder_path}{simulation.simulation_name}"
    pdf_file = (
        f"{dir_path}/{model_name.replace("/","_")}_{simulation.simulation_name}.pdf"
    )

    features_to_plot = []

    data_df = pd.read_csv(data_file)
    y_test_df = pd.read_csv(y_test_file)
    sample_0_df = pd.read_csv(sample_0_file)

    merged_df = pd.merge(
        y_test_df, sample_0_df, on=["time_period", "location"], how="left"
    )

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Prepare to store evaluation metrics for all locations
    all_metrics = []

    # Plot the data and evaluate the model for each location
    locations = merged_df["location"].unique()

    for location in locations:
        location_data = merged_df[merged_df["location"] == location]
        location_data_full = data_df[data_df["location"] == location]

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

        # Subplot 1: Time Series of Actual vs Predicted
        plt.subplot(2, 1, 1)
        plt.plot(
            location_data_full["time_period"],
            location_data_full["disease_cases"],
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
        plt.xticks(data_df["time_period"][::5], rotation=45)
        plt.grid(True)

        # Subplot 2: Residuals (Error over Time)
        residuals = actual - predicted
        plt.subplot(2, 1, 2)
        plt.plot(
            location_data["time_period"], residuals, label="Residuals", color="purple"
        )
        plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
        plt.title(f"{location} - Residuals (Actual - Predicted)")
        plt.xlabel("Time Period")
        plt.ylabel("Residuals")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)

        plt.tight_layout()
        plot_file = f"{dir_path}/{location}_plot.png"
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
    pdf.cell(200, 10, txt=f"Model: {model_name}", ln=True, align="C")
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
        pdf.cell(0, 8, f"Neighbours: {', '.join(map(str, region.neighbour))}", ln=True)

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
        plot_file = f"{dir_path}/{location}_plot.png"
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"{location} - Time Series and Residuals", ln=True, align="C")
        pdf.ln(10)
        pdf.image(plot_file, x=10, y=30, w=190)  # Adjust positioning as needed

    # Save the PDF
    pdf.output(pdf_file)
    print(f"PDF report saved to: {pdf_file}")


def generate_report_v2(
    simulation,
    folder_path,
    model_name,
):
    data_file = f"{folder_path}{simulation.simulation_name}/dataset.csv"
    y_test_file = f"{folder_path}test1/dataset.csv"
    sample_0_file = f"{folder_path}test1/predictions.csv"
    dir_path = f"{folder_path}{simulation.simulation_name}"
    pdf_file = (
        f"{dir_path}/{model_name.replace("/","_")}_{simulation.simulation_name}.pdf"
    )

    features_to_plot = []

    data_df = pd.read_csv(data_file)
    y_test_df = pd.read_csv(y_test_file)
    sample_0_df = pd.read_csv(sample_0_file)

    merged_df = pd.merge(
        y_test_df, sample_0_df, on=["time_period", "location"], how="left"
    )

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Prepare to store evaluation metrics for all locations
    all_metrics = []

    # Plot the data and evaluate the model for each location
    locations = merged_df["location"].unique()

    for location in locations:
        location_data = merged_df[merged_df["location"] == location]
        location_data_full = data_df[data_df["location"] == location]

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

        # Subplot 1: Time Series of Actual vs Predicted
        plt.subplot(2, 1, 1)
        plt.plot(
            location_data_full["time_period"],
            location_data_full["disease_cases"],
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
        plt.xticks(data_df["time_period"][::5], rotation=45)
        plt.grid(True)

        # Subplot 2: Residuals (Error over Time)
        residuals = actual - predicted
        plt.subplot(2, 1, 2)
        plt.plot(
            location_data["time_period"], residuals, label="Residuals", color="purple"
        )
        plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
        plt.title(f"{location} - Residuals (Actual - Predicted)")
        plt.xlabel("Time Period")
        plt.ylabel("Residuals")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)

        plt.tight_layout()
        plot_file = f"{dir_path}/{location}_plot.png"
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
    pdf.cell(200, 10, txt=f"Model: {model_name}", ln=True, align="C")
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
        pdf.cell(0, 8, f"Neighbours: {', '.join(map(str, region.neighbour))}", ln=True)

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
        plot_file = f"{dir_path}/{location}_plot.png"
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"{location} - Time Series and Residuals", ln=True, align="C")
        pdf.ln(10)
        pdf.image(plot_file, x=10, y=30, w=190)  # Adjust positioning as needed

    # Save the PDF
    pdf.output(pdf_file)
    print(f"PDF report saved to: {pdf_file}")


def get_model_from_directory_or_github_url(
    model_path,
    base_working_dir=Path("runs/"),
    ignore_env=False,
    run_dir_type: Literal["timestamp", "latest", "use_existing"] = "timestamp",
):
    """
    NB! Copied from chap-core repository, but does not throw error if model doesn't have MLProject file

    Gets the model and initializes a working directory with the code for the model.
    model_path can be a local directory or github url

    Parameters
    ----------
    model_path : str
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

    is_github = False
    commit = None
    if isinstance(model_path, str) and model_path.startswith("https://github.com"):
        dir_name = model_path.split("/")[-1].replace(".git", "")
        model_name = dir_name
        if "@" in model_path:
            model_path, commit = model_path.split("@")
        is_github = True
    else:
        model_name = Path(model_path).name

    if run_dir_type == "use_existing" and not Path(model_path).exists():
        run_dir_type = "latest"

    if run_dir_type == "latest":
        working_dir = base_working_dir / model_name / "latest"
        # clear working dir
        if working_dir.exists():
            shutil.rmtree(working_dir)
    elif run_dir_type == "timestamp":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_identifier = timestamp + "_" + str(uuid.uuid4())[:8]
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S%f")
        working_dir = base_working_dir / model_name / unique_identifier
        # check that working dir does not exist
        assert (
            not working_dir.exists()
        ), f"Working dir {working_dir} already exists. This should not happen if make_run_dir is True"
    elif run_dir_type == "use_existing":
        working_dir = Path(model_path)
    else:
        raise ValueError(f"Invalid run_dir_type: {run_dir_type}")

    if is_github:
        working_dir.mkdir(parents=True)
        repo = git.Repo.clone_from(model_path, working_dir)
        if commit:
            repo.git.checkout(commit)

    else:
        # copy contents of model_path to working_dir
        shutil.copytree(model_path, working_dir)

    assert os.path.isdir(working_dir), working_dir
    assert os.path.isdir(os.path.abspath(working_dir)), working_dir
    # assert that a config file exists
    if (working_dir / "MLproject").exists():
        return get_model_from_mlproject_file(
            working_dir / "MLproject", ignore_env=ignore_env
        )
    else:
        return working_dir
