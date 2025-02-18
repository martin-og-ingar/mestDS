import csv
from collections import defaultdict
from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from fpdf import FPDF


def train_test_split_csv(
    file,
    directory_to_store,
    exclude_feature,
    test_size=0.2,
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
        # Sort the data by 'time_period' to ensure chronological order
        data.sort(
            key=lambda row: datetime.strptime(
                row[header.index("time_period")], "%Y-%m-%d"
            )
        )

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
    x_test_file = (
        f"{directory_to_store}/{file.split('/')[-1].replace('.csv', '_x_test.csv')}"
    )
    with open(x_test_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [header[i] for i in range(len(header)) if header[i] != "disease_cases"]
        )
        for row in test_rows:
            writer.writerow(
                [row[i] for i in range(len(row)) if header[i] != "disease_cases"]
            )

    # Write the Y_test data (including "time_period", "location", and "disease_cases") to a new file
    y_test_file = (
        f"{directory_to_store}/{file.split('/')[-1].replace('.csv', '_y_test.csv')}"
    )
    with open(y_test_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_period", "location", "disease_cases"])  # Updated header
        for row in test_rows:
            # Include time_period, location, and disease_cases in the y_test file
            writer.writerow(
                [
                    row[header.index("time_period")],
                    row[header.index("location")],
                    row[header.index("disease_cases")],
                ]
            )

    print(f"Data split completed by location. Files saved in {directory_to_store}")


def generate_report(
    data_file,
    sample_0_file,
    dir_path,
    split_regions=True,
    subtitle="",
    model_name="",
    simulation_name="",
):
    """
    Plots data from the main CSV file, adds 'sample_0' predictions,
    evaluates model performance, and generates a combined PDF report.

    Args:
    - data_file (str): Path to the main CSV file containing time_period, location, disease_cases, etc.
    - sample_0_file (str): Path to the CSV file containing the 'sample_0' column.
    - dir_path (str): Directory path to save plots and reports.
    - split_regions (bool): Whether to split plots by region (default: True).
    - subtitle (str): Subtitle for the plots.
    """
    features_to_plot = []
    # Load the main data CSV file
    data_df = pd.read_csv(data_file)

    # Load the 'sample_0' column from a separate CSV file
    sample_0_df = pd.read_csv(sample_0_file)

    # Merge the data based on 'time_period' and 'location'
    merged_df = pd.merge(
        data_df, sample_0_df, on=["time_period", "location"], how="left"
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
        plt.xticks(rotation=45)
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
    pdf.set_font("Arial", "B", 16)
    pdf.cell(
        0,
        10,
        f"{model_name} - {simulation_name} - Evaluation Report",
        ln=True,
        align="C",
    )
    pdf.ln(10)

    # Subtitle
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 10, subtitle, ln=True, align="C")
    pdf.ln(10)

    # Add Table of Metrics
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

    # Add Plots for Each Location
    for location in locations:
        plot_file = f"{dir_path}/{location}_plot.png"
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"{location} - Time Series and Residuals", ln=True, align="C")
        pdf.ln(10)
        pdf.image(plot_file, x=10, y=30, w=190)  # Adjust positioning as needed

    # Save the PDF
    pdf.output(f"{dir_path}/model_evaluation_report.pdf")
    print(f"PDF report saved to: {dir_path}/model_evaluation_report.pdf")
