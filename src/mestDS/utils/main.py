import csv
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


def train_test_split_csv(file, directory_to_store, test_size=0.2):
    """
    Splits a CSV file into train, x_test, and y_test with an 80/20 split for each location,
    and includes location and time_period in the y_test file.
    The data is first sorted by time_period for correct chronological split.
    Args:
    - file (str): The path to the CSV file to be split.
    - directory_to_store (str): The directory where the split files should be saved.
    - test_size (float): The proportion of the data for each location to be used for testing (20% by default).
    """
    header = [
        "time_period",
        "rainfall",
        "mean_temperature",
        "disease_cases",
        "location",
    ]

    # Read the entire file into a list of rows
    with open(file, mode="r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        rows = list(reader)

    # Group the rows by location
    location_data = defaultdict(list)
    for row in rows:
        location_data[row[header.index("location")]].append(row)

    # Initialize lists for training and testing data
    train_rows = []
    test_rows = []

    # Split each location's data
    for location, data in location_data.items():
        # Sort the data by 'time_period' to ensure chronological order
        data.sort(key=lambda row: row[header.index("time_period")])

        # Split the data into train and test based on test_size
        test_size_count = int(len(data) * test_size)

        # Take the first 'test_size_count' for the test set and the rest for the train set
        test_rows.extend(data[:test_size_count])  # 20% for testing
        train_rows.extend(data[test_size_count:])  # 80% for training

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


def plot_data_with_sample_0(
    data_file, sample_0_file, dir_path, split_regions=True, subtitle=""
):
    """
    Plots data from the main CSV file and adds the 'sample_0' column from another CSV file.
    Args:
    - data_file (str): The path to the main CSV file containing time_period, location, disease_cases, etc.
    - sample_0_file (str): The path to the CSV file containing the 'sample_0' column.
    """
    # Load the main data CSV file
    data_df = pd.read_csv(data_file)

    # Load the 'sample_0' column from a separate CSV file
    sample_0_df = pd.read_csv(sample_0_file)

    # Merge the data based on 'time_period' and 'location' (assuming these columns are common)
    merged_df = pd.merge(
        data_df, sample_0_df, on=["time_period", "location"], how="left"
    )

    # Plot the data for each location
    locations = merged_df["location"].unique()

    # Loop through each location and plot its data
    for location in locations:
        location_data = merged_df[merged_df["location"] == location]

        plt.figure(figsize=(12, 8))
        # Plot disease_cases, rainfall, mean_temperature, and sample_0 over time
        plt.plot(
            location_data["time_period"],
            location_data["disease_cases"],
            label=f"{location} - Disease Cases",
        )
        plt.plot(
            location_data["time_period"],
            location_data["sample_0"],
            label=f"{location} - Predicated Cases",
            linestyle=":",
        )
        # Add labels and legend
        plt.suptitle("Disease Cases and Predicted Disease Cases over Time")
        plt.title(subtitle)
        plt.xlabel("Time Period")
        plt.ylabel("Values")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.savefig(f"{dir_path}/{location}.png")
        plt.close()
