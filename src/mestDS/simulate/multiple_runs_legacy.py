from collections import defaultdict
import argparse
import random
from mestDS.classes.ClimateHealthData import to_gluonTS_format, to_dataset_format
from numpy import mean
from .main import generate_data


def get_arguments():
    parser = argparse.ArgumentParser(description="Climate Data generation")
    parser.add_argument("runs", type=int, help="Number of runs")
    parser.add_argument(
        "regions", type=str, nargs="+", help="Regions to generate data for"
    )

    parser.add_argument(
        "enable_seasonality",
        type=str,
        choices=["True", "False"],
        help="Choose True/False for seasonallity",
    )
    parser.add_argument(
        "rain_season_randomness", type=str, help="Choose True/False for randomness"
    )
    parser.add_argument(
        "start_date", type=str, help="start date is in the format YYYYMMDD"
    )
    parser.add_argument(
        "duration",
        type=int,
        help="how many days/weeks/months/years to generate data for",
    )
    parser.add_argument(
        "time_granularity",
        type=str,
        choices=["D", "W", "M"],
        help="How to group the data.",
    )
    return parser.parse_args()


def generate_multiple_datasets(
    regions,
    enable_seasonality,
    rain_season_randomness,
    start_date,
    duration,
    time_granularity,
    sickness_pessimism,
    sickness_agressiveness,
    sickness_increase,
):

    all_data = []
    for sp in sickness_pessimism:
        for sa in sickness_agressiveness:
            for si in sickness_increase:
                # current_run = {}
                if rain_season_randomness:
                    rainy_season_1, rainy_season_2 = randomIntervals()
                else:
                    rainy_season_1, rainy_season_2 = (11, 24), (36, 40)

                    # current_run[reg] = []

                data = generate_data(
                    regions,
                    enable_seasonality,
                    rainy_season_1,
                    rainy_season_2,
                    start_date,
                    duration,
                    time_granularity,
                    sp,
                    sa,
                    si,
                )
                # current_run[reg].extend(data[reg])
                current_run_with_parameters = {
                    "sp": sp,
                    "sa": sa,
                    "si": si,
                    "data": data,
                }
                all_data.append(current_run_with_parameters)
    return all_data


def randomIntervals():
    rain_season_start = random.randint(8, 18)
    rain_season_duration = random.randint(6, 10)
    rainy_season_1 = (rain_season_start, rain_season_start + rain_season_duration)

    rain_season_start2 = random.randint(30, 40)
    rain_season_duration_2 = random.randint(4, 6)
    rainy_season_2 = (rain_season_start2, rain_season_start2 + rain_season_duration_2)

    return rainy_season_1, rainy_season_2


def convert_datasets_to_gluonTS(datasets):
    converted_data = []
    for ds in datasets:
        data_set = to_dataset_format(ds["data"])
        gluon_data = to_gluonTS_format(data_set)
        converted_data.append(gluon_data)
    return converted_data


if __name__ == "__main__":
    generate_multiple_datasets()
