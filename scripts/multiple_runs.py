from collections import defaultdict
from datetime import datetime
import argparse
import random
from mestDS.classes.ClimateHealthData_module import toGluonTsFormat, toDataSetFormat
from numpy import mean
from mestDS import generate_data
import sys


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


def getGluonTsFormat():
    args = get_arguments()
    if not args:
        sys.exit(1)

    runs = args.runs
    regions = args.regions
    enable_seasonality = args.enable_seasonality == "True"
    rain_season_randomness = args.rain_season_randomness == "True"
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    duration = args.duration
    time_granularity = args.time_granularity

    all_data = {}
    gluon_list = []
    for i in range(int(runs)):
        if rain_season_randomness:
            rainy_season_1, rainy_season_2 = randomIntervals()
        else:
            rainy_season_1, rainy_season_2 = (11, 24), (36, 40)

        for reg in regions:
            all_data[reg] = []

            data = generate_data(
                reg,
                enable_seasonality,
                rainy_season_1,
                rainy_season_2,
                start_date,
                duration,
                time_granularity,
            )
            all_data[reg].extend(data[reg])
        toDataSet = toDataSetFormat(all_data)
        gluon = toGluonTsFormat(toDataSet)
        gluon_list.append(gluon)

    return gluon_list


def randomIntervals():
    rain_season_start = random.randint(8, 18)
    rain_season_duration = random.randint(6, 10)
    rainy_season_1 = (rain_season_start, rain_season_start + rain_season_duration)

    rain_season_start2 = random.randint(30, 40)
    rain_season_duration = random.randint(4, 6)
    rainy_season_2 = (rain_season_start2, rain_season_start2 + rain_season_duration)

    return rainy_season_1, rainy_season_2


# Remove?
def calculate_average(data):
    agg_sickness = defaultdict(list)
    agg_rainfall = defaultdict(list)
    agg_temperature = defaultdict(list)
    for d in data:
        for country, observations in d.items():
            for obs in observations:
                agg_sickness[country].append(obs.disease_cases)
                agg_rainfall[country].append(obs.rainfall)
                agg_temperature[country].append(obs.temperature)
    averages = {}
    for c in agg_sickness:
        avg_sick = mean(agg_sickness[c])
        avg_rain = mean(agg_rainfall[c])
        avg_temp = mean(agg_temperature[c])

        averages[c] = {
            "sickness": avg_sick,
            "rainfall": avg_rain,
            "temperature": avg_temp,
        }
    return averages


if __name__ == "__main__":
    getGluonTsFormat()
