from collections import defaultdict
from datetime import datetime
import argparse
from mestDS.classes.ClimateHealthData_module import (
    Obs,
    toGluonTsFormat,
    toDataSetFromat,
)
from numpy import mean
from mestDS import generate_data, graph, calculate_weekly_averages
import sys
from typing import List, Dict, Any


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


def main():
    args = get_arguments()
    if not args:
        # print("No arguments provided, Usage: [number of runs] [average]")
        sys.exit(1)

    runs = args.runs
    regions = args.regions
    enable_seasonality = args.enable_seasonality == "True"

    duration = args.duration
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    time_granularity = args.time_granularity

    all_data = {}
    for reg in regions:
        region_avg = []
        all_data[reg] = []

        for i in range(int(runs)):
            data = generate_data(
                reg, enable_seasonality, start_date, duration, time_granularity
            )
            all_data[reg].extend(data[reg])
            region_avg.append(data)

            run_averages = calculate_average([data])

            for country, avg_data in run_averages.items():
                print(
                    f"Run {i+1}, Averages: {{'{country}': {{'sickness': {avg_data['sickness']:.2f}, 'rainfall': {avg_data['rainfall']:.2f}, 'temperature': {avg_data['temperature']:.2f}}}}}"
                )

        cummlative_averages = calculate_average(region_avg)
        for country, avg_data in cummlative_averages.items():
            print(f"Country: {country}")
            print(f"  Average Sickness: {avg_data['sickness']:.2f}")
            print(f"  Average Rainfall: {avg_data['rainfall']:.2f}")
            print(f"  Average Temperature: {avg_data['temperature']:.2f}")

    print("Done")

    print(all_data)
    """
    combined_data = defaultdict(list)
    for d in all_data:
        for region, observations in d.items():
            combined_data[region].extend(observations)
    combined_data = dict(combined_data)
    dataset = toDataSetFromat(combined_data)
    print(dataset)

    """
    observation_dict = {
        "Oslo": [
            Obs(
                time_period="2020-01-01", disease_cases=10, rainfall=0.1, temperature=20
            ),
            Obs(
                time_period="2020-01-02", disease_cases=11, rainfall=0.2, temperature=22
            ),
            Obs(
                time_period="2020-01-03", disease_cases=12, rainfall=0.3, temperature=21
            ),
        ],
        "Troms": [
            Obs(
                time_period="2020-01-01", disease_cases=2, rainfall=1.1, temperature=10
            ),
            Obs(
                time_period="2020-01-02", disease_cases=2, rainfall=2.2, temperature=11
            ),
            Obs(
                time_period="2020-01-03", disease_cases=2, rainfall=0.3, temperature=12
            ),
        ],
    }
    print("DICT")
    print(observation_dict)
    # Note, can ONLY use toDataSetFromat for one single run. not if the dict contains multiple runs.
    dataset = toDataSetFromat(observation_dict)
    print("DATASET2")
    dataset2 = toDataSetFromat(all_data)
    gluon = toGluonTsFormat(dataset2)
    print(list(gluon))


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


def merge_observations(input_data: List[Dict[str, List[Obs]]]) -> Dict[str, List[Obs]]:
    merged_data = {}
    for run_data in input_data:
        for region, observation in run_data.items():
            if region not in merged_data:
                merged_data[region] = []
            merged_data[region].extend(observation)
    return merged_data


if __name__ == "__main__":
    main()
