from collections import defaultdict
from datetime import datetime
import argparse
from numpy import mean
from mestDS import generate_data, graph, calculate_weekly_averages
import sys


def get_arguments():
    parser = argparse.ArgumentParser(description="Climate Data generation")
    parser.add_argument("runs", type=int, help="Number of runs")

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
    enable_seasonality = args.enable_seasonality == "True"

    duration = args.duration
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    time_granularity = args.time_granularity

    all_data = []

    for i in range(int(runs)):
        data = generate_data(enable_seasonality, start_date, duration, time_granularity)
        all_data.append(data)

        averages = calculate_average(all_data)

        for country, avg_data in averages.items():
            print(
                f"Run {i+1}, Averages: {{'{country}': {{'sickness': {avg_data['sickness']:.2f}, 'rainfall': {avg_data['rainfall']:.2f}, 'temperature': {avg_data['temperature']:.2f}}}}}"
            )

    for country, avg_data in averages.items():
        print(f"Country: {country}")
        print(f"  Average Sickness: {avg_data['sickness']:.2f}")
        print(f"  Average Rainfall: {avg_data['rainfall']:.2f}")
        print(f"  Average Temperature: {avg_data['temperature']:.2f}")

    print("Done")


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
    main()
