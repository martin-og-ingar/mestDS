from collections import defaultdict
import datetime

from numpy import mean
from mestDS import generate_data, graph, calculate_weekly_averages
import sys


def main():
    args = sys.argv[1:]
    if not args:
        print("No arguments provided, Usage: [number of runs] [average]")
        sys.exit(1)

    action = args[0]
    all_data = []

    for i in range(int(action)):
        start_date = datetime.date(2024, 1, 1)
        data = generate_data(True, 10000, start_date, "W")
        all_data.append(data)

        averages = calculate_average(all_data)

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
