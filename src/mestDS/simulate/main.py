from collections import defaultdict
import math
import random
from ..classes.ClimateHealthData_module import ClimatHealthData
from ..classes.default_variables import DEFAULT_TEMPERATURES
import numpy as np


def test():
    data = ClimatHealthData([1, 2, 3], [1, 2, 3], [1, 2, 3])
    return data


def generate_data(season_enabled, length):
    data = ClimatHealthData([], [], [])
    data.precipitation = [random.randint(0, 100)]
    data.sickness = [random.randint(999, 1000)]
    data.temperature = [random.randint(20, 30)]

    for i in range(1, length):
        precipitation = get_precipitation(season_enabled, i)
        data.precipitation.append(precipitation)

        temperature = get_temp(i)
        data.temperature.append(temperature)
        input = np.array([precipitation, temperature])
        weight = np.array([0.7, 0.3])
        sickness = get_sickness(data.sickness[i - 1], input, weight)
        data.sickness.append(sickness)

    return data


# Generate precepitation data
def get_precipitation(season_enabled, week):
    if season_enabled == True:
        rain_prob = get_rain_prob(week)
        r = random.uniform(0.0, 1.00)
        if r < rain_prob:
            rain = random.randint(50, 100)
            return rain
        else:
            # contraint the decrease of rain during non-rainy season.
            rain = random.randint(15, 24)
            return rain
    else:
        return random.randint(0, 100)


def get_rain_prob(i):
    week = get_weeknumber(i)
    if week > 11 and week < 24:
        return 0.8
    elif week > 36 and week < 40:
        return 0.8
    else:
        return 0.4


# Generate temperature data
def get_temp(week):
    week = get_weeknumber(week)
    month = get_monthnumber(week)
    return DEFAULT_TEMPERATURES[month]


# Generate sickness data
#
def get_sickness(sickness, input, weight):
    sum = np.dot(input, weight)
    max_dot = 77.28
    # min_dot = 6.69

    if sum > 0.75 * max_dot:
        sickness = sickness + random.randint(4, 7)
    elif sum >= 0.5 * max_dot:
        sickness = sickness + random.randint(0, 4)
    elif sum < 0.25 * max_dot:
        sickness = sickness + random.randint(-10, -6)
    elif sum < 0.5 * max_dot:
        sickness = sickness + random.randint(-5, 0)
    if sickness < 3:
        return random.randint(0, 3)
    else:
        sickness = sickness + random.randint(-3, 3)
        return sickness


def get_weeknumber(week):
    week = week / 52
    week = 52 if ((week % 1) * 52 == 0) else (week % 1) * 52
    return round(week)


def get_monthnumber(week):
    if week == 52:
        return 11
    month = week / 4.33
    return math.floor(month)


# calculate average
def calculate_weekly_averages(data):
    average_data = ClimatHealthData([], [], [])
    for i in range(52):
        average_data.precipitation.append(0)
        average_data.sickness.append(0)
        average_data.temperature.append(0)

    if (len(data.precipitation)) < 52:
        raise Exception("Length is under 52, does not calculate average")
    print(len(data.precipitation))
    for i in range(len(data.precipitation)):
        week_number = get_weeknumber(i + 1)
        week_number = 52 if (week_number % 52 == 0) else week_number % 52
        average_data.precipitation[week_number - 1] += data.precipitation[i]
        average_data.sickness[week_number - 1] += data.sickness[i]
        average_data.temperature[week_number - 1] += data.temperature[i]

    # Calculate averages
    for i in range(len(average_data.sickness)):
        divider = get_divider(i, data)
        average_data.precipitation[i] = (average_data.precipitation)[i] / divider
        average_data.sickness[i] = average_data.sickness[i] / divider
        average_data.temperature[i] = average_data.temperature[i] / divider

    return average_data


def get_divider(i, data):
    decimal, whole_number = math.modf(len(data.precipitation) / 52)
    limit = 52 * decimal
    if i < limit:
        return whole_number + 1
    else:
        return whole_number
