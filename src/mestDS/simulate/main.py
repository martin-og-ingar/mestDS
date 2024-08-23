from collections import defaultdict
import math
import random
from ..classes.ClimateHealthData_module import ClimatHealthData
import numpy as np


def test():
    data = ClimatHealthData([1, 2, 3], [1, 2, 3], [1, 2, 3])
    return data


def generate_data(season_enabled, length):
    data = ClimatHealthData([], [], [])
    data.precipitation = [random.randint(0, 100)]
    data.sickness = [random.randint(50, 100)]
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
            rain = random.randint(0, 50)
            return rain
    else:
        return random.randint(0, 100)


def get_rain_prob(i):
    week = get_weeknumber(i)
    if week > 11 and week < 24:
        return 0.7
    elif week > 36 and week < 40:
        return 0.8
    else:
        return 0.3


# Generate temperature data
def get_temp(week):
    week = get_weeknumber(week)
    month = week / 4.33

    if month < 1:
        return 23.72
    elif month < 2:
        return 24.26
    elif month < 3:
        return 24.25
    elif month < 4:
        return 23.71
    elif month < 5:
        return 23.18
    elif month < 6:
        return 22.67
    elif month < 7:
        return 22.31
    elif month < 8:
        return 22.68
    elif month < 9:
        return 22.86
    elif month < 10:
        return 23.16
    elif month < 11:
        return 23.21
    else:
        return 23.03


# Generate sickness data
def get_sickness(sickness, input, weight):
    sum = np.dot(input, weight)
    if sum > 65.2:
        sickness = sickness + random.randint(6, 10)
    elif sum >= 49.2:
        sickness = sickness + random.randint(0, 5)
    elif sum < 33.2:
        sickness = sickness + random.randint(-10, -6)
    elif sum < 49.2:
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


# calculate average
def calculate_weekly_averages(data):
    average_data = ClimatHealthData([], [], [])
    for i in range(52):
        average_data.precipitation.append(0)
        average_data.sickness.append(0)
        average_data.temperature.append(0)

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
