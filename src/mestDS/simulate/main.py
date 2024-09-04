from collections import defaultdict
import math
import random
from ..classes.ClimateHealthData_module import ClimatHealthData, Obs
from ..classes.default_variables import DEFAULT_TEMPERATURES, TIMEDELTA, DATEFORMAT
import numpy as np
from climate_health.data import DataSet, PeriodObservation
from datetime import datetime, timedelta


def generate_data(season_enabled, length, start_date, period):
    data_observation = {"Uganda": []}
    precipitation = random.randint(0, 100)
    sickness = random.randint(50, 100)
    temperature = random.randint(20, 30)
    start_date_formatted = datetime.strftime(start_date, DATEFORMAT)
    obs = Obs(
        time_period=start_date_formatted,
        disease_cases=sickness,
        rainfall=precipitation,
        temperature=temperature,
    )
    data_observation["Uganda"].append(obs)

    period = "W" if period is None else period
    delta = TIMEDELTA[period]

    for i in range(1, length):
        precipitation = get_precipitation(season_enabled, i)
        temperature = get_temp(i)
        input = np.array([precipitation, temperature])
        weight = np.array([0.7, 0.3])
        sickness = get_sickness(
            data_observation["Uganda"][i - 1].disease_cases, input, weight
        )
        current_date = start_date + (i * delta)
        current_date = datetime.strftime(current_date, DATEFORMAT)
        obs = Obs(
            time_period=current_date,
            disease_cases=sickness,
            rainfall=precipitation,
            temperature=temperature,
        )
        data_observation["Uganda"].append(obs)
    return data_observation


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
    average_data = {"Uganda": []}
    for i in range(52):
        obs = Obs(time_period=str(i), disease_cases=0, rainfall=0, temperature=0)
        average_data["Uganda"].append(obs)

    if (len(data["Uganda"])) < 52:
        raise Exception("Length is under 52, does not calculate average")

    for i in range(len(data["Uganda"])):
        week_number = get_weeknumber(i + 1)
        week_number = 52 if (week_number % 52 == 0) else week_number % 52
        average_data["Uganda"][week_number - 1].rainfall += data["Uganda"][i].rainfall
        average_data["Uganda"][week_number - 1].disease_cases += data["Uganda"][
            i
        ].disease_cases
        average_data["Uganda"][week_number - 1].temperature += data["Uganda"][
            i
        ].temperature

    # Calculate averages
    for i in range(len(average_data["Uganda"])):
        divider = get_divider(i, data)
        average_data["Uganda"][i].rainfall = (
            average_data["Uganda"][i].rainfall / divider
        )
        average_data["Uganda"][i].disease_cases = (
            average_data["Uganda"][i].disease_cases / divider
        )
        average_data["Uganda"][i].temperature = (
            average_data["Uganda"][i].temperature / divider
        )
    return average_data


def get_divider(i, data):
    decimal, whole_number = math.modf(len(data["Uganda"]) / 52)
    limit = 52 * decimal
    if i < limit:
        return whole_number + 1
    else:
        return whole_number
