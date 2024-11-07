import math
import random
from ..classes.ClimateHealthData_module import Obs
from ..classes.SimulationInput import DEFAULT_TEMPERATURES, TIMEDELTA, DATEFORMAT

import numpy as np
from chap_core.data import DataSet, PeriodObservation
from datetime import datetime
from ..classes.SimulationInput import Simulation, RainSeason


def generate_data(simulation: Simulation):
    data_observation = {region: [] for region in simulation.regions}
    for region in simulation.regions:
        precipitation = random.randint(0, 100)
        sickness = random.randint(50, 100)
        temperature = random.randint(20, 30)
        start_date_formatted = datetime.strftime(
            simulation.simulation_start_date, DATEFORMAT
        )
        obs = Obs(
            time_period=start_date_formatted,
            disease_cases=sickness,
            rainfall=precipitation,
            mean_temperature=temperature,
            population=10000,
        )
        data_observation[region].append(obs)

        delta = TIMEDELTA[simulation.time_granularity]

        for i in range(1, simulation.simulation_length):
            week_number = get_weeknumber(i)
            rain_season = is_rain_season(week_number, simulation.rain_season)

            precipitation = get_precipitation(rain_season)
            temperature = get_temp(week_number)

            input = np.array([precipitation, temperature])
            weight = np.array([0.7, 0.3])
            sickness = get_sickness(
                data_observation[region][i - 1].disease_cases,
                input,
                weight,
                simulation.normal_dist_mean,
                simulation.normal_dist_stddev,
                simulation.nomral_dist_scale,
            )
            current_date = simulation.simulation_start_date + (i * delta)
            current_date = datetime.strftime(current_date, DATEFORMAT)
            obs = Obs(
                time_period=current_date,
                disease_cases=sickness,
                rainfall=precipitation,
                mean_temperature=temperature,
                population=10000,
            )
            data_observation[region].append(obs)
    return data_observation


# Generate precepitation data
def get_precipitation(rain_season):
    # use gamma if seasons are enabled.
    noise = np.random.randint(-5, 5)
    if rain_season:
        shape, scale = (10, 7)
    else:
        shape, scale = (8, 2)
    rain = np.random.gamma(shape, scale)

    rain = max(0, rain + noise)
    return rain


# Generate temperature data
def get_temp(week):
    month = get_monthnumber(week)
    temperature = DEFAULT_TEMPERATURES[month]
    temperature += random.randint(-3, 3)
    return temperature


# Generate sickness data
#
def get_sickness(sickness, input, weight, sp, sa, si):
    sum = np.dot(input, weight)
    max_dot = 77.28
    # random_noise = np.clip(np.random.normal((sum / max_dot) - sp, sa), -3, 3)
    # sickness = sickness + int(random_noise * si)
    sickness = sickness + int(np.random.normal((sum / max_dot) - sp, sa) * si)
    sickness = max(min(sickness, 1000), 1)

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
def calculate_weekly_averages(data, region):
    average_data = {region: []}
    for i in range(52):
        obs = Obs(time_period=str(i), disease_cases=0, rainfall=0, temperature=0)
        average_data[region].append(obs)

    if (len(data[region])) < 52:
        raise Exception("Length is under 52, does not calculate average")

    for i in range(len(data[region])):
        week_number = get_weeknumber(i + 1)
        week_number = 52 if (week_number % 52 == 0) else week_number % 52
        average_data[region][week_number - 1].rainfall += data[region][i].rainfall
        average_data[region][week_number - 1].disease_cases += data[region][
            i
        ].disease_cases
        average_data[region][week_number - 1].temperature += data[region][i].temperature

    # Calculate averages
    for i in range(len(average_data[region])):
        divider = get_divider(i, data, region)
        average_data[region][i].rainfall = average_data[region][i].rainfall / divider
        average_data[region][i].disease_cases = (
            average_data[region][i].disease_cases / divider
        )
        average_data[region][i].temperature = (
            average_data[region][i].temperature / divider
        )
    return average_data


def get_divider(i, data, region):
    decimal, whole_number = math.modf(len(data[region]) / 52)
    limit = 52 * decimal
    if i < limit:
        return whole_number + 1
    else:
        return whole_number


def is_rain_season(week, rain_seasons):

    for season in rain_seasons:
        if season.start <= week <= season.end:
            return True
    return False
