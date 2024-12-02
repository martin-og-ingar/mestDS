import math
import random
from ..classes.ClimateHealthData import Obs
from ..default_variables import DATEFORMAT, TIMEDELTA, DEFAULT_TEMPERATURES
import numpy as np
from chap_core.data import DataSet, PeriodObservation
from datetime import datetime
from ..classes.Simulation import Simulation, RainSeason


def generate_data(simulation: Simulation):
    data_observation = {region: [] for region in simulation.regions}
    for region in simulation.regions:
        precipitation = random.randint(0, 100)
        sickness = random.randint(10, 15)
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
        for region in simulation.regions:
            week_number = get_weeknumber(i, simulation.time_granularity)
            rain_season = is_rain_season(week_number, simulation.rain_season)

            precipitation = get_precipitation(rain_season)
            temperature = get_temperature_new(week_number)
            total_neighbour_influence = get_influence_from_neighbours(
                region_index=simulation.regions.index(region),
                t=i,
                regions=simulation.regions,
                neighbors=simulation.neighbors,
                simulated_data=data_observation,
            )
            sickness = get_disease_cases_new(
                prev_sickness=data_observation[region][i - 1].disease_cases,
                rainfall=precipitation,
                temperature=temperature,
                intercept=0,
                beta_rainfall=simulation.beta_rainfall,
                beta_temp=simulation.beta_temp,
                beta_lag_sickness=simulation.beta_lag_sickness,
                beta_neighbour_influence=simulation.beta_neighbour_influence,
                neighbour_sickness=total_neighbour_influence,
                noise_std=simulation.noise_std,
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


def get_weeknumber(i, time_granularity):
    if time_granularity == "D":
        return ((i - 1) // 7 + 1) % 52
    else:
        return i % 52


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


def get_temperature_new(week_number):
    seasonal_temp = 24 + 5 * np.sin(2 * np.pi * week_number / 52)

    random_noise = np.random.normal(0, 2)
    return seasonal_temp + random_noise


def get_rainfall_new(rain_season):
    if rain_season:
        return np.random.gamma(shape=2, scale=1.0) * 2
    else:
        return np.random.gamma(shape=2, scale=0.5) * 0.5


def get_influence_from_neighbours(region_index, t, regions, neighbors, simulated_data):
    if t == 0:
        return 0

    neighbours = np.where(neighbors[region_index] == 1)[0]
    total_influence = 0

    for neighbour_index in neighbours:
        neighbour_region = regions[neighbour_index]

        if t - 1 < len(simulated_data[neighbour_region]):
            neighbour_sickness = simulated_data[neighbour_region][t - 1].disease_cases

            total_influence += neighbour_sickness

    return total_influence / len(neighbours)


def get_disease_cases_new(
    prev_sickness,
    rainfall,
    temperature,
    intercept,
    beta_rainfall,
    beta_temp,
    beta_lag_sickness,
    beta_neighbour_influence,
    neighbour_sickness,
    noise_std,
):
    noise = np.random.laplace(0, noise_std)

    sickness = (
        intercept
        + beta_rainfall * rainfall
        + beta_temp * temperature
        + beta_lag_sickness * prev_sickness
        + beta_neighbour_influence * neighbour_sickness
        + noise
    )
    return round(sickness)
