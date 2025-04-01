import numpy as np
from chap_core.data.datasets import ISIMIP_dengue_harmonized


def constant(t, value=0):
    return np.full_like(t, value, dtype=float)


def random_walk(t):
    walk = np.cumsum(np.random.normal(0, 1, size=t.shape))
    walk = np.abs(walk)
    return walk


def normal_distribution(mean, std_dev, t=None, current_i=None):
    return np.random.normal(mean, std_dev)


def poisson_distribution(lam, scale, t=None, current_i=None):
    events = np.random.poisson(lam)
    rainfall = events * np.random.exponential(scale)
    return rainfall


def exponential_growth(rate, t):
    return np.exp(rate * t)


def extreme_event(probability, magnitude, t=None, current_i=None):
    return magnitude if np.random.rand() < probability else 0


def seasonal(average, amplitude, phase, noise, current_i, t=None):
    n = np.random.normal(0, noise)
    shift = phase * np.pi
    sine_wave = average + amplitude * np.sin(2 * np.pi * current_i / 12 + shift) + n
    return sine_wave


def seasonal_disease_cases(average, amplitude, phase, noise, current_i, t):
    shift = phase * np.pi
    disease_cases = []

    for current_i in range(t):
        noise = np.random.normal(0, 0.5)

        sine_wave = np.int32(
            average + amplitude * np.sin(2 * np.pi * current_i / 12 + shift) + noise
        )
        disease_cases.append(sine_wave)
    return disease_cases


def spike(magnitude, t, current_i, spike_position=0.5):
    spike_index = int(spike_position * t)
    return magnitude if current_i == spike_index else 0


def trend(rate, t):
    return rate * t


def stochastic_noise(mean, std_dev, t=None, current_i=None):
    return np.random.normal(mean, std_dev)


def autoregression(phi, noise_std, current_i, history, population, t=None):
    p = len(phi)
    if len(history) < p:
        history = np.concatenate([np.random.normal(0, 1, p - len(history)), history])
        print("here?")
    disease_case_array = []

    print(history)

    for i in range(p, len(history)):
        ar_sum = sum(phi[j] * history[i - (j + 1)] for j in range(p))

        noise = np.random.normal(0, noise_std)

        next_value = ar_sum + noise
        disease_case_array.append(next_value)

    print(disease_case_array)
    return disease_case_array


def climate_dependent_disease_cases(
    lags,
    population,
    features,
    auto_regressive,
    phi,
    current_i,
    t,
):
    poisson_rate = np.zeros_like(next(iter(features.values())))
    for (feature_name, covariate), lag in zip(features.items(), lags):
        print("Using feature: ", feature_name)
        lagged_covariate = apply_lag(covariate, lag)
        poisson_rate += lagged_covariate / np.max(lagged_covariate)  # scale

    disease_cases = apply_sigmoid_capping(poisson_rate, population)

    disease_cases[disease_cases > population] = population

    if auto_regressive:
        p = len(phi)
        ar_cases = disease_cases.copy()

        for i in range(p, len(disease_cases)):
            ar_sum = sum(phi[j] * disease_cases[i - (j + 1)] for j in range(p))
            noise = np.random.normal(0, 1)
            ar_cases[i] = ar_sum + noise
        return ar_cases

    return disease_cases


def apply_sigmoid_capping(disease_cases, population):
    disease_cases = apply_sigmoid(disease_cases, population)
    disease_cases = np.random.poisson(disease_cases)
    # disease_cases[disease_cases > population] = population
    return disease_cases


def apply_sigmoid(disease_cases, population):
    disease_cases = np.int32((1 / (1 + np.exp(-disease_cases))) * population)
    return disease_cases


def apply_lag(data, lag: int):
    return np.roll(data, lag)


def realistic_data_generation(feature_name, country, t, current_i=None):
    lower_case_country = country.lower()
    print(lower_case_country)
    df = ISIMIP_dengue_harmonized[lower_case_country].to_pandas()
    feature = df[feature_name].values[:t]
    feature = (feature / feature.max()) * 4
    return feature


FUNCTION_POOL = {
    "constant": constant,
    "random_walk": random_walk,
    "seasonal": seasonal,
    "seasonal_disease_cases": seasonal_disease_cases,
    "trend": trend,
    "spike": spike,
    "stochastic_noise": stochastic_noise,
    "normal_distribution": normal_distribution,
    "poisson_distribution": poisson_distribution,
    "exponential_growth": exponential_growth,
    "extreme_event": extreme_event,
    "autoregression": autoregression,
    "realistic_data_generation": realistic_data_generation,
    "climate_dependent_disease_cases": climate_dependent_disease_cases,
}
