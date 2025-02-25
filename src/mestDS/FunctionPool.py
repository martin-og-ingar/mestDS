import numpy as np
from chap_core.data.datasets import ISIMIP_dengue_harmonized


def constant(t, value=10):
    return np.full_like(t, value, dtype=float)


def random_walk(t):
    walk = np.cumsum(np.random.normal(0, 1, size=t.shape))
    walk = np.abs(walk)
    return walk


def normal_distribution(mean, std_dev):
    return np.random.normal(mean, std_dev)


def poisson_distribution(lam, scale, t=None, current_i=None):
    events = np.random.poisson(lam)
    rainfall = events * np.random.exponential(scale)
    return rainfall


def exponential_growth(rate, t):
    return np.exp(rate * t)


def lognormal(mean, std_dev):
    return np.random.lognormal(mean, std_dev)


def complex_seasonal(average, amplitude1, amplitude2, phase1, phase2, t, current_i):
    sine_wave1 = amplitude1 * np.sin(2 * np.pi * current_i / 365 + phase1)
    sine_wave2 = amplitude2 * np.sin(2 * np.pi * current_i / 365 + phase2)
    return average + sine_wave1 + sine_wave2


def picewivse_trend(rate1, rate2, switch, t, current_i):
    return np.where(
        current_i < switch, rate1 * t, rate2 * (t - switch) + rate1 * switch
    )


def extreme_event(probability, magnitude):
    return magnitude if np.random.rand() < probability else 0


def seasonal(average, amplitude, phase, current_i, t=None):
    sine_wave = average + amplitude * np.sin(2 * np.pi * current_i / 365 + phase)
    return sine_wave


def trend(rate, t):
    return rate * t


def stochastic_noise(mean, std_dev, t=None, current_i=None):
    return np.random.normal(mean, std_dev)


def autoregression(phi, noise_std, history, current_i, t=None):
    p = len(phi)
    if len(history) < p:
        return ValueError("NOT ENOUGH HISTORY")

    ar_sum = sum(phi[j] * history[current_i - j - 1] for j in range(p))
    noise = np.random.normal(0, noise_std)

    sickness_value = ar_sum + noise
    return sickness_value


def climate_dependent_disease_cases(
    temp_effect, rain_effect, rainfall, temperature, t=None, current_i=None
):
    # temp effect
    temp_mod = temp_effect * temperature
    # rainfall effect

    rain_mod = rain_effect * rainfall

    estimated_cases = temp_mod + rain_mod

    return estimated_cases


def realistic_data_generation(feature_name, t):
    df = ISIMIP_dengue_harmonized["brazil"].to_pandas()
    feature = df[feature_name].values[:t]
    feature = (feature / feature.max()) * 4
    return feature


FUNCTION_POOL = {
    "constant": constant,
    "random_walk": random_walk,
    "seasonal": seasonal,
    "trend": trend,
    "stochastic_noise": stochastic_noise,
    "normal_distribution": normal_distribution,
    "poisson_distribution": poisson_distribution,
    "exponential_growth": exponential_growth,
    "lognormal": lognormal,
    "complex_seasonal": complex_seasonal,
    "picewise_trend": picewivse_trend,
    "extreme_event": extreme_event,
    "autoregression": autoregression,
    "realistic_data_generation": realistic_data_generation,
    "climate_dependent_disease_cases": climate_dependent_disease_cases,
}
