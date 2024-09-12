import matplotlib.pyplot as plt


def graph(data, region, precipitation_enabled, sickness_enabled, temperature_enabled):

    if sickness_enabled:
        plt.plot(
            [obs.disease_cases for obs in data[region]],
            label="Sickness",
            color="green",
        )

    if temperature_enabled:
        plt.plot(
            [obs.temperature for obs in data[region]],
            label="Temperature",
            color="red",
        )
    if precipitation_enabled:
        plt.plot(
            [obs.rainfall for obs in data[region]],
            label="Precipitation",
            color="blue",
        )
    plt.xlabel("Week")
    plt.grid()
    plt.show()
