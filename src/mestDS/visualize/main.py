import matplotlib.pyplot as plt


def graph(data, precipitation_enabled, sickness_enabled, temperature_enabled):
    if sickness_enabled:
        plt.plot(data.sickness, label="Sickness", color="green")

    if temperature_enabled:
        plt.plot(data.temperature, label="Temperature", color="red")
    if precipitation_enabled:
        plt.plot(data.precipitation, label="Precipitation", color="blue")
    plt.xlabel("Week")
    plt.grid()
    plt.show()
