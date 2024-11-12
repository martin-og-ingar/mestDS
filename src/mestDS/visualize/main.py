import matplotlib.pyplot as plt

import datetime


def graph(data, show_rain=False, show_temperature=False, show_sickness=True):
    from mestDS.classes.Simulation import DATEFORMAT, TIMEDELTA

    num_plots = sum([show_rain, show_temperature, show_sickness])
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    dates = [
        data.simulation_start_date + i * TIMEDELTA[data.time_granularity]
        for i in range(data.simulation_length)
    ]

    for region, observations in data.simulated_data.items():
        if show_rain:
            rainfall = [obs.rainfall for obs in observations]
            axes[0].plot(dates, rainfall, label=f"{region} - Rainfall")
            axes[0].set_ylabel("Rainfall (mm)")
            axes[0].legend()
            axes[0].set_title("Rainfall Over Time")

        if show_temperature:
            temperatures = [obs.mean_temperature for obs in observations]
            temp_axis = axes[1] if show_rain else axes[0]
            temp_axis.plot(dates, temperatures, label=f"{region} - Temperature")
            temp_axis.set_ylabel("Temperature (°C)")
            temp_axis.legend()
            temp_axis.set_title("Temperature Over Time")

        if show_sickness:
            cases = [obs.disease_cases for obs in observations]
            sick_axis = (
                axes[2]
                if show_rain and show_temperature
                else axes[1] if show_rain or show_temperature else axes[0]
            )
            sick_axis.plot(dates, cases, label=f"{region} - Disease Cases")
            sick_axis.set_ylabel("Disease Cases")
            sick_axis.legend()
            sick_axis.set_title("Sickness Over Time")

    for ax in axes:
        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(DATEFORMAT))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
        ax.grid(True)

    plt.tight_layout()
    plt.show()
