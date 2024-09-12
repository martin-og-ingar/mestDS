from mestDS import generate_data, graph, calculate_weekly_averages
import datetime

start_date = datetime.date(2024, 1, 1)
data = generate_data("Uganda", True, start_date, 1000, "W")
graph(
    data,
    "Uganda",
    sickness_enabled=True,
    temperature_enabled=True,
    precipitation_enabled=True,
)

average_data = calculate_weekly_averages(data, "Uganda")
graph(
    average_data,
    "Uganda",
    sickness_enabled=True,
    temperature_enabled=True,
    precipitation_enabled=True,
)
