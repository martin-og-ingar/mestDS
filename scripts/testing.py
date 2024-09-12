from mestDS import generate_data, graph, calculate_weekly_averages
import datetime

start_date = datetime.date(2024, 1, 1)
data = generate_data(True, start_date, 1000, "W")
graph(data, sickness_enabled=True, temperature_enabled=True, precipitation_enabled=True)

average_data = calculate_weekly_averages(data)
graph(
    average_data,
    sickness_enabled=True,
    temperature_enabled=True,
    precipitation_enabled=True,
)

print(data)
