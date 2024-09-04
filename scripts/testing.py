from mestDS import generate_data, graph, calculate_weekly_averages

data = generate_data(True, 100, "2024-01-01")
graph(data, sickness_enabled=True, temperature_enabled=True, precipitation_enabled=True)

average_data = calculate_weekly_averages(data)
graph(
    average_data,
    sickness_enabled=True,
    temperature_enabled=True,
    precipitation_enabled=True,
)

print(average_data)
