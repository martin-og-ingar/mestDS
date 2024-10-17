from mestDS import (
    generate_data,
    graph,
    calculate_weekly_averages,
    generate_multiple_datasets,
    convert_datasets_to_gluonTS,
)
import datetime

start_date = datetime.date(2024, 1, 1)
# data = generate_data("Uganda", True, start_date, 100, "D")
# graph(
#     data,
#     "Uganda",
#     sickness_enabled=True,
#     temperature_enabled=True,
#     precipitation_enabled=True,
# )

multiple_data = generate_multiple_datasets(
    3, ["Uganda", "Oslo"], True, True, start_date, 10, "D"
)

gluon_datasets = convert_datasets_to_gluonTS(multiple_data)


graph(multiple_data[0], "Uganda", True, True, True)
# should assure that the data-generations is Week for this to be run.
# average_data = calculate_weekly_averages(data, "Uganda")
# graph(
#     average_data,
#     "Uganda",
#     sickness_enabled=True,
#     temperature_enabled=True,
#     precipitation_enabled=True,
# )
