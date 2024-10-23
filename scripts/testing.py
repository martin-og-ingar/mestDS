from mestDS import (
    generate_data,
    graph,
    calculate_weekly_averages,
    generate_multiple_datasets,
    convert_datasets_to_gluonTS,
)
import datetime

start_date = datetime.date(2024, 1, 1)
sp = [0.4]
sa = [0.4]
si = [10, 11]
multiple_data = generate_multiple_datasets(
    ["Uganda", "Oslo"], True, True, start_date, 100, "D", sp, sa, si
)
ds_gluon_format = convert_datasets_to_gluonTS(multiple_data)
# print(multiple_data)
graph(multiple_data, True, True, False)
