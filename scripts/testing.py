from mestDS import (
    generate_data,
    graph,
    calculate_weekly_averages,
    generate_multiple_datasets,
    convert_datasets_to_gluonTS,
)
import datetime

start_date = datetime.date(2024, 1, 1)
sp = [0.5, 0.4, 0.6]
sa = [0.3, 0.2, 0.4]
si = [10, 9, 11]
multiple_data = generate_multiple_datasets(
    ["Uganda", "Oslo"], True, True, start_date, 200, "D", sp, sa, si
)
ds_gluon_format = convert_datasets_to_gluonTS(multiple_data)
graph(multiple_data, True, True, False)
