from mestDS import (
    graph,
    generate_multiple_datasets,
    convert_datasets_to_gluonTS,
    evaluate_chap_model,
    Simulation,
    RainSeason,
)
import datetime
from chap_core.predictor.model_registry import registry

# start_date = datetime.date(2024, 1, 1)
# sp = [0.5]
# sa = [0.3]
# si = [10]
# multiple_data = generate_multiple_datasets(
#     ["Uganda", "Oslo"], True, True, start_date, 10, "D", sp, sa, si
# )
# evaluate_chap_model(multiple_data)


simulation_one = Simulation()
simulation_one.simulate()
# print(simulation_one.simulated_data)

# evaluate_chap_model([simulation_one.simulated_data])
simulation_one.chap_evaluation_on_model(registry.get_model("chap_ewars_monthly"))
