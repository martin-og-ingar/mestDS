from mestDS import Simulation
from chap_core.predictor.model_registry import registry

simulation_one = Simulation()
simulation_one.simulate()
simulation_one.chap_evaluation_on_model(registry.get_model("chap_ewars_monthly"))
