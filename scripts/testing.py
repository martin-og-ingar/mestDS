from mestDS import Simulation
from chap_core.predictor.model_registry import registry

sim = Simulation()
sim.simulate()
sim.chap_evaluation_on_model(registry.get_model("chap_ewars_monthly"))
