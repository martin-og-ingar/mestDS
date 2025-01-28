from mestDS.classes import Simulation


sim = Simulation(simulation_length=1000, noise_std=5, country="Norway")
beta_values = ["/bls:0.5\br:0.3", "/br:0.4\bt:04\bn:0.7", ""]
for beta in beta_values:
    sim.regex(beta)
    sim.simulate
