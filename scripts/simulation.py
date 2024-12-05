country = [Region(), Region(), Region()]

sim = Simulation(
    simulation_length=1000,
    noise_std=5,
    country=country,
)
grid = Grid(
    attributes=["beta_lag_sickness", "beta_temp"], values=[0.0, 0.2, 0.5, 0.7, 1.0]
)
sim.simulate_over_grid(grid)
