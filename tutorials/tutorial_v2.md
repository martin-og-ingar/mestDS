# How to simulate data using the MultipleClimateHealth class

This tutorial takes you through the steps required for simulating data using DSL with the new class MultipleClimateHealth

## 1. Define simulation using dsl

In our new DSL, you must define the features of the dataset yourself.
Features requires a name and a list of parameters (region and i (the current timestep) does not require to be in the parameters list, but can be used as a parameter for the function). In addition, you ,must chose between "function" or "calculation". In "function", you must define a function that will return the value of the feature for each timestep. This has been done in the example below for the features rain and temperature. In "calculation", you must define the value to be returned for each timestep in a single code line, as done for the features lagged sickness and sickness.

```yaml
model:
  time_granularity: "D"
  simulation_length: 100
  sickness_calculation_method: "linear"
  features:
    - name: "rain"
      parameters: []
      function: |
        def get_rainfall(region, i):
          rain_season = False
          for season in region.rain_season:
            if season.start <= i <= season.end:
              rain_season = True
          if rain_season:
              return np.random.gamma(shape=6, scale=1.0) * 2
          else:
              return np.random.gamma(shape=2, scale=0.5) * 0.5
    - name: "temperature"
      parameters: ["rain"]
      function: |
        def get_temperature(i, rain):
          seasonal_temp = 24 + 5 * np.sin(2 * np.pi * i / 52)
          random_noise = np.random.normal(0, 2)
          if rain[-1] > 50:
            seasonal_temp -= 3
          return seasonal_temp + random_noise
    - name: "lagged_sickness"
      parameters: ["sickness"]
      calculation: "parameters['sickness'][-1]"
    - name: "sickness"
      parameters: ["lagged_sickness"]
      calculation: "parameters['lagged_sickness'][-1] * np.random.uniform(0.7,1.3)"
  regions:
    - name: "Finnmark"
      region_id: 1
      rain_season: [[10, 23], [45, 52]]
      neighbour: [2]
    - name: "Troms"
      region_id: 2
      rain_season: [[10, 20], [25, 45]]
      neighbour: [1, 3]
    - name: "Nordland"
      region_id: 3
      neighbour: [2, 4]
    - name: "Trøndelag"
      region_id: 4
      rain_season: [[5, 10], [15, 25], [45, 52]]
      neighbour: [3]
```

## 2. Initialize MultipleClimateHealth

Once the DSL has been defined, you must initialize the class MultipleClimateHealth with the path for the DSL file as an argument.

```python
from mestDS.classes.ClimateHealth import MultipleClimateHealth

ch_sim = MultipleClimateHealth("scripts/simulation5.yaml")
ch_sim.simulate()
```
