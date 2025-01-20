# Tutorial on how to use DSL and Simulations

Our framework **mestDS** allows you to utilize our **Simulations** class and **DSL** to simulate data. The flexible and generic domain specific language lets the user specify which features should be included in the dataset and functions for how they should be calculated. This means that you can easily tailor the dataset to fit your needs in the DSL.

## 1. Define the simulation using our DSL

The example below shows how a simulation can be defined. Essentially, what the dsl does, is defining the variables of the **Simulation** class. If a variable is not specified in the DSL, it will use the default value (which you can find by inspecting the **Simulation** class).

A feature is defined under _features:_. Each tick represents a feature, and a name and function must be specified. The functions will be used to calculate the feature it belongs to for each time step and region in the simulation. The parameter list in the function signature can include the other features, or region (which is the current region you are simulating) and i (which is the current time step).

A region is defined under _regions:_. Just as in features, each tick represents a region. To define a region, you must specify it's name, id, rain seasons and neighbours. A rain season is defined as a list containing two numbers, the start and end of a rain season. A region can have multiple rain seasons.

```yaml
model:
  time_granularity: "D"
  simulation_length: 52
  features:
    - name: "rainfall"
      function: |
        def get_rainfall(region, i):
          i = (i - 1) % 52 + 1
          rain_season = False
          for season in region.rain_season:
            if season.start <= i <= season.end:
              rain_season = True
          if rain_season:
              return np.random.gamma(shape=6, scale=1.0) * 4
          else:
              return np.random.gamma(shape=2, scale=0.5) * 2
    - name: "mean_temperature"
      function: |
        def get_temperature(i):
          i = (i - 1) % 52 + 1
          seasonal_temp = 24 + 5 * np.sin(2 * np.pi * i / 52)
          random_noise = np.random.normal(0, 2)
          return seasonal_temp + random_noise
    - name: "lagged_sickness"
      function: |
        def get_lagged_sickness(disease_cases):
          return disease_cases[-1]
    - name: "disease_cases"
      function: |
        def get_disease_cases(mean_temperature, rainfall, lagged_sickness):
          sickness = (mean_temperature[-1] * 0.2) + (rainfall[-1] *0.12) + (lagged_sickness[-1] * 0.7)
          return sickness

  regions:
    - name: "Finnmark"
      region_id: 1
      rain_season: [[10, 23], [45, 52]]
      neighbour: [2]
    - name: "Troms"
      region_id: 2
      rain_season: [[10, 16], [37, 45]]
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

Pass the path of the DSL/yaml file when you are initializing the **Simulations** class and run _simulate()_ to simulate the data.

```python
from mestDS.classes.Simulation import Simulations

ch_sim = Simulations("your yaml file here")
ch_sim.simulate()
```

## 3. Test a chap model

The **Simulations** class contain a function for testin **CHAP** models called _eval_chap_model()_. Since a **CHAP** model expects the train and test datasets in a csv file, you must first convert the data to csv.

Pass a folder name/path to the _convert_to_csv()_ function. The datasets from the simulations and results from the model evaluation will be stored under here.

Pass the folder name/path to the **CHAP** model you wish to evaluate to the _eval_chap_model()_ function.

```python
ch_sim.convert_to_csvs("testing_minimalist_multiregion/")
ch_sim.eval_chap_model("models/minimalist_multiregion")
```

The image shows the disease cases vs. the predicted cases defined, simulated and predicted from the dsl/code snippets above. The image was generated during the model evaluation.

![alt text](image.png)
