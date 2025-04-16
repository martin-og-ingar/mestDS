# Tutorial for new simulation

This tutorial walks you through the step for installation and setting up simulations using the new implementation which is based on version 3. The intention of this version is to generalize the composition of our DSL in order to provide a more flexible framework.

## 1. Installation.

Start by cloning the mestDS github repo

```bash
git clone https://github.com/martin-og-ingar/mestDS.git
```

Navigate to the root folder and install necessary packages using requirements.txt

```bash
pip install -r requirements.txt
```

## 2. Define the simulation using DSL.

When everything is installed you can configure your YAML.
The example below displays how you can define a simulation. The DSL assigned variables from the **Simulation** class. If a value is not specified the in the DSL, it will use default values.

### Model

The model-key defines the baseline model. The baseline is applied to all future simulations. A feature is defined under **features:**. Each tick defines a new feature. You are free to create any feature you want.

### Features

For each feature you define your desired functionality, which is defined under the **modification** key. You can use any of the functions defined in the **function pool** which will be explained later. The modification has some required variables.
**function**: This is responsible for what function you want to use. **params** are parameters you pass to your function in order to tailor it to your intention.

### Regions

A region is defined under _regions:_. Just as in features, each tick represents a region. To define a region, you must specify it's name, id, rain seasons, neighbours, and population. A rain season is defined as a list containing two numbers, the start and end of a rain season. A region can have multiple rain seasons.

### Example configuration

```yaml
model:
  simulation_name: "climate_dependent_non_autoregressive"
  time_granularity: "D"
  simulation_length: 100
  features:
    - name: "rainfall"
      modification:
        - function: "poisson_distribution"
          params:
            lam: 2
            scale: 1
        - function: "stochastic_noise"
          params:
            mean: 1
            std_dev: 2
    - name: "temperature"
      modification:
        - function: "seasonal"
          params:
            average: 25
            amplitude: 5
            pahse: 0
    - name: "sickness"
      modifications:
        function: "climate_dependent_disease_cases"
        params:
          temp_effect: 0.5
          rain_effect: 0.5
  regions:
    - name: "Region"
      region_id: 1
      rain_season: [[10, 23], [35, 40]]
      neighbour: [2]
      population: 100
```

### Function Pool

The function pool acts like a pool with functions that describe both environmental and climate health factors. The functions contains statistical and mathematical properties that describe central concepts in environmental data. You are free to choose any of the functions in the pool. This way you can mix and compose any statistical characteristic on a feature to achieve the desired behavior.

The only requirement is that the function defined in the DSL must match the function name and pass the correct parameters.

Functions in the pool with respective parameters:

- normal_distribution
  - mean - float
  - std_dev - float
- poisson_distribution
  - lam - float
  - scale - float
- exponential_growth
  - rate - float
- extreme_event
  - probability - float
  - magnitude - float
- seasonal
  - average - int
  - amplitude - int
  - phase - int
  - noise - float
- seasonal_disease_cases
  - average - int
  - amplitude - int
  - phase - int
  - noise - float
- spike
  - magnitude - int
  - spike_position - relativ positing of sim_length (0 - 1)
- trend
  - rate - folat
- stochastic_noise
  - mean - float
  - std_dev - float
- climate_dependent_disease_cases
  - lags - list[int, .., ]
  - auto_regressive - True/False
  - phi - list[float, ..., ]
- realistic_data_generation
  - feature_name - String
  - country - String
- correlation
  - correlation_feature - String (The feature that you want influence from)
  - correlation - float (Number between 0-1, tell how much you want from the feature)
  - lag - int - delayed effect.
- rain_season
  - peak_weeks - int
  - width - int (how wide the rain season is)
  - amplitude - int (max rainfall)
  - shape - String (gaussian/sinusodial)

### Realistic data

The current version supports data retrival from 16 different countries. This is gatherer from the chap-core.

- Argetina
- Brazil
- Cambodia
- Colombia
- Ecuador
- El Salvador
- Indonesia
- Laos
- Malaysia
- Mexico
- Nicaragua
- Panama
- Paraguay
- Peru
- Thailand
- Vietnam

The realistic datasets can be used like this

```yaml
model:
  features:
    - name: "temperature"
      modification:
        - function: "realistic_data_generation"
          params:
            feature_name: "mean_temperature"
            country: "<one_of_the_countries_listed_above>"
```

### Multiple simulations

You can perform multiple simulations within the same DSL configuratio.

By using the **simulations** key you define a new simulation. This uses the baseline model defined under **model** and alter the characteristics you modify with the same key-value pairs as the baseline.

```yaml
simulations:
  - simulation_name: "realistic_climate_autoregression"
    features:
      - name: "rainfall"
        modification:
          - function: "realistic_data_generation"
            params:
              feature_name: "rainfall"
              country: "Brazil"
      - name: "temperature"
        modification:
          - function: "realistic_data_generation"
            params:
              feature_name: "mean_temperature"
              country: "Brazil"
```

This means that you can edit the characteristics of a specific feature. The features you don't edit will behave as before.

### Edge cases.

The DSL supports two different configurations of a feature. The idea behind this, is that the current function pool might not be sufficient or you want a very specific behaviour.

The function approach provides high flexibility.

A feature can be defined like this:

```yaml
model:
  simulation_name: "function_approach"
  time_granularity: "D"
  simulation_length: 100
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
    - name: "temperature"
      function: |
        def get_temperature(i):
          i = (i - 1) % 52 + 1
          seasonal_temp = 24 + 5 * np.sin(2 * np.pi * i / 52)
          random_noise = np.random.normal(0, 2)
          return seasonal_temp + random_noise
```

You have the possibiliy to define any function you want. The only requirements is that it is written in Python.

## 3. Initialize SimulationDemo

When you have congifured the DSL you can initialize it and perform the simulation.
Pass the path of the DSL file when you initialize the **SimulationsDemo** class and run **simulate()** to simulate the data and display plot

Create a new .py file. Place it in the root of your project and define your simulation object like this.

```python
from src.mestDS.SimulationDemo import SimulationsDemo

ch_sim = SimulationsDemo("path/to/yaml")
ch_sim.simulate()

```

## 4. Retreive model for testing.

For demonstration purposes we will be using the **minimalistic_multiregion** model provided by the chap team to demonstrate how to pass the simulated data to a model.

create a folder called models/, go inside it and clone the model.

```bash
mkdir -p models
cd models
git clone https://github.com/dhis2-chap/minimalist_multiregion
```

## 5. Test on a chap model

The **Simulations** class contain a function for testin **CHAP** models called _eval_chap_model()_. Since a **CHAP** model expects the train and test datasets in a csv file, you must first convert the data to csv.

Pass a folder name/path to the _convert_to_csv()_ function. The datasets from the simulations and results from the model evaluation will be stored under here.

Pass the folder name/path to the **CHAP** model you wish to evaluate to the _eval_chap_model()_ function.

Use the same SimulationsDemo instance when you evaluate the report.

```python
ch_sim.convert_to_csvs("path/to/where/you/want/to/save/your/results")
# example
# ch_sim.convert_to_csvs("minimalist_multiregion_testing")
ch_sim.eval_chap_model("path/to/model")
```

The evaluation produces a PDF report with data from the model testing, a .csv with predictions and a images showing predictions vs actual cases.

## 3. Use cases/Examples.

In this tutorial I want to show-case two different scenarios.

### Example one.

Simulate 100 months. The simulation should have one Region. It should have a population of 100 and two rain-seasons free of choise. Remeber to provide region_id and neighbor. The disease cases should be dependent on the climate sensors rainfall and mean_temperature. Apply a lagged effect to both of them of 2 months.

The rainfall should use a poisson distribution to mimic real world behaviour. mean_temperature should use a seasonal pattern with some applied noise. Disease_cases should not use past cases in the calculation (no autoregression)

Step 1.

- Create a new .yaml file in your project. This is where the simulation is defined.
- Set up the baseline model with the model key and choose a simulation_name, time_granularity, and simulation_length

Step 2 - Define the features.

- Use the features key to define the climate sensors, how they should behave, and how you want to calculate disease_cases.

Step 3 - Create a region.

- Provide the simulation a region. Add region_id
- You can choose the name and the two rains-season intervalls yourself.

Step 4 - Initialise simulation.

- Create a new .py file located in the root of your project.
- Import the SimulationsDemo from src.mestDS.SimulationDemo
- Create a simulation object using the SimulationsDemo class.
- apply the method simulate() to perform the simulation. Remember to pass the path to your .yaml.

### Example two.

This simulation should create two different simulations. Simulate 200 months. The simulations should have one region like example one. Now you should have one rain_season between week 12 and week 20. Add a population of 200.

The simulation should use rainfall, humidity and mean_temperature.
rainfall should follow a poisson distribution. Humidity should follow a Normal (gaussian) distribution. mean_temperature should follow a cyclical pattern without noise. rainfall and humidity should have a lag of 3, while mean_temperature should have a lag of 2

Disease_cases should be dependent on all the climate variables. It will also be using autoregression.

The second simulation should use the same baseline but the behaviour of rainfall should be replaced by a function of your choice, using the edge case functionality. This does not need to be something fancy or complex. You are free to choose yourself.

Step 1.

- Create a new .yaml file in your project. This is where the simulation is defined.
- Set up the baseline model with the model key and choose a simulation_name, time_granularity, and simulation_length

Step 2 - Define the features.

- Use the features key to define the climate sensors, how they should behave, and how you want to calculate disease_cases.
- Remember to add

Step 3 - Create a region.

- Provide the simulation a region. Add region_id
- You can choose the name and the two rains-season intervalls yourself.

Step 4 - Multiple simulations

- With the use of the simulations key, replace the rainfall with the new functionality.
- Use the function key instead of modification key and create a python function free of choice.
