# Tutorial for new simulation

This tutorial walks you through the step for installation and setting up simulations using the new implementation which is based on version 3. The intention of this version is to generalize the composition of our DSL in order to provide a more flexible framework.

## 1. Installation.

Start by cloning the mestDS github repo

```bash
git clone https://github.com/martin-og-ingar/mestDS.git
```

Navigate to the root folder and install necessary packages used requirements.txt

```bash
pip install -r requirements.txt
```

## 2. Define the simulation using DSL.

The example below displays how you can define a simulation. The DSL assigned variables from the **Simulation** class. If a value is not specified the in the DSL, it will use default values.

The model-key defines the baseline model. The baseline is applied to all future simulations. A feature is defined under **features:**. Each tick defines a new feature. You are free to choose any feature you want.

For each feature you are free to pick a characteristic, which is defined under the **modification** key. You can use any of the functions defined in the **function pool** which will be explained later. The modification has some required variables.
**function**: this is responsible for what function you want to use. **params** are parameters you pass to your function in order to tailor it to your intention.

A region is defined under _regions:_. Just as in features, each tick represents a region. To define a region, you must specify it's name, id, rain seasons and neighbours. A rain season is defined as a list containing two numbers, the start and end of a rain season. A region can have multiple rain seasons.

```yaml
model:
  simulation_name: "climate_dependent_non_autoregressive"
  time_granularity: "D"
  simulation_length: 100
  baseline_func: "constant"
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
```

## Realistic data

The current version provides support for retrieving real world data. This is gatherer from the chap-core and you can choose between 16 different countries

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

## Function pool

The function pool acts like a pool with functions that describe both environmental and climate health factors. You are free to choose any of the functions in the pool. This way you can mix and compose any statistical characteristic on a feature to achieve the desired behavior.

The only requirement is that the function defined in the DSL must match the function name and you have to pass the correct parameters.

```Python
def poisson_distribution(lam, scale, t=None, current_i=None):
    events = np.random.poisson(lam)
    rainfall = events * np.random.exponential(scale)
    return rainfall

def seasonal(average, amplitude, phase, current_i, t=None):
    sine_wave = average + amplitude * np.sin(2 * np.pi * current_i / 365 + phase)
    return sine_wave

def stochastic_noise(mean, std_dev, t=None, current_i=None):
    return np.random.normal(mean, std_dev)


def climate_dependent_disease_cases(
    temp_effect, rain_effect, rainfall, temperature, t=None, current_i=None
):
    temp_mod = temp_effect * temperature

    rain_mod = rain_effect * rainfall

    estimated_cases = temp_mod + rain_mod

    return estimated_cases
```

## Multiple simulations

You can perform multiple simulations within the same DSL.

By using the **simulations** key you define a new simulation. This uses the baseline model defined under **model** and alter the characteristics you specify.

```yaml
simulations:
  - simulation_name: "realistic_climate_autoregression"
  features:
    - name: "rainfall"
      modification:
        - function: "realistic_data_generation"
          params:
            feature_name: "rainfall"
    - name: "temperature"
      modification:
        - function: "realistic_data_generation"
          params:
            feature_name: "mean_temperature"
    - name: "sickness"
      modification:
        - function: "autoregression"
          params:
            phi: [0.5, 0.3]
            noise_std: 1

```

## 2. Initialize SimulationDemo

Pass the path of the DSL file when you initialize the **Simulations** class and run **simulate()** to simulate the data and display plot

```python
from mestDS.classes.Simulation import Simulations

ch_sim = Simulations("path/to/yaml")
ch_sim.simulate()

```

## 4. Retreive model for testing.

For demonstration purposes we provide an illustration using the **minimalistic_multiregion** model provided by the chap team.

In your root folder

```bash
git clone "github"
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

![alt text](Test.png)
