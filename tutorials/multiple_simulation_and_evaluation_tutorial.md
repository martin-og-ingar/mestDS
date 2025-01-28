# How to use MultipleSimulation for model evaluation

This tutorial takes you through how to use the `class ModelSimulation` for simulating multiple datasets with different charcteristics, to thoroughly evaluate a CHAP model or test different hyphotheses.

## Introduction of MultipleSimulation and other classes

`MultipleSimulation` is simple a class containing a list of `Simulation` instances, with some other variables and helper functions. `Simulation` looks like this:

```python
class Simulation:
    time_granularity: Literal["D", "W", "M"]
    simulation_length: int
    simulation_start_date: datetime.date
    regions: list[Region]
    simulated_data: Dict[str, list[Obs]]
    beta_rainfall: float
    beta_temp: float
    beta_lag_sickness: float
    beta_neighbour_influence: float
    neighbors: np.ndarray
    noise_std: float
```

- **time_granularity** represents the granularity of the dataset. In other words, what each row in the dataset represent, either a day ("W"), week ("W") or month ("W").
- **simulation_length** decides the lenght of the simulation.
- **simulation_start_date** marks the start of the simulation.
- **regions** is a list of `Region` instances.
- **simulated_data** is a dictionalty where a list of `Obs` is attatched to a `Region`.
- **beta_rainfall** decides how much rain should be weighted when calculating sickness
- **beta_temp** decides how much temperature should be weighted when calculating sickness
- **beta_lag_sickness** decides how much previous sickness should be weighted when calculating sickness
- **beta_neighbour_influence** decides how much neighbours' sickness cases should be weighted when calculating sickness
- **noise_std** is the noise added when calculating sickness.

`Region` looks as follows

```python
class Region:
    name: str
    region_id: int
    rain_season: list[RainSeason]
    neighbour: list[int]
```

- **name** is the name of the region.
- **region_id** is an integer that identifies the region
- **rain_season** is a list of rain seasons that occur in the region.
- **neighbour** is a list of neighbour regions' id.

```python
class RainSeason:
    start: int
    end: int
```

- **start** is the start of the rain season.
- **end** is the end of the rain season.
- Depending on `time_granularity` set in `Simulation`, the start and end value represent a day-, week- or month number.

## Import

The required module can be imported as follows

```python
from mestDS import MultipleSimulations
```

## Initialization

The class can be initialized in different ways:

### 1. Initialize with yaml file

Using a yaml file to define `MultipleSimulation` allows you to easily create multiple simulations with different characteristics.The yaml file requires you to follow a somewhat strict schema, but don't worry, it's quite intuitive.

The simulations must be defined inside `simulation` as done in the example below. Furthermore, you can define common arguments inside `base`, which will be used on all simulations. Inside `sims`, a simulation is defined with a tick (`-`). For each tick, you can define arguments specific for that simulation.

You can also define regions in the yaml file. These will be used for all simulations that you define in the file. The regions must be defined inside `regions`, and each tick represents one region. For each tick, you must define the `name`, `region_id`, `rain_season` and `neighbour` to that region. `rain_season` is defined as a 2-dimensional list that defines multiple rain seasons, where each sublist contains two values: the start and end points of a specific rain season.

The values not defined in the yaml file will use the default values defined in the `__init__` functions' parameter list.

```yaml
simulation:
  base:
    time_granularity: "D"
    simulation_length: 500
  sims:
    - beta_rainfall: 1.5
      beta_lag_sickness: 0.4
    - beta_rainfall: 0.3
      beta_lag_sickness: 1.9
    - beta_rainfall: 0.5
      beta_lag_sickness: 0.9
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

Once you are happy with your yaml file, you can initialize `MultipleSimulation` like this

```python
multiple_simulations = MultipleSimulations(yaml_path=[path to the yaml file])
```

## Simulate

Before evaluating the model, you must simulate the data. This must be done by calling `simulate()`.

```python
multiple_simulations.simulate()
```

## Evaluate

Lastly, you can evaluate your CHAP model by calling `eval_chap_model()`. The function will convert the simulated data to a csv file, split it to train and test csv files and generate a plot for each region's predicted and true number of disease cases.

```python
multiple_simulations.eval_chap_model([Path to folder you wish to the csv files and plotd], [Path to the CHAP model])
```
