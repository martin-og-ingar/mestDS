# mestDS - Data simulation tool for model evaluation and stress testing

mestDS started as a small pilot project to familiarize with simulation of climate health data. August 2024, mestDS was uploaded to PiPy, and marks the start of this open source.

## 1. Installation.

Start by cloning the mestDS github repo

```bash
git clone https://github.com/martin-og-ingar/mestDS.git
```

Navigate to the root folder and install necessary packages using requirements.txt

```bash
pip install -r requirements.txt
```

## 2. DSL configuration

The framework takes advatage of a custom domain-specific language configured using YAML. A more detailed overview of each key-value pair is described in [tutorial_v4.md](/tutorials/tutorial_v4.md)

Some example yaml-scirpt can be found [Here](/example_run/)

The yaml has some required key-value pairs.

- Required

  - model

    - simulation_name
    - time_granularity
    - simulation_length
    - features
      - name
      - modification / function
        - function-pool name || define your Python function.
          - parameters
    - regions
      - name

  - simulations
    - simulation_name
    - features
      - name
      - modification / function

The model flag acts as a baseline for all simulations you define.

Under the features flag you can choose either to modify the feature using the function pool. The tutorial provides an overview of what functions it contains and the required parameters.

The function approach lets you define your own Python function to solve edge cases.

## 3. Initialize SimulationDemo

When you have congifured the DSL you can initialize it and run the simulation.
Pass the path of the DSL file when you initialize the **SimulationsDemo** class and run **simulate()** to simulate the data and display plot

Create a new .py file. Place it in the root of your project and define your simulation object like this.

```python
from src.mestDS.SimulationDemo import SimulationsDemo

ch_sim = SimulationsDemo("path/to/yaml")
ch_sim.simulate()

```

### Example run

In the [run.py](/example_run/run.py) you can find an example script of how to simulate your data based on the DSL-configuration and perform model testing.

## 4. External model.

For demonstration purposes we will be using the **minimalistic_multiregion** model provided by the chap team to demonstrate how to pass the simulated data to a model.

create a folder called models/, go inside it and clone the model.

```bash
mkdir -p models
cd models
git clone https://github.com/dhis2-chap/minimalist_multiregion
```

## 5. Test on External model

```python
ch_sim.convert_to_csvs("path/to/where/you/want/to/save/your/results")
# example
# ch_sim.convert_to_csvs("minimalist_multiregion_testing")

ch_sim.eval_chap_model("path/to/model")
```
