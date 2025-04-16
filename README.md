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

## 2. Initialize SimulationDemo

When you have congifured the DSL you can initialize it and run the simulation.
Pass the path of the DSL file when you initialize the **SimulationsDemo** class and run **simulate()** to simulate the data and display plot

Create a new .py file. Place it in the root of your project and define your simulation object like this.

```python
from src.mestDS.SimulationDemo import SimulationsDemo

ch_sim = SimulationsDemo("path/to/yaml")
ch_sim.simulate()

```

## 3. External model.

For demonstration purposes we will be using the **minimalistic_multiregion** model provided by the chap team to demonstrate how to pass the simulated data to a model.

create a folder called models/, go inside it and clone the model.

```bash
mkdir -p models
cd models
git clone https://github.com/dhis2-chap/minimalist_multiregion
```

## 4. Test on External model

```python
ch_sim.convert_to_csvs("path/to/where/you/want/to/save/your/results")
# example
# ch_sim.convert_to_csvs("minimalist_multiregion_testing")

ch_sim.eval_chap_model("path/to/model")
```
