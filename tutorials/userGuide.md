# User-guide for installation and usage of mestDS

mestDS started as a small pilot project to familiarize with simulation of climate health data. August 2024, mestDS was uploaded to PiPy, and marks the start of this open source.

The package is used for simulation of climateHealth data.

## CONTENT:
- ### Installation.
- ### Usage.



## Installation.
Installation can be done by cloning the git repo and installed locally:
```
$ git clone https://github.com/martin-og-ingar/mestDS.git 
```

Once you have it locally, you can install it.
```
$ pip install -e
```



## Usage
When the package is installed, you can start your simulation.
```python
from mestDS import generate_data, graph, calculate_weekly_average
import datetime  
```

Set the start-date and generate data.
```python
start-date = datetime.date(2024, 1, 1)

# Data generation.
data = generate_data(True, 100, start-date, "W")
```