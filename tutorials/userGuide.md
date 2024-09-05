# User-guide for installation and usage of mestDS

mestDS started as a small pilot project to familiarize with simulation of climate health data. August 2024, mestDS was uploaded to PiPy, and marks the start of this open source.

The package is used for simulation of climateHealth data.

## CONTENT
- ### [Installation.](#installation)
- ### [Usage.](#usage)



## Installation.
Installation can be done by cloning the git repo and installed locally:
```
$ git clone https://github.com/martin-og-ingar/mestDS.git 
```

Once you have it locally, you can install it.
```
$ pip install -e
```
After successfully installation of the repo, install chap-core.
```
$ pip install git+https://github.com/dhis2/chap-core.git
```



## Usage
When the package is installed, you can start your simulation.
```python
from mestDS import generate_data
import datetime  
```

Set the start-date and generate data.
```python
start-date = datetime.date(2024, 1, 1)

# Data generation.
data = generate_data(True, 100, start-date, "W")
```

### Plotting.
It is also possible to visualize the data with a plot.
This can be done like this:
```python
# Here we assume you have created the ClimateHealth-dict
# If not, see example above.
from mestDS import graph

graph(data, sickness_enabled=True, temperature_enabled=True, precipitation_enabled=True)
```
```python
from mestDS import calculate_weekly_average
average_data = calculate_weekly_average(data)

graph(average_data,
  sickness_enabled=True,
  temperature_enabled=True,
  precipitation_enabled=True,)
```