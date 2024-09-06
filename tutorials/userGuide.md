# User-guide for installation and usage of mestDS

mestDS started as a small pilot project to familiarize with simulation of climate health data. August 2024, mestDS was uploaded to PiPy, and marks the start of this open source.

The package is used for simulation of climateHealth data.

## CONTENT
- ### [Installation.](#installation)
- ### [Usage.](#usage-1)



## Installation.
Installation can be done by cloning the git repo and installed locally:
```
$ git clone https://github.com/martin-og-ingar/mestDS.git 
```

Once you have a copy locally, you can install it:
```
$ pip install -e
```
After successfully installation of the repo, install chap-core:
```
$ pip install git+https://github.com/dhis2/chap-core.git
```



## Usage
### Data generation.
When the package is installed, you can generate your data.
```python
from mestDS import generate_data
import datetime  
```

Set the start-date and generate data.
```python
# Start date will be changed to the format "%date-%month-%Year.
start_date = datetime.date(2024, 1, 1)

data = generate_data(True, 100, start-date, "W")
"""
p1: Boolean - Enables/Disables seasonallity.
p2: Duration of the simulation.
p3: set start-date.
p4: Choose format of time-period. Choose between D(date), W(week), M(month)
"""

# Calculation of weekly average using the data.
average_data = calculate_weekly_average(data)
```

### Plotting.
It is also possible to visualize the data with a plot.
This can be done like this:
```python
# Here we assume you have created the ClimateHealth-dict
# If not, see example above.
from mestDS import graph

graph(data,
 sickness_enabled=True,
 temperature_enabled=True, 
 precipitation_enabled=True)
```
Visualization of weekly average.
```python
from mestDS import calculate_weekly_average
average_data = calculate_weekly_average(data)

graph(average_data,
  sickness_enabled=True,
  temperature_enabled=True,
  precipitation_enabled=True,)

"""
Choose what data to calculate average of.
"""
```