import csv
import datetime
import traceback
from dateutil.relativedelta import relativedelta
import inspect
import os
from typing import Any, Dict, Literal
import numpy as np
import matplotlib.pyplot as plt


from ..default_variables import DATEFORMAT, DEFAULT_REGIONS, TIMEDELTA
from .Feature import Variable
from .Region import Region


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Simulation:
    id: int
    name: str
    description: str
    time_delta: Literal["D", "W", "M"] = "W"
    length: int = 100
    start_date: datetime.date = datetime.date(2024, 1, 1)
    regions: list[Region]
    x: list[Variable]
    y: list[Variable]
    public_lists: dict[str, Any]
    data: Dict[str, Dict[str, list[float]]]
    current_i: int
    current_region: str
    full_set_path: str
    train_set_x_path: str
    train_set_y_path: str
    test_set_x_path: str
    test_set_y_path: str

    def __init__(self):
        self.x = []
        self.y = []
        self.public_lists = {}
        self.regions = []

    def simulate(self):
        self.initialize_data()
        delta = TIMEDELTA[self.time_delta]
        for i in range(1, self.length):

            if self.time_delta == "W":
                current_date = self.start_date + datetime.timedelta(weeks=i)
                iso_year, iso_week, _ = current_date.isocalendar()
                current_date_str = f"{iso_year}W{iso_week}"  # Correct week format
            elif self.time_delta == "M":
                current_date = self.start_date + relativedelta(months=i)
                current_date_str = current_date.strftime(DATEFORMAT[self.time_delta])
            else:
                current_date = self.start_date + (i * delta)
                current_date_str = current_date.strftime(DATEFORMAT[self.time_delta])

            self.current_i = i
            for region in self.regions:
                self.data[region.name]["time_period"].append(current_date_str)
                self.current_region = region
                for variable in self.x:
                    self.calculate_variable(variable)
                for variable in self.y:
                    self.calculate_variable(variable)

    def calculate_variable(self, variable: Variable):
        try:
            local_context = {}
            exec(
                variable.function,
                {
                    "np": np,
                    "i": self.current_i,
                    "region": self.current_region,
                },
                local_context,
            )

            # Extract the function from the exec environment
            func_name = list(local_context.keys())[0]
            func = local_context[func_name]

            # Extract required parameters from the function signature
            signature = inspect.signature(func)
            parameters_required = signature.parameters

            # Build arguments from dependencies
            args = [
                self.get_variable(param, variable.params)
                for param in parameters_required
            ]

            # Call the function
            result = func(*args)

            # Store the result
            self.data[self.current_region.name][variable.name].append(result)

        except Exception as e:
            error_message = (
                f"\nError in variable calculation:\n"
                f"  Variable: '{variable.name}'\n"
                f"  Time Index: {self.current_i}\n"
                f"  Region: {self.current_region.name if hasattr(self.current_region, 'name') else self.current_region}\n"
                f"  Function Source:\n{variable.function.strip()}\n"
                f"  Exception: {type(e).__name__}: {str(e)}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise RuntimeError(error_message) from e

    def get_variable(self, param_name, variable_params):
        # Special keywords
        if param_name == "region":
            return self.current_region
        if param_name == "i":
            return self.current_i

        # Check if param is explicitly passed in params
        if variable_params and param_name in variable_params:
            param_name = str(param_name)
            if param_name.startswith(
                "variable"
            ):  # the name "variable" is a reserved keyword. If param_name is "variable", it should look for the value in the dataset
                param_name = variable_params[param_name]
            else:
                return variable_params[param_name]

        if self.public_lists is not None and param_name in self.public_lists:
            return self.public_lists[param_name]
        # Else, treat it as a reference to a public variable
        if param_name in self.data[self.current_region.name]:
            return self.data[self.current_region.name][param_name]

        raise ValueError(f"Parameter '{param_name}' could not be resolved.")

    def initialize_data(self):
        if len(self.regions) == 0:
            self.regions = DEFAULT_REGIONS
        self.data = {region.name: {} for region in self.regions}
        for region in self.regions:
            self.data[region.name] = {}
            if self.time_delta == "W":
                current_date = self.start_date + datetime.timedelta(weeks=0)
                iso_year, iso_week, _ = current_date.isocalendar()
                current_date_str = f"{iso_year}W{iso_week}"
                self.data[region.name]["time_period"] = [current_date_str]
            else:
                self.data[region.name]["time_period"] = [
                    datetime.datetime.strftime(
                        self.start_date, DATEFORMAT[self.time_delta]
                    )
                ]

            for variable in self.x:
                self.data[region.name][variable.name] = [0]
            for variable in self.y:
                self.data[region.name][variable.name] = [0]

    def plot_data(self, dont_show, filename=None):
        regions = self.data.keys()
        variables = [
            variable.name for variable in self.x if variable.name not in dont_show
        ]
        variables += [
            variable.name for variable in self.y if variable.name not in dont_show
        ]
        for region in regions:
            plt.figure(figsize=(10, 6))

            for var in variables:
                plt.plot(self.data[region][var], label=f"{region} - {var}")

            plt.title(self.name)
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.legend()
            plt.tight_layout()
            if filename:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
            else:
                plt.show()
            plt.close()

    def to_csv(self, file_path):

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        csv_rows = []
        columns = ["time_period"]
        columns += [variable.name for variable in self.x]
        columns += [variable.name for variable in self.y]
        columns.append("location")
        csv_rows.append(columns)

        for region in self.regions:
            for i in range(len(self.data[region.name][self.x[0].name])):
                row = []
                row.append(self.data[region.name]["time_period"][i])
                for variable in self.x:
                    row.append(self.data[region.name][variable.name][i])
                for variable in self.y:
                    row.append(self.data[region.name][variable.name][i])
                row.append(region.name)
                csv_rows.append(row)

        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_rows)
