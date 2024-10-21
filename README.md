# Minimalist example of model integration with forecasts for multiple regions  
This document demonstrates a minimalist example of how to write a fully functional CHAP-compatible forecasting model. The example is based on the same simplistic regression as the "minimalist_example" model, but supports forecasts for multiple regions. It is thus fully functional, though still highly simplistic. 

## Running the model without CHAP integration
The example can be run in isolation (e.g. from the command line) using the file isolated_run.py:
```
python isolated_run.py  
```

For details on code files and data, please consult the "minimalist_example" model. The only differences are that:
* The training data file ("traindata.csv") here contains lines with multiple values (loc1 and loc2) in the location column:
```csv
time_period,rainfall,mean_temperature,disease_cases,location
2023-05,10,30,200,loc1
2023-06,2,30,100,loc1
2023-05,2,30,1000,loc2
2023-06,10,20,2000,loc2
```
* The train function (in "train.py") calls a utility function "get_df_per_location" to get an iterator over data frames (df) for each location in the input file: 
```csv
    locations = ```(csv_fn)
    for location,df in locations.items():
        #train model based on df, which only contains data for one location
```
* The predict function (in "predict.py") similarly loops through one model per location when making predictions.