import joblib
import pandas as pd

from utils import get_df_per_location

def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    models = joblib.load(model_fn)
    locations = get_df_per_location(future_climatedata_fn)
    for location,df in locations.items():
        X = df[['rainfall', 'mean_temperature']]
        model = models[location]
        y_pred = model.predict(X)
        df['sample_0'] = y_pred
        df.to_csv(predictions_fn, index=False, mode='a', header=False)

        print("predict - forecast values: ", y_pred)