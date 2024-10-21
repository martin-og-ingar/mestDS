import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from isolated_run_utils import get_df_per_location


def read_pandas_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

#print(read_pandas_from_csv("trainData.csv"))

#SHOWN TO USER
def train(csv_fn, model_fn):
    features = ['rainfall', 'mean_temperature']
    models = {}
    locations = get_df_per_location(csv_fn)
    for location,df in locations.items():

        X = df[features]
        Y = df['disease_cases']
        model = LinearRegression()
        model.fit(X, Y)
        models[location] = model
        print(f"Train - model coefficients for location {location}: ", list(zip(features,model.coef_)))

    joblib.dump(models, model_fn)







