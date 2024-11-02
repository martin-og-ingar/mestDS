import pandas as pd

#Note: this function is here copied in from the repository chap_model_dev_toolkit to make this example repository easier to develop self-contained.
#For real development, we recommend installing chap_model_dev_toolkit and instead importing this function from there.
def get_df_per_location(csv_fn: str) -> dict:
    full_df = pd.read_csv(csv_fn)
    unique_locations_list = full_df['location'].unique()
    locations = {location: full_df[full_df['location'] == location] for location in unique_locations_list}
    return locations
