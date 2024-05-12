import pandas as pd
from libpysal.weights import Queen
import numpy as np
LABEL_NAME = "percent_vaccinated_log"

def load_train_and_valid_data(path):
    return pd.read_csv(path, dtype={"municipality_code_str": str})

def split_x_and_y(data: pd.DataFrame):
    y = data[LABEL_NAME].tolist()
    X = data.drop(LABEL_NAME, axis=1)
    return X, y

def load_test_data(path):
    return pd.read_csv(path, dtype={"municipality_code_str": str})

def _test_spatial_lag(df):
    for municipality_id in df['municipality_code_str'].unique():
        spatial_lag_value = df.loc[df['municipality_code_str'] == municipality_id, 'spatial_lag_percent_vaccinated'].iloc[0]
        neighbors = df.loc[df['municipality_code_str'] == municipality_id, 'neighbors'].iloc[0]
        if neighbors is not None:
            neighbor_values = df.iloc[neighbors]['percent_vaccinated_log']
            neighbor_avg = neighbor_values.mean()
            result = neighbor_avg == spatial_lag_value
            assert result, f"Test failed for municipality {municipality_id}"
        else:
            assert print(f"No neighbors found for municipality {municipality_id}")
    
def calculate_spatial_lag(main_dataframe, shape_data):
    shape_data_subset = shape_data[shape_data["mncplty_c"].isin(main_dataframe["municipality_code_str"])]
    shape_data_subset.reset_index(drop=True, inplace=True)
    wq = Queen.from_dataframe(shape_data_subset, use_index=False)
    wq.transform = 'b'
    percent_vaccinated_log_values = main_dataframe['percent_vaccinated_log'].values
    municipality_ids = main_dataframe['municipality_code_str'].values
    values_dict = dict(zip(municipality_ids, percent_vaccinated_log_values))

    spatial_lag = {}
    index_to_id = dict(zip(range(len(shape_data_subset)), shape_data_subset['mncplty_c']))
    for i, ids in enumerate(municipality_ids):
        index = shape_data_subset.index[shape_data_subset['mncplty_c'] == ids][0]
        neighbors_index = wq.neighbors.get(index)
        if neighbors_index:
            neighbors_ids = [index_to_id[idx] for idx in neighbors_index]
            lag = np.mean([values_dict[n] for n in neighbors_ids])
            spatial_lag[ids] = lag
        else:
            spatial_lag[ids] = np.nan  # Set to NaN if no neighbors are found
            
    main_dataframe["spatial_lag_percent_vaccinated"] = main_dataframe["municipality_code_str"].map(spatial_lag)
    main_dataframe["neighbors"] = wq.neighbors.values()
    _test_spatial_lag(main_dataframe)
    return main_dataframe



