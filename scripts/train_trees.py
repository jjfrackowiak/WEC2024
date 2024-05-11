import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from utils import split_x_and_y, load_train_and_valid_data, load_test_data, calculate_spatial_lag
from collections import OrderedDict
import argparse


def _test_model(model, shape_data):
    test_data = load_test_data("../_data/test_sample.csv")
    test_data = calculate_spatial_lag(test_data, shape_data)
    string_columns = test_data.select_dtypes(include=['object']).columns.tolist()
    string_columns.append("neighbors")
    test_data.drop(string_columns, axis=1, inplace=True)
    X_test, y_test = split_x_and_y(test_data)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse
    
def train_cross_validation(training_data: pd.DataFrame, shape_data: pd.DataFrame, model):
    unique_voivodeships = list(OrderedDict.fromkeys(training_data["voivodeship"]))
    eval_scores = []
    test_scores = []
    for voivodeship in unique_voivodeships:
        training_sample = training_data[training_data["voivodeship"] != voivodeship].copy()
        training_sample = calculate_spatial_lag(training_sample, shape_data)
        eval_sample = training_data[training_data["voivodeship"] == voivodeship].copy()
        eval_sample = calculate_spatial_lag(eval_sample, shape_data)
        
        assert voivodeship not in training_sample["voivodeship"], "Excluded voivodeship should not be present in the training sample"
        assert len(eval_sample["voivodeship"].unique()) == 1 and eval_sample["voivodeship"].unique()[0] == voivodeship, "Evaluation sample should contain only the excluded voivodeship"
        string_columns = training_sample.select_dtypes(include=['object']).columns.tolist()
        string_columns.append("neighbors")
        training_sample.drop(string_columns, axis=1, inplace=True)
        eval_sample.drop(string_columns, axis=1, inplace=True)
        
        training_sample.reset_index(drop=True, inplace=True)
        eval_sample.reset_index(drop=True, inplace=True)
        X_train, y_train = split_x_and_y(training_sample)
        X_eval, y_eval = split_x_and_y(eval_sample)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_eval)
        eval_rmse = np.sqrt(mean_squared_error(y_eval, predictions))
        test_rmse = _test_model(model, shape_data=shape_data)
        eval_scores.append(eval_rmse)
        test_scores.append(test_rmse)
        
        print(f"Voivodeship {voivodeship} | EVAL RMSE: {eval_rmse} | TEST RMSE: {test_rmse}")
        
    results = pd.DataFrame({"excluded_voivodeship": list(unique_voivodeships), "eval_rmse": eval_scores, "test_rmse": test_scores})
    return results
    
def main(args):
    if args.model == "rf":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = XGBRegressor(n_estimators=200, random_state=42)
    shape_data = gpd.read_file("../_data/shapefile/map_municipalities.shp")
    data = load_train_and_valid_data("../_data/train_and_valid.csv")
    results = train_cross_validation(data, shape_data, model)
    results.to_csv(f"results_{args.model}.csv", index=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to train and test", choices=["rf", "xgboost"])
    args = parser.parse_args()
    main(args)