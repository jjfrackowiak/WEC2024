import pandas as pd



LABEL_NAME = "percent_vaccinated_log"

def remove_voivevodship_from_sample(data: pd.DataFrame, voivevodship_name: str) -> pd.DataFrame:
    return data[data["voivevodship"] != voivevodship_name]

def load_train_and_valid_data(path):
    return pd.read_csv(path)

def split_x_and_y(data: pd.DataFrame):
    y = data[LABEL_NAME].tolist()
    X = data.drop(LABEL_NAME, axis=1)
    return X, y

def load_test_data(path):
    return pd.read_csv(path)
    



