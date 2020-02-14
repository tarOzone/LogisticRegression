import os
import json
import pickle
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import StandardScaler


def get_datetime():
    """
    return the current datetime in format - "%d_%m_%Y-%H_%M_%S"
    :param
    :return
        now: datetime string
    """
    now = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    return now


def preprocess_data(data):
    """
    pre-processing the data using standard scaler
    :param
        data: dataframe from pandas
    :return
        _data: preprocessed data in dataframe
    """
    scaler = StandardScaler()
    scaler.fit(data)
    _data = scaler.transform(data)
    return _data


def load_dataset(filename, to_be_dropped, target):
    """
    get data from csv file
    :param
        filename: eg. ./data/diabetes2_csv.csv
        to_be_dropped: list of column to be dropped
        target: string of column to be a target
    :return
        X, y
    """
    ext = os.path.splitext(filename)[1]
    if ext == ".csv":
        df = pd.read_csv(filename)
    if ext == ".xlsx":
        df = pd.read_excel(filename)
    df.drop(to_be_dropped, axis=1, inplace=True)
    targets = df[[target]]
    pd.DataFrame.pop(df, target)
    features = df
    return features, targets


def train_test_fixed_split(x_prep, y, _from, _to):
    '''
    split data with fixed index
    :param
        X_prep: preprocessed feature data
        y: labels data
        _from: index from
        _to: index to
    :return
        X_train, X_test, y_train, y_test
    '''
    X_train = x_prep[_from:_to + 1, :]
    X_test = np.delete(x_prep, tuple(i for i in range(_from, _to + 1)), axis=0)
    y_train = y.loc[_from:_to]
    y_test = y.drop([i for i in range(_from, _to + 1)], axis=0)
    return X_train, X_test, y_train, y_test


def save_model(model):
    """
    save the scikit-learn model
    :param
        model: from sklearn
    :return
        save_dir: save directory
    """
    save_dir = os.path.join("saved_model", get_datetime() + ".pickle")
    with open(save_dir, "wb") as pkl:
        pickle.dump(model, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    return save_dir


def load_model(model_dir):
    """
    load the saved model in .pickle file
    :param
        model_dir: eg. ./saved_model/10_02_2020-09_15_48.pickle
    :return:
        loaded_model: sklearn model
    """
    with open(model_dir, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


def load_config(config_file):
    """
    load config file (.json)
    :param
    :return
        dictionary of config
    """
    try:
        with open(config_file, "r") as f:
            config = json.loads(f.read())
        return config
    except FileNotFoundError:
        return {}