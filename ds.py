import yaml
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from pathlib import Path
from model import MultiLSTM
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datamodule import TrafficVolumeDatasetMulti
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

with open("config_m.yaml", "r") as file:
    config = yaml.safe_load(file)

torch.manual_seed(config["train"]["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

best_model_path = Path(config["best_model_path"])
data_path = Path(config["data"]["path"])

batch_size = config["train"]["batch_size"]
learning_rate = config["train"]["lr"]
num_epochs = config["train"]["epochs"]
patience = config["train"].get("early_stopping_patience", 10)


def process_data(data_path):
    target = ["traffic_volume"]
    cat_vars = [
        "holiday",
        "snow_1h",
        "weekday",
        "hour",
        "month",
        "year",
        "fog",
        "haze",
        "mist",
        "thunderstorm",
        "rain_1h",
    ]
    num_vars = ["temp", "clouds_all"]

    data_features = pd.read_csv(data_path)
    traffic_volume_data = data_features["traffic_volume"].values
    data_features = data_features.drop(columns=target)

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("oneHot", OneHotEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_vars),
            ("cat", categorical_transformer, cat_vars),
        ]
    )
    data_feature_transformed = preprocessor.fit_transform(data_features).toarray()
    return data_feature_transformed, traffic_volume_data


def prepare_datasets(data_path, config):
    
    data_features, traffic_volume = process_data(data_path)
    
    data_rate = int(len(data_features) * 0.8)
    train_dataset = data_features[:data_rate]
    test_dataset = data_features[data_rate:]
    
    train_label = traffic_volume[:data_rate]
    test_label = traffic_volume[data_rate:]

    train_dataset = TrafficVolumeDatasetMulti(
        train_dataset, train_label, time_step=config["data"]["time_step"]
    )

    test_dataset = TrafficVolumeDatasetMulti(
        test_dataset, test_label, time_step=config["data"]["time_step"]
    )

    return train_dataset, test_dataset


train, test = prepare_datasets(data_path, config)

print(train[0])
print(train[1])
print(train[2])