import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from model import MultiLSTM
from datamodule import TrafficVolumeDatasetMulti


def load_config(config_path="config_m.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def transform_data(data):
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

    data_features = data
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

    scaler = MinMaxScaler(feature_range=(0, 1))
    traffic_volume_scaled = scaler.fit_transform(
        traffic_volume_data.reshape(-1, 1)
    ).flatten()

    return data_feature_transformed, traffic_volume_scaled, scaler


def load_data(data_path, time_step):

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
    data_features_copy = data_features.copy()[-263:]
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
    data_feature_transformed = preprocessor.fit_transform(data_features).toarray()[
        -263:
    ]

    scaler = MinMaxScaler(feature_range=(0, 1))
    traffic_volume_scaled = scaler.fit_transform(
        traffic_volume_data.reshape(-1, 1)
    ).flatten()[-263:]

    dataset = TrafficVolumeDatasetMulti(
        data_feature_transformed, traffic_volume_scaled, time_step
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_features_copy, loader, scaler


def load_model(model_params, best_model_path):

    model = MultiLSTM(model_params)
    model.load_state_dict(
        torch.load(
            best_model_path, map_location=torch.device("cpu"), weights_only=False
        )
    )
    model.eval()
    return model


def predict(model, loader):
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
    return np.concatenate(predictions)


def plot_results(actual, predicted, title, filename):
    plt.figure(figsize=(20, 10), dpi=300)
    plt.plot(actual, label="True")
    plt.plot(predicted, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Traffic Volume")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()


def forecast(model, history, scaler, time_step, forecast_steps):
    print(f"history shape: {history.shape}")
    data_feature_transformed, traffic_volume_scaled, scaler = transform_data(history)
    
    predictions = []
    predictions.append(data_feature_transformed[-time_step:])
    with torch.no_grad():
        for _ in range(forecast_steps):
            input_tensor = torch.FloatTensor(predictions[-time_step:]).view(1, time_step, 1)
            pred = model(input_tensor).item()
            predictions = np.append(predictions, pred)

    forecast_data = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    dates = pd.date_range(
        start=history.index[-1], periods=forecast_steps + 1, freq="h"
    )[1:]
    return pd.DataFrame(forecast_data, index=dates, columns=["traffic_volume"])


def main():
    config = load_config()
    config["model"]["input_size"] = 49
    time_step = config["data"]["time_step"]
    data_path = config["data"]["path"]

    data, loader, scaler = load_data(data_path, time_step)
    model = load_model(config["model"], config["best_model_path"])
    y_pred = predict(model, loader)
    y_pred = scaler.inverse_transform(y_pred)
    actual = data[time_step + 1 :]
    actual = actual.set_index("date_time")
    actual = actual[["traffic_volume"]]
    actual.reindex()
    print(f"Actual data: {actual}")

    pred_data = pd.DataFrame(y_pred, columns=["traffic_volume"])
    pred_data.index = actual.index
    plot_results(actual, pred_data, "Traffic Volume Prediction", "images/plot_m.png")

    history = data[-130:]
    print(f"History data: {history}")

    forecast_steps = 5 * time_step
    forecast_data = forecast(model, history, scaler, time_step, forecast_steps)
    plot_results(
        history, forecast_data, "Traffic Volume Forecast", "images/forecast_m.png"
    )
    print(f"Forecast data: {forecast_data}")


if __name__ == "__main__":
    main()
