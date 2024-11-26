import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


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

    scaler = MinMaxScaler(feature_range=(0, 1))
    traffic_volume_scaled = scaler.fit_transform(
        traffic_volume_data.reshape(-1, 1)
    ).flatten()

    return data_feature_transformed, traffic_volume_scaled, scaler


features, target, scaler = process_data("/home/pod/alex/ML/20241120/data/features.csv")


def create_dataset(data, target, time_step=24):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step)])
        y.append(target[i])
    return np.array(X), np.array(y)


time_step = 24
X, y = create_dataset(features.values, target.values, time_step)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = models.Sequential()
model.add(
    layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(X.shape[1], X.shape[2]),
    )
)
model.add(layers.LSTM(50, return_sequences=False))
model.add(layers.Dense(1))

model.compile(optimizer="adam", loss="mean_absolute_error")

history = model.fit(
    X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test)
)

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# plt.figure(dpi=300)
# plt.title("Training and Validation Loss Over Epochs")
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, color="orange", label='Validation Loss')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.legend()
# plt.savefig("multi_train_val_loss.png")
# plt.close()

loss = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6), dpi=300)
plt.plot(y_test[-200:], label="True")
plt.plot(y_pred[-200:], label="Predicted", color="orange")
plt.title("True vs Predicted Traffic Volume in Test Set(200)")
plt.xlabel("Time", fontsize=14)
plt.ylabel("Traffic Volume", fontsize=14)
plt.legend()
plt.grid()
plt.savefig("mutil_test_200.png")
plt.close()
