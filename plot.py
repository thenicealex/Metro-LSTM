import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from model import LSTM
from datamodule import TrafficVolumeDataset
from torch.utils.data import DataLoader

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

best_model_path = Path(config["best_model_path"])
data_path = Path(config["data"]["path"])

data = pd.read_csv(
    data_path,
    parse_dates=True,
    usecols=["date_time", "traffic_volume"],
)
data["date_time"] = pd.to_datetime(data.date_time)
data = data.set_index("date_time")
data = data[-270:]
traffic_volume = data["traffic_volume"].values

scaler = MinMaxScaler()
traffic_volume_scaled = scaler.fit_transform(traffic_volume.reshape(-1, 1)).flatten()

time_step = 24
test_dataset = TrafficVolumeDataset(traffic_volume_scaled, time_step=time_step)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

model_params = config["model"]
model = LSTM(model_params)
model.load_state_dict(torch.load(best_model_path, weights_only=False))
model.eval()
predictions = []
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        predictions.append(test_outputs)

y_pred = np.concatenate(predictions)
y_pred = scaler.inverse_transform(np.array(y_pred))
data = data[time_step + 1 :]

pred_data = pd.DataFrame(y_pred, columns=["traffic_volume"])
pred_data.index = data.index

plt.figure(figsize=(20, 10), dpi=300)
plt.plot(data, label="True")
plt.plot(pred_data, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.title("Traffic Volume Prediction")
plt.legend()
plt.grid()
plt.savefig("images/plot.png")
plt.close()
