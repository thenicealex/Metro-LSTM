import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from model import BiLSTM
from datamodule import TrafficVolumeDataset

torch.manual_seed(42)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

best_model_path = Path("best_model.pth")

data = pd.read_csv(
    "/home/pod/shared-nvme/datasets/Metro_Interstate_Traffic_Volume.csv",
    parse_dates=True,
    index_col="date_time",
    usecols=["date_time", "traffic_volume"],
)
data = data[-270:]
date_time = data.index
traffic_volume = data["traffic_volume"].values

scaler = MinMaxScaler()
traffic_volume_scaled = scaler.fit_transform(traffic_volume.reshape(-1, 1)).flatten()

time_step = 24
test_dataset = TrafficVolumeDataset(traffic_volume_scaled, time_step=time_step)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

model = BiLSTM(num_layers=1)
model.load_state_dict(
    torch.load(best_model_path, weights_only=False)["model_state_dict"]
)
model.eval()

predictions = []
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        predictions.append(test_outputs.cpu().numpy())

y_pred = np.concatenate(predictions)
y_pred = scaler.inverse_transform(np.array(y_pred))
y_true = traffic_volume[time_step:]

plt.figure(figsize=(20, 10), dpi=300)
plt.plot(range(time_step, len(traffic_volume)), y_true, label="True")
plt.plot(range(time_step, len(y_pred)+time_step), y_pred, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.title("Traffic Volume Prediction")
plt.legend()
plt.savefig("images/traffic_volume_prediction.png")

# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))