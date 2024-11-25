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

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path="config_m.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    return config


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


def prepare_datasets(data_path, config):
    
    data_features, traffic_volume, scaler = process_data(data_path)
    
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

    logging.info(
        f"Train Dataset Length: {len(train_dataset)}, Test Dataset Length: {len(test_dataset)}"
    )
    return train_dataset, test_dataset, scaler



config = load_config()

best_model_path = Path(config["best_model_path"])
data_path = Path(config["data"]["path"])

batch_size = config["train"]["batch_size"]
learning_rate = config["train"]["lr"]
num_epochs = config["train"]["epochs"]
patience = config["train"].get("early_stopping_patience", 10)
epochs_no_improve = config["train"]["epochs_no_improve"]

train_dataset, test_dataset, scaler = prepare_datasets(data_path, config)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

single_data = train_dataset[0]
config['model']['input_size'] = single_data[0].shape[1]

model_params = config["model"]
model = MultiLSTM(model_params)
logging.info(f"Model Architecture: {model}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


logging.info(
    f"Total parameters of LSTM model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    return epoch_train_loss / len(train_loader)


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            epoch_val_loss += loss.item()
    return epoch_val_loss / len(val_loader)


train_losses = []
test_losses = []
best_val_loss = float("inf")

model.to(device)
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_loss = evaluate_model(model, test_loader, criterion, device)
    scheduler.step()

    if test_loss < best_val_loss:
        best_val_loss = test_loss
        torch.save(model.state_dict(), best_model_path)
        logging.info(
            f"Saved best model at epoch {epoch + 1} with test loss: {test_loss:.8f}"
        )
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 5 == 0:
        train_losses.append((epoch + 1, train_loss))
        test_losses.append((epoch + 1, test_loss))
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}"
        )

logging.info("Training Completed!")
epochs_plot = [x[0] for x in train_losses]
train_losses_values = [x[1] for x in train_losses]
test_losses_values = [x[1] for x in test_losses]

plt.figure(figsize=(10, 5), dpi=300)
plt.plot(
    epochs_plot, train_losses_values, label="Training Loss", color="blue", marker="o"
)
plt.plot(
    epochs_plot, test_losses_values, label="Validation Loss", color="orange", marker="x"
)
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs_plot)
plt.grid(True)
plt.legend()
plt.savefig("images/multi_train_val_loss.png")
plt.close()

logging.info(f"loss image saved at images/multi_train_val_loss.png")

predictions = []
ground_true = []
model.load_state_dict(torch.load(best_model_path, weights_only=False))
model.eval()
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_outputs = model(test_inputs)
        predictions.append(test_outputs.cpu().numpy())
        ground_true.append(test_targets.cpu().numpy())

y_pred = scaler.inverse_transform(np.concatenate(predictions))
y_test = scaler.inverse_transform(np.concatenate(ground_true).reshape(-1, 1))

logging.info(f"y_pred length: {len(y_pred)}, y_test length: {len(y_test)}")
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Prediction")
plt.title("Traffic Volume Prediction")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/multi_traffic_volume_test.png")
plt.close()

logging.info(f"test image saved at images/multi_traffic_volume_test.png")

logging.info("Done!")