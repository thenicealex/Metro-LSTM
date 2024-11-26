import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

data_path = "/home/pod/shared-nvme/datasets/Metro_Interstate_Traffic_Volume.csv"
data = pd.read_csv(
    data_path,
    usecols=["date_time", "traffic_volume"],
)
data['date_time'] = pd.to_datetime(data['date_time'])
data.set_index("date_time", inplace=True)

scaler_target = MinMaxScaler(feature_range=(0, 1))
data['traffic_volume'] = scaler_target.fit_transform(data[['traffic_volume']])

target = data['traffic_volume']

def create_dataset(data, time_step=24):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

n_hours = 24
X, y = create_dataset(target, n_hours)

X = X.astype(np.float32)
y = y.astype(np.float32)
print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = models.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.LSTM(50, return_sequences=False))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_absolute_error')

model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MAE): {loss}")

# Go to 24 data and predict the next one in turn
history_data = data[-n_hours:]
save_data = history_data.copy()
print(save_data)
hd = history_data['traffic_volume'].values.reshape(1, n_hours, 1)
print(type(hd))

predictions = []
num_predictions = 5 * 24

for _ in range(num_predictions):
    y_pred = model.predict(hd[-n_hours:])
    predictions.append(y_pred[0, 0])
    hd = np.append(hd[:, 1:, :], [[y_pred[0]]], axis=1)

predictions = np.array(predictions)
print(predictions.shape)
# y_pred_rescaled = scaler_target.inverse_transform(predictions.reshape(-1, 1))
# print(y_pred_rescaled.shape)

last_timestamp = save_data.index[-1]
predicted_index = pd.date_range(start=last_timestamp, periods=num_predictions + 1, freq='h')[1:]
print(predicted_index.shape)

pred_df = pd.DataFrame(predictions, index=predicted_index, columns=['traffic_volume'])
print(pred_df)

all_data = pd.concat([save_data, pred_df])

plt.figure(figsize=(12, 6), dpi=300)
plt.plot(save_data, label='True')
plt.plot(pred_df, label='Predicted', color='orange')
# plt.plot(history_data.index, original_data_rescaled, label='True')
# plt.plot(predicted_index, y_pred_rescaled, label='Predicted', color='orange')
plt.title('True vs Predicted Traffic Volume')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Traffic Volume', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('single_pred.png')
plt.close()