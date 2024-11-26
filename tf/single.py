import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

data_path = "/home/pod/shared-nvme/datasets/Metro_Interstate_Traffic_Volume.csv"
data = pd.read_csv(
    data_path,
    parse_dates=True,
    index_col="date_time",
    usecols=["date_time", "traffic_volume"],
)

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

print(X.shape, y.shape)
print(X[0], y[0])
print(X[1], y[1])
print(X[2], y[2])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = models.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(layers.LSTM(50, return_sequences=False))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_absolute_error')

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
print(history)

loss = model.evaluate(X_test, y_test)

train_loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(dpi=300)
plt.title("Training and Validation Loss Over Epochs")
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, color="orange", label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig("single_train_val_loss.png")
plt.close()

# y_pred = model.predict(X_test)

# y_pred = scaler_target.inverse_transform(y_pred)
# y_test = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# plt.figure(figsize=(12, 6), dpi=300)
# plt.plot(y_test[-200:], label='True')
# plt.plot(y_pred[-200:], label='Predicted', color='orange')
# plt.title('True vs Predicted Traffic Volume in Test Set(200)')
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Traffic Volume', fontsize=14)
# plt.grid()
# plt.savefig('single_test_200.png')
# plt.close()