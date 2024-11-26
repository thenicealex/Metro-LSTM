import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/pod/shared-nvme/datasets/Metro_Interstate_Traffic_Volume.csv")

df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['weekday'] = df['date_time'].dt.weekday
df['month'] = df['date_time'].dt.month

df['holiday'] = df['holiday'].apply(lambda x: 1 if x != 'none' else 0)

categorical_features = ['weather_main', 'weather_description']
encoder = OneHotEncoder(sparse_output=False)  # 使用 sparse_output=False 返回密集矩阵

encoded_weather = encoder.fit_transform(df[categorical_features])

encoded_weather_df = pd.DataFrame(encoded_weather, columns=encoder.get_feature_names_out(categorical_features))

df = pd.concat([df, encoded_weather_df], axis=1)

df = df.drop(columns=categorical_features)

df['temp'] = (df['temp'] - 32) * 5/9
df['rain_1h'] = np.log1p(df['rain_1h'])  # log(1+x) 避免取log(0)
df['snow_1h'] = np.log1p(df['snow_1h'])
scaler = MinMaxScaler()
df[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(df[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])

scaler_target = MinMaxScaler(feature_range=(0, 1))
df['traffic_volume'] = scaler_target.fit_transform(df[['traffic_volume']])

features = df.columns.difference(['traffic_volume', 'date_time'])
target = df['traffic_volume']

def create_dataset(data, target, n_hours):
    X, y = [], []
    for i in range(n_hours, len(data)):
        X.append(data[i-n_hours:i])
        y.append(target[i])
    return np.array(X), np.array(y)

n_hours = 24
X, y = create_dataset(df[features].values, target.values, n_hours)

X = X.astype(np.float32)
y = y.astype(np.float32)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # [samples, timesteps, features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = models.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.LSTM(50, return_sequences=False))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_absolute_error')

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

train_loss = history.history['loss']
val_loss = history.history['val_loss']

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
print(f"Test Loss (MAE): {loss}")

print(X_test.shape)

save_data = X_test[-n_hours:].copy()
print(save_data.shape)


predictions = []
num_predictions = 5 * 24
print(df.columns)
print(df['date_time'].values)
print(df['date_time'].values[-1])

for _ in range(num_predictions):
    y_pred = model.predict(save_data)
    predictions.append(y_pred[0, 0])
    
predictions = np.array(predictions)
print(predictions.shape)

last_timestamp = df['date_time'].values[-1]
predicted_index = pd.date_range(start=last_timestamp, periods=num_predictions + 1, freq='h')[1:]
print(predicted_index.shape)


pred_df = pd.DataFrame(predictions, index=predicted_index, columns=['traffic_volume'])
print(pred_df)

save_date = df['date_time'].values[-n_hours:]
save_data = pd.DataFrame(y_test[-n_hours:], index=save_date, columns=['traffic_volume'])


plt.figure(figsize=(12, 6), dpi=300)
plt.plot(save_data, label='True')
plt.plot(pred_df, label='Predicted', color='orange')
plt.title('True vs Predicted Traffic Volume')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Traffic Volume', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('multi_pred.png')
plt.close()


# y_pred = model.predict(X_test)

# y_pred_rescaled = scaler_target.inverse_transform(y_pred)
# y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# plt.figure(figsize=(12, 6), dpi=300)
# plt.plot(y_test_rescaled[-200:], label='True')
# plt.plot(y_pred_rescaled[-200:], label='Predicted', color='orange')
# plt.title('True vs Predicted Traffic Volume in Test Set(200)')
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Traffic Volume', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('mutil_test_200.png')
# plt.close()