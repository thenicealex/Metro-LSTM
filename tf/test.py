
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 假设 df 是你的数据框
df = pd.read_csv('simulated_traffic_data.csv')  # 读取数据
# 数据框包含 'rain', 'snow', 'traffic_volume' 列

# 数据准备
data = df[['rain', 'snow', 'traffic_volume']].values

# 数据归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 创建时间窗口
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :-1])  # 取特征
        y.append(data[i + time_step, -1])       # 取目标值
    return np.array(X), np.array(y)

# 设置时间步长
time_step = 10  # 例如使用过去10个时间步的数据
X, y = create_dataset(data_scaled, time_step)

# 拆分数据集
X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]

# 调整输入形状为 [样本数, 时间步长, 特征数]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)  # 2为特征数
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

# 构建模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))  # 输出层，预测一个值

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32)

# 进行未来两个小时的预测
# 假设每个时间步代表30分钟，那么两个小时就是4个时间步
num_predictions = 4
predictions = []

# 使用最后一个时间窗口的数据进行预测
last_time_step_data = X_test[-1:]  # 取测试集的最后一个样本

for _ in range(num_predictions):
    next_value = model.predict(last_time_step_data)  # 预测下一个流量值
    predictions.append(next_value[0, 0])  # 取出预测值

    # 更新输入数据，确保维度匹配
    new_input = np.array([[next_value[0, 0], last_time_step_data[0, -1, 0], last_time_step_data[0, -1, 1]]])
    last_time_step_data = np.append(last_time_step_data[:, 1:, :], new_input.reshape(1, 1, -1), axis=1)

# 反归一化所有预测值
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

print("未来两个小时的交通流量预测值:", predictions)