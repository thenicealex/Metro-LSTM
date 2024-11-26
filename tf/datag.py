import pandas as pd
import numpy as np

# 设置随机种子以便重现
np.random.seed(42)

# 生成时间序列
date_rng = pd.date_range(start='2023-01-01', end='2023-01-10', freq='30T')  # 每30分钟一个数据点

# 生成下雨和下雪的随机数据
rain = np.random.choice([0, 1], size=len(date_rng), p=[0.8, 0.2])  # 80%概率没有下雨
snow = np.random.choice([0, 1], size=len(date_rng), p=[0.9, 0.1])   # 90%概率没有下雪

# 生成交通流量数据（假设流量在500到3000之间波动）
traffic_volume = 500 + np.random.randint(0, 2500, size=len(date_rng))

# 将数据组合成DataFrame
data = pd.DataFrame(data={
    'timestamp': date_rng,
    'rain': rain,
    'snow': snow,
    'traffic_volume': traffic_volume
})

# 设置时间戳为索引
data.set_index('timestamp', inplace=True)

# 打印前几行数据
print(data.head())

# 保存数据到CSV文件（可选）
data.to_csv('simulated_traffic_data.csv')