import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "/home/pod/shared-nvme/datasets/Metro_Interstate_Traffic_Volume.csv",
    parse_dates=True,
    usecols=["date_time", "traffic_volume"],
)
data["date_time"] = pd.to_datetime(data.date_time)
data = data.set_index("date_time")
data = data[-270:]

plt.figure(figsize=(20, 10), dpi=300)
plt.plot(data, label="True")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.title("Traffic Volume Prediction")
plt.legend()
plt.grid()
plt.savefig("images/plot.png")
plt.close()
