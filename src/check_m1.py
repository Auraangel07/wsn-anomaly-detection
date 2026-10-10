import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("normal_data.csv", parse_dates=["timestamp"])
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for n in range(1, 6):
    d = df[(df.node_id == n) & (df.timestamp < "2026-10-04")]
    ax[0].plot(d.timestamp, d.temperature, label=f"Node {n}")
    ax[1].plot(d.timestamp, d.humidity)
    ax[2].plot(d.timestamp, d.light)
ax[0].set_ylabel("Temp °C"); ax[1].set_ylabel("Humidity %"); ax[2].set_ylabel("Light")
ax[0].legend(); plt.tight_layout(); plt.show()
