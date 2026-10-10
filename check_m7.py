import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("net_data.csv", parse_dates=["timestamp"])
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for n in range(1, 6):
    g = d[d.node_id == n]
    ax[0].plot(g.timestamp, g.battery_pct, label=f"Node {n}")
    ax[1].plot(g.timestamp, g.delivered.rolling(120).mean(), label=f"Node {n}")
ax[0].set_ylabel("Battery %"); ax[0].legend()
ax[1].set_ylabel("Delivery ratio (2h window)")
plt.tight_layout(); plt.show()