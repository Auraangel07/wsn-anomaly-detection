import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("faulty_data.csv", parse_dates=["timestamp"])
d = df[df.node_id == 1]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(d.timestamp, d.temperature.where(d.temperature.between(0, 50)), lw=0.6, label="temperature")
f = d[(d.is_fault == 1) & d.temperature.between(0, 50)]
ax.scatter(f.timestamp, f.temperature, c="red", s=6, label="fault")
ax.legend(); ax.set_title("Node 1: faults in red"); plt.tight_layout(); plt.show()