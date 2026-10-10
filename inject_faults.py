import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

df = pd.read_csv("normal_data.csv", parse_dates=["timestamp"])
df["fault_type"] = df["fault_type"].astype(object)

# Make sure columns can hold NaN
for c in ["temperature", "humidity", "light"]:
    df[c] = df[c].astype(float)

FAULTS = ["stuck", "spike", "drift", "dropout", "corrupt"]
N_EVENTS_PER_NODE = 12   # fault episodes per node


def inject(df_node):
    """Inject fault episodes into one node's rows (index-based)."""
    idx = df_node.index.to_numpy()
    n = len(idx)
    used = np.zeros(n, dtype=bool)   # avoid overlapping episodes

    for _ in range(N_EVENTS_PER_NODE):
        ftype = FAULTS[rng.integers(len(FAULTS))]
        length = {
            "stuck": rng.integers(30, 90),
            "spike": rng.integers(1, 4),
            "drift": rng.integers(120, 240),
            "dropout": rng.integers(20, 60),
            "corrupt": rng.integers(3, 10),
        }[ftype]

        start = rng.integers(60, n - length - 60)
        if used[start - 30 : start + length + 30].any():
            continue   # overlaps another episode, skip it
        used[start : start + length] = True

        rows = idx[start : start + length]

        if ftype == "stuck":
            df.loc[rows, "temperature"] = df.loc[rows[0], "temperature"]
            df.loc[rows, "humidity"] = df.loc[rows[0], "humidity"]

        elif ftype == "spike":
            df.loc[rows, "temperature"] += rng.choice([-1, 1]) * rng.uniform(8, 15)
            df.loc[rows, "humidity"] += rng.choice([-1, 1]) * rng.uniform(15, 25)

        elif ftype == "drift":
            ramp = np.linspace(0, rng.uniform(5, 9), length)   # grows to +5..+9 °C
            df.loc[rows, "temperature"] += ramp

        elif ftype == "dropout":
            df.loc[rows, ["temperature", "humidity", "light"]] = np.nan

        elif ftype == "corrupt":
            df.loc[rows, "temperature"] = rng.choice([-999.0, 255.0, 0.0])
            df.loc[rows, "humidity"] = rng.choice([-999.0, 255.0, 0.0])

        df.loc[rows, "is_fault"] = 1
        df.loc[rows, "fault_type"] = ftype


for node in range(1, 6):
    inject(df[df.node_id == node])

df.to_csv("faulty_data.csv", index=False)

print("shape:", df.shape)
print("fault rate: {:.2f}%".format(df.is_fault.mean() * 100))
print(df[df.is_fault == 1].groupby("fault_type").size())