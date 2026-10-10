import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("net_data.csv", parse_dates=["timestamp"])
df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
df["minute_of_day"] = df.timestamp.dt.hour * 60 + df.timestamp.dt.minute
df["raw_nan"] = df[["temperature", "humidity", "light"]].isna().any(axis=1)
cut = df.timestamp.quantile(0.6)

# Expected daily pattern per node, from clean delivered training rows only
clean = (df.timestamp <= cut) & (df.fault_type == "none") & (~df.raw_nan)
for c in ["temperature", "humidity"]:
    ref = (df[clean].groupby(["node_id", "minute_of_day"])[c].mean()
            .rename("exp_" + c).reset_index())
    df = df.merge(ref, on=["node_id", "minute_of_day"], how="left")
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

def feats(g):
    g = g.copy()
    for c in ["temperature", "humidity"]:
        f = g[c].ffill().bfill()
        g["res_" + c] = f - g["exp_" + c]
        g["res_" + c + "_m30"] = g["res_" + c].rolling(30, min_periods=1).mean()
        g["res_" + c + "_m10"] = g["res_" + c].rolling(10, min_periods=1).mean()
        g[c + "_std10"] = f.rolling(10, min_periods=1).std().fillna(0)
        g[c + "_absdiff"] = f.diff().abs().fillna(0)
    # watchdog: alarm once 5+ consecutive readings are missing
    nan = g.raw_nan.astype(int)
    run = nan.groupby((nan == 0).cumsum()).cumsum()
    g["wd"] = (run >= 5).astype(int)
    return g

df = pd.concat([feats(g) for _, g in df.groupby("node_id")], ignore_index=True)

F = ["res_temperature", "res_humidity", "res_temperature_m30", "res_humidity_m30",
        "res_temperature_m10", "res_humidity_m10", "temperature_std10", "humidity_std10",
        "temperature_absdiff", "humidity_absdiff"]

# Isolation Forest, trained on clean normal rows only (no labels)
iso = IsolationForest(n_estimators=300, random_state=42)
iso.fit(df.loc[clean, F])
cutoff = np.percentile(iso.score_samples(df.loc[clean, F]), 1)
iso_flag = (iso.score_samples(df[F]) < cutoff) & (~df.raw_nan.values)

t, h = df.temperature, df.humidity
range_rule = ((t < 15) | (t > 35) | (h < 20) | (h > 90)).values
df["classical"] = (range_rule | (df.wd.values == 1)).astype(int)
df["combined"] = (df.classical.values.astype(bool) | iso_flag).astype(int)

# ---- Overall metrics on the held-out test split ----
test = df[df.timestamp > cut]
y = test.is_fault.values
rows = []
for name in ["classical", "combined"]:
    p = test[name].values
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    rows.append({"System": name, "Precision": round(precision_score(y, p), 3),
                    "Recall": round(recall_score(y, p), 3), "F1": round(f1_score(y, p), 3),
                    "FPR": round(fp / (fp + tn), 4), "TP": tp, "FP": fp, "FN": fn})
print("\n=== OVERALL (test split) ===")
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== RECALL BY FAULT TYPE ===")
for ft in ["stuck", "spike", "drift", "dropout", "corrupt", "battery_death", "link_loss"]:
    m = (test.fault_type == ft).values
    if m.sum():
        print(f"{ft:14s} n={m.sum():5d}  classical={test.classical[m].mean():.3f}  "
                f"combined={test.combined[m].mean():.3f}")

# ---- Detection latency per fault episode (1 row = 1 minute) ----
lat_rows = []
for n, g in df.groupby("node_id"):
    g = g.reset_index(drop=True)
    f, ty = g.is_fault.values, g.fault_type.values
    i = 0
    while i < len(g):
        if f[i] == 1:
            j = i
            while j + 1 < len(g) and f[j + 1] == 1 and ty[j + 1] == ty[i]:
                j += 1
            if g.timestamp.iloc[i] > cut:
                for name in ["classical", "combined"]:
                    hit = np.flatnonzero(g[name].values[i:j + 1] == 1)
                    lat_rows.append((ty[i], name, hit[0] if len(hit) else np.nan))
            i = j + 1
        else:
            i += 1
lat = pd.DataFrame(lat_rows, columns=["fault", "method", "latency_min"])
print("\n=== EPISODE DETECTION (size=episodes, count=detected, median=minutes to first alarm) ===")
print(lat.groupby(["fault", "method"]).latency_min.agg(["size", "count", "median"]))

# ---- Low-battery early warning ----
print("\n=== LOW-BATTERY WARNING (<=10%) ===")
for n, g in df.groupby("node_id"):
    g = g.reset_index(drop=True)
    dead = g.index[g.fault_type == "battery_death"]
    if len(dead):
        warn = g.index[g.battery_pct <= 10]
        print(f"node {n}: warning fires {(dead[0] - warn[0]) / 60:.1f} h before the node dies")

df["split"] = np.where(df.timestamp > cut, "test", "train")
df[["timestamp", "node_id", "temperature", "humidity", "light", "is_fault", "fault_type",
    "battery_pct", "delivered", "latency_ms", "hops", "classical", "combined", "split"]
    ].to_csv("dashboard_v3.csv", index=False)
print("\nsaved dashboard_v3.csv")