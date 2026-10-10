import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("faulty_data.csv", parse_dates=["timestamp"])
df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

df["missing"] = df[["temperature", "humidity", "light"]].isna().any(axis=1).astype(int)
df["minute_of_day"] = df.timestamp.dt.hour * 60 + df.timestamp.dt.minute

cut = df["timestamp"].quantile(0.6)
is_train = df.timestamp <= cut

# Expected value per node and minute-of-day, learned from NORMAL training rows only
for c in ["temperature", "humidity"]:
    ref = (df[is_train & (df.is_fault == 0)]
            .groupby(["node_id", "minute_of_day"])[c].mean()
            .rename("exp_" + c).reset_index())
    df = df.merge(ref, on=["node_id", "minute_of_day"], how="left")
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

def feats(g):
    g = g.copy()
    for c in ["temperature", "humidity"]:
        g[c] = g[c].ffill().bfill()
        g["res_" + c] = g[c] - g["exp_" + c]
        g["res_" + c + "_m30"] = g["res_" + c].rolling(30, min_periods=1).mean()
        g["res_" + c + "_m10"] = g["res_" + c].rolling(10, min_periods=1).mean()
        g[c + "_std10"] = g[c].rolling(10, min_periods=1).std().fillna(0)
        g[c + "_absdiff"] = g[c].diff().abs().fillna(0)
    return g

df = pd.concat([feats(g) for _, g in df.groupby("node_id")]).sort_values(["node_id", "timestamp"]).reset_index(drop=True)

F = ["res_temperature", "res_humidity",
        "res_temperature_m30", "res_humidity_m30",
        "res_temperature_m10", "res_humidity_m10",
        "temperature_std10", "humidity_std10",
        "temperature_absdiff", "humidity_absdiff", "missing"]

train = df[df.timestamp <= cut]
test = df[df.timestamp > cut]
y = test.is_fault.values

iso = IsolationForest(n_estimators=300, random_state=42)
iso.fit(train[train.is_fault == 0][F])

# Cutoff from normal training scores: allow ~1% false alarms on healthy data
normal_scores = iso.score_samples(train[train.is_fault == 0][F])
cutoff = np.percentile(normal_scores, 1)
pred = (iso.score_samples(test[F]) < cutoff).astype(int)

tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print("=== Isolation Forest v2 ===")
print("Precision", round(precision_score(y, pred, zero_division=0), 3))
print("Recall   ", round(recall_score(y, pred), 3))
print("F1       ", round(f1_score(y, pred), 3))
print("FPR      ", round(fp / (fp + tn), 4))
print("TP/FP/FN/TN", tp, fp, fn, tn)

print("\nRecall by fault type")
for ft in ["stuck", "spike", "drift", "dropout", "corrupt"]:
    m = (test.fault_type == ft).values
    print(f"{ft:8s} n={m.sum():5d}  recall={pred[m].mean():.3f}")
    
out = test[["timestamp", "node_id", "temperature", "humidity", "light",
            "is_fault", "fault_type"]].copy()
out["iso_v2"] = pred
out["thr"] = ((test.missing == 1) | (test.temperature < 15) | (test.temperature > 35) | (test.humidity < 20) | (test.humidity > 90)).astype(int).values
out.to_csv("dashboard_data.csv", index=False)
print("saved dashboard_data.csv")