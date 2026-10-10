import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("faulty_data.csv", parse_dates=["timestamp"])
df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

# ---------- Feature engineering (per node, rolling windows) ----------
def add_features(g):
    g = g.copy()
    g["missing"] = g[["temperature", "humidity", "light"]].isna().any(axis=1).astype(int)
    # fill NaN so models can run; 'missing' flag keeps the information
    for c in ["temperature", "humidity", "light"]:
        g[c] = g[c].ffill().bfill()
    for c in ["temperature", "humidity"]:
        g[c + "_diff"] = g[c].diff().fillna(0)
        g[c + "_std10"] = g[c].rolling(10, min_periods=1).std().fillna(0)
        g[c + "_mean30"] = g[c].rolling(30, min_periods=1).mean()
        g[c + "_dev30"] = g[c] - g[c].rolling(120, min_periods=1).mean()
    hour = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60
    g["hour_sin"] = np.sin(hour / 24 * 2 * np.pi)
    g["hour_cos"] = np.cos(hour / 24 * 2 * np.pi)
    return g

# keep raw copy for the threshold baseline (needs the real NaN / garbage values)
raw = df[["temperature", "humidity", "light"]].copy()
df = df.groupby("node_id", group_keys=False).apply(add_features)

FEATURES = ["temperature", "humidity", "light", "missing",
            "temperature_diff", "humidity_diff",
            "temperature_std10", "humidity_std10",
            "temperature_mean30", "humidity_mean30",
            "temperature_dev30", "humidity_dev30",
            "hour_sin", "hour_cos"]

# ---------- Time-based split: first 60% train, last 40% test ----------
cut = df["timestamp"].quantile(0.6)
train = df[df.timestamp <= cut]
test = df[df.timestamp > cut]
y_test = test["is_fault"].values

# ---------- 1. Threshold baseline ----------
r = raw.loc[test.index]
thr_pred = (
    r.temperature.isna() | r.humidity.isna() | r.light.isna()
    | (r.temperature < 15) | (r.temperature > 35)
    | (r.humidity < 20) | (r.humidity > 90)
).astype(int).values

# ---------- 2. Isolation Forest (trained on NORMAL data only) ----------
normal_train = train[train.is_fault == 0]
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(normal_train[FEATURES])
iso_pred = (iso.predict(test[FEATURES]) == -1).astype(int)

# ---------- 3. Random Forest (supervised) ----------
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                            random_state=42, n_jobs=-1)
rf.fit(train[FEATURES], train["is_fault"])
rf_pred = rf.predict(test[FEATURES])

# ---------- Metrics ----------
def metrics(name, y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return {
        "Method": name,
        "Precision": round(precision_score(y, p, zero_division=0), 3),
        "Recall": round(recall_score(y, p), 3),
        "F1": round(f1_score(y, p), 3),
        "FPR": round(fp / (fp + tn), 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }

results = pd.DataFrame([
    metrics("Threshold", y_test, thr_pred),
    metrics("Isolation Forest", y_test, iso_pred),
    metrics("Random Forest", y_test, rf_pred),
])
print("\n=== OVERALL RESULTS (test set) ===")
print(results.to_string(index=False))

# ---------- Per fault type: recall ----------
print("\n=== RECALL BY FAULT TYPE ===")
rows = []
for ft in ["stuck", "spike", "drift", "dropout", "corrupt"]:
    m = (test.fault_type == ft).values
    if m.sum() == 0:
        continue
    rows.append({"Fault": ft, "n": int(m.sum()),
                    "Threshold": round(thr_pred[m].mean(), 3),
                    "IsoForest": round(iso_pred[m].mean(), 3),
                    "RandForest": round(rf_pred[m].mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

results.to_csv("results_overall.csv", index=False)
pd.DataFrame(rows).to_csv("results_by_fault.csv", index=False)
test.assign(thr=thr_pred, iso=iso_pred, rf=rf_pred).to_csv("test_predictions.csv", index=False)