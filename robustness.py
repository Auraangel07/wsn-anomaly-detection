import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

base = pd.read_csv("net_data.csv", parse_dates=["timestamp"])
base = base.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
cut = base.timestamp.quantile(0.6)

def run(noise_sd, pct, seed=0):
    df = base.copy()
    rng = np.random.default_rng(seed)
    if noise_sd > 0:
        for c in ["temperature", "humidity"]:
            df[c] = df[c] + rng.normal(0, noise_sd, len(df))   # NaN stays NaN
    df["minute_of_day"] = df.timestamp.dt.hour * 60 + df.timestamp.dt.minute
    df["raw_nan"] = df[["temperature", "humidity", "light"]].isna().any(axis=1)
    clean = (df.timestamp <= cut) & (df.fault_type == "none") & (~df.raw_nan)
    for c in ["temperature", "humidity"]:
        ref = (df[clean].groupby(["node_id", "minute_of_day"])[c].mean()
               .rename("exp_" + c).reset_index())
        df = df.merge(ref, on=["node_id", "minute_of_day"], how="left")
        df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
    clean = (df.timestamp <= cut) & (df.fault_type == "none") & (~df.raw_nan)

    def feats(g):
        g = g.copy()
        for c in ["temperature", "humidity"]:
            f = g[c].ffill().bfill()
            g["res_" + c] = f - g["exp_" + c]
            g["res_" + c + "_m30"] = g["res_" + c].rolling(30, min_periods=1).mean()
            g["res_" + c + "_m10"] = g["res_" + c].rolling(10, min_periods=1).mean()
            g[c + "_std10"] = f.rolling(10, min_periods=1).std().fillna(0)
            g[c + "_absdiff"] = f.diff().abs().fillna(0)
        nan = g.raw_nan.astype(int)
        g["wd"] = (nan.groupby((nan == 0).cumsum()).cumsum() >= 5).astype(int)
        return g

    df = pd.concat([feats(g) for _, g in df.groupby("node_id")], ignore_index=True)
    F = [c for c in df.columns if c.startswith("res_") or c.endswith("_std10")
         or c.endswith("_absdiff")]
    iso = IsolationForest(n_estimators=200, random_state=42)
    iso.fit(df.loc[clean, F])
    cutoff = np.percentile(iso.score_samples(df.loc[clean, F]), pct)
    iso_flag = (iso.score_samples(df[F]) < cutoff) & (~df.raw_nan.values)
    t, h = df.temperature, df.humidity
    classical = ((t < 15) | (t > 35) | (h < 20) | (h > 90)).values | (df.wd.values == 1)
    df["combined"] = (classical | iso_flag).astype(int)

    te = df[(df.timestamp > cut) & (~df.fault_type.isin(["battery_death", "link_loss"]))]
    y, p = te.is_fault.values, te.combined.values
    fpr = ((p == 1) & (y == 0)).sum() / (y == 0).sum()
    return (round(precision_score(y, p, zero_division=0), 3),
            round(recall_score(y, p), 3), round(f1_score(y, p), 3), round(fpr, 4))

print("=== NOISE SWEEP (cutoff = 1st percentile) ===")
rows = [(sd, *run(sd, 1)) for sd in [0, 0.3, 0.6, 1.0, 1.5]]
print(pd.DataFrame(rows, columns=["extra_noise_sd", "Precision", "Recall", "F1", "FPR"]).to_string(index=False))

print("\n=== CUTOFF SWEEP (no extra noise) ===")
rows = [(p, *run(0, p)) for p in [0.5, 1, 2, 5]]
print(pd.DataFrame(rows, columns=["cutoff_pct", "Precision", "Recall", "F1", "FPR"]).to_string(index=False))