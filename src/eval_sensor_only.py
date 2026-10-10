import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

d = pd.read_csv("dashboard_v3.csv")
t = d[(d.split == "test") & (~d.fault_type.isin(["battery_death", "link_loss"]))]
y = t.is_fault.values

rows = []
for name in ["classical", "combined"]:
    p = t[name].values
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    rows.append({"System": name,
                    "Precision": round(precision_score(y, p, zero_division=0), 3),
                    "Recall": round(recall_score(y, p), 3),
                    "F1": round(f1_score(y, p), 3),
                    "FPR": round(fp / (fp + tn), 4),
                    "TP": tp, "FP": fp, "FN": fn})
print("Sensor-fault-only evaluation (test split):")
print(pd.DataFrame(rows).to_string(index=False))