import pandas as pd

d = pd.read_csv("live_data.csv", parse_dates=["timestamp"])
d = d.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

# Rule layer: catches missing and garbage values instantly
d["rule_flag"] = (
    d.temperature.isna() | d.humidity.isna() | d.light.isna()
    | (d.temperature < 15) | (d.temperature > 35)
    | (d.humidity < 20) | (d.humidity > 90)
).astype(int)

tp = int(((d.rule_flag == 1) & (d.is_fault == 1)).sum())
fp = int(((d.rule_flag == 1) & (d.is_fault == 0)).sum())
fn = int(((d.rule_flag == 0) & (d.is_fault == 1)).sum())

print("Messages received over MQTT:", len(d))
print("Nodes:", sorted(d.node_id.unique()))
print("Faults in stream:", int(d.is_fault.sum()))
print("Rule layer caught:", tp, "| false alarms:", fp, "| missed:", fn)
print("\nBy fault type (rule layer recall):")
for ft in ["stuck", "spike", "drift", "dropout", "corrupt"]:
    m = d.fault_type == ft
    if m.sum():
        print(f"  {ft:8s} n={int(m.sum()):4d}  caught={d.rule_flag[m].mean():.2f}")