import numpy as np
import pandas as pd

rng = np.random.default_rng(11)

df = pd.read_csv("faulty_data.csv", parse_dates=["timestamp"])
df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
T = int((df.node_id == 1).sum())

PARENT = {1: 0, 2: 0, 3: 0, 4: 2, 5: 3}     # 0 = sink
HOPS = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
CAP = {n: 1200 * rng.uniform(0.95, 1.05) for n in range(1, 6)}   # mAh
TX_COST, FWD_COST, BASE_LOSS = 0.083, 0.060, 0.02

batt = dict(CAP)
bp = {n: np.zeros(T) for n in CAP}                # battery history
lat = {n: np.full(T, np.nan) for n in CAP}        # latency ms
cause = {n: np.zeros(T, dtype=int) for n in CAP}  # 0 ok, 1 random loss, 2 dead, 3 orphaned

for t in range(T):
    for n in range(1, 6):
        if batt[n] <= 0:
            cause[n][t] = 2
            bp[n][t] = 0
            continue
        batt[n] -= TX_COST
        p = PARENT[n]
        relay_ok = (p == 0) or (batt[p] > 0)
        lost = rng.random() < BASE_LOSS
        if relay_ok and not lost:
            if p != 0:
                batt[p] -= FWD_COST            # relay pays to forward
            lat[n][t] = sum(30 + rng.exponential(15) for _ in range(HOPS[n]))
        elif not relay_ok:
            cause[n][t] = 3
        else:
            cause[n][t] = 1
        bp[n][t] = max(batt[n], 0) / CAP[n] * 100

parts = []
for n in range(1, 6):
    g = df[df.node_id == n].copy().reset_index(drop=True)
    g["battery_pct"] = bp[n].round(2)
    g["hops"] = HOPS[n]
    g["latency_ms"] = lat[n].round(1)
    g["delivered"] = (cause[n] == 0).astype(int)
    c = cause[n]
    # packets that never arrive have no sensor values at the sink
    for col in ["temperature", "humidity", "light"]:
        g.loc[c > 0, col] = np.nan
    g.loc[c == 2, ["is_fault", "fault_type"]] = [1, "battery_death"]
    g.loc[c == 3, ["is_fault", "fault_type"]] = [1, "link_loss"]
    # c == 1 (random 2% loss) keeps its original label: normal network noise, not a fault
    parts.append(g)

net = pd.concat(parts, ignore_index=True)
net.to_csv("net_data.csv", index=False)

print("shape:", net.shape)
for n in range(1, 6):
    g = net[net.node_id == n]
    dead = g[g.fault_type == "battery_death"]
    died = f"day {(dead.index[0] - g.index[0]) / 1440:.2f}" if len(dead) else "survived"
    print(f"node {n}: hops={HOPS[n]}  delivery={g.delivered.mean():.3f}  "
            f"mean latency={g.latency_ms.mean():.0f} ms  battery death: {died}")
print(net.groupby("fault_type").size())