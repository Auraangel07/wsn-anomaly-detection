import numpy as np
import pandas as pd

rng = np.random.default_rng(42)   # fixed seed = reproducible results

N_NODES = 5
DAYS = 7
MINUTES = DAYS * 24 * 60          # one reading per minute

def generate_node(node_id):
    t = np.arange(MINUTES)
    hour = (t / 60) % 24

    # Temperature: daily wave, warmest mid-afternoon, small per-node offset
    temp = 24 + 4 * np.sin((hour - 9) / 24 * 2 * np.pi) + node_id * 0.3
    temp += rng.normal(0, 0.3, MINUTES)

    # Humidity: moves opposite to temperature
    hum = 55 - 8 * np.sin((hour - 9) / 24 * 2 * np.pi) + node_id * 0.5
    hum += rng.normal(0, 1.0, MINUTES)

    # Light: daylight bump 6am-6pm, zero at night
    light = np.clip(400 * np.sin((hour - 6) / 12 * np.pi), 0, None)
    light += rng.normal(0, 8, MINUTES)
    light = np.clip(light, 0, None)

    # Motion (PIR): more likely during the day
    p_motion = 0.05 + 0.35 * (light > 50)
    motion = (rng.random(MINUTES) < p_motion).astype(int)

    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01", periods=MINUTES, freq="min"),
        "node_id": node_id,
        "temperature": temp.round(2),
        "humidity": hum.round(2),
        "light": light.round(1),
        "motion": motion,
        "is_fault": 0,
        "fault_type": "none",
    })

if __name__ == "__main__":
    df = pd.concat([generate_node(i) for i in range(1, N_NODES + 1)], ignore_index=True)
    df.to_csv("normal_data.csv", index=False)
    print(df.shape)
    print(df.groupby("node_id")[["temperature", "humidity", "light"]].mean().round(2))