import json
import time
import pandas as pd
import paho.mqtt.client as mqtt

SPEED = 0.002   # seconds between rows (small = fast replay)
LIMIT = 3000    # rows per node to send; raise for a longer demo

df = pd.read_csv("faulty_data.csv", parse_dates=["timestamp"])
df = df.sort_values(["timestamp", "node_id"]).groupby("node_id").head(LIMIT)
df = df.sort_values(["timestamp", "node_id"])

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883)
client.loop_start()

for _, r in df.iterrows():
    payload = {
        "timestamp": str(r.timestamp),
        "node_id": int(r.node_id),
        "temperature": None if pd.isna(r.temperature) else float(r.temperature),
        "humidity": None if pd.isna(r.humidity) else float(r.humidity),
        "light": None if pd.isna(r.light) else float(r.light),
        "motion": int(r.motion),
        "is_fault": int(r.is_fault),
        "fault_type": r.fault_type,
    }
    client.publish(f"wsn/node{int(r.node_id)}/data", json.dumps(payload))
    time.sleep(SPEED)

client.loop_stop()
client.disconnect()
print("done publishing", len(df), "messages")