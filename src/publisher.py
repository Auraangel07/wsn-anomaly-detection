import json
import csv
import os
import paho.mqtt.client as mqtt

OUT = "live_data.csv"
COLS = ["timestamp", "node_id", "temperature", "humidity", "light",
        "motion", "is_fault", "fault_type"]

new_file = not os.path.exists(OUT)
f = open(OUT, "a", newline="")
w = csv.DictWriter(f, fieldnames=COLS)
if new_file:
    w.writeheader()

count = 0

def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe("wsn/+/data")
    print("subscribed to wsn/+/data")

def on_message(client, userdata, msg):
    global count
    d = json.loads(msg.payload)
    w.writerow({k: d.get(k) for k in COLS})
    count += 1
    if count % 500 == 0:
        f.flush()
        print("received", count)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883)

try:
    client.loop_forever()
except KeyboardInterrupt:
    f.close()
    print("stopped, total received:", count)