# Real-Time Anomaly & Fault Detection in a Wireless Sensor Network (ML)

Course: Adhoc and Wireless Sensor Networks (thematic assessment)
Team: Angel Suri (231302156), Pratishtha (231302058), Himanshu Singh (231302024)

A simulated 5-node WSN (DHT22, PIR, LDR) with a sink, 2-hop relays, an energy model, packet loss and latency.
Sensor faults and network faults are injected with ground-truth labels, then detected by threshold rules,
a watchdog, Isolation Forest (no labels) and Random Forest. Hardware procurement slipped, so the full pipeline was
validated in simulation; ESP32 firmware was verified in Wokwi. Physical deployment is Phase 2.

## Pipeline
`simulator.py` -> `inject_faults.py` -> `network_layer.py` -> detectors -> dashboard
Data also flows publisher -> Mosquitto -> subscriber over MQTT (`wsn/+/data`).

## Results (held-out 40% test split, simulated data)
| Method | Precision | Recall | F1 | FPR | Labels |
|---|---|---|---|---|---|
| Threshold | 1.000 | 0.064 | 0.120 | 0.000 | No |
| Isolation Forest v2 | 0.817 | 0.743 | 0.778 | 0.020 | No |
| Random Forest | 0.981 | 0.611 | 0.753 | 0.001 | Yes |

Sensor faults only (rules + watchdog vs plus Isolation Forest): F1 0.079 -> 0.786 at 1.9% false alarms.
Drift caught in 6/6 episodes (median 30 min) vs 1/6 (209 min). Relay nodes die near day 5.9 and cut off their children.

## Run
```
pip install -r requirements.txt
python src/simulator.py && python src/inject_faults.py && python src/network_layer.py
python src/detect_v2.py
```
MQTT demo: start Mosquitto, run `src/subscriber.py`, then `src/publisher.py`.

## Limitations
All ML results use simulated data; episode counts are small (6 drift, 3 stuck); the battery model is parameterised, not measured.

`docs/` holds the presentation. `firmware/sketch.ino` is the Wokwi ESP32 node.
