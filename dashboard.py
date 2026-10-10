import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="WSN Anomaly Detection", layout="wide")
st.title("WSN Fault Detection: Threshold vs Isolation Forest")

df = pd.read_csv("dashboard_data.csv", parse_dates=["timestamp"])

node = st.sidebar.selectbox("Node", sorted(df.node_id.unique()))
sensor = st.sidebar.selectbox("Sensor", ["temperature", "humidity", "light"])
method = st.sidebar.radio("Detector", ["Isolation Forest v2", "Threshold"])
col = "iso_v2" if method.startswith("Isolation") else "thr"

d = df[df.node_id == node]
days = sorted(d.timestamp.dt.date.unique())
day_range = st.sidebar.select_slider("Days", options=days, value=(days[0], days[-1]))
d = d[(d.timestamp.dt.date >= day_range[0]) & (d.timestamp.dt.date <= day_range[1])]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Readings", len(d))
c2.metric("True faults", int(d.is_fault.sum()))
c3.metric("Flagged", int(d[col].sum()))
c4.metric("Caught", int(((d[col] == 1) & (d.is_fault == 1)).sum()))

fig = go.Figure()
fig.add_trace(go.Scatter(x=d.timestamp, y=d[sensor], mode="lines",
                            name=sensor, line=dict(width=1)))
flag = d[d[col] == 1]
fig.add_trace(go.Scatter(x=flag.timestamp, y=flag[sensor], mode="markers",
                            name="flagged", marker=dict(color="red", size=5)))
missed = d[(d.is_fault == 1) & (d[col] == 0)]
fig.add_trace(go.Scatter(x=missed.timestamp, y=missed[sensor], mode="markers",
                            name="missed fault", marker=dict(color="orange", size=5, symbol="x")))
ranges = {"temperature": [10, 45], "humidity": [30, 90], "light": [0, 450]}
fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                  yaxis=dict(range=ranges[sensor]))
st.plotly_chart(fig, width="stretch")
st.caption("Y-axis clipped to the normal range; corrupt readings (-999, 255) fall outside the view but are still counted and flagged.")
st.subheader("Overall results (test set)")
try:
    st.dataframe(pd.read_csv("results_overall.csv"), use_container_width=True)
except FileNotFoundError:
    st.info("Run detect.py to generate results_overall.csv")